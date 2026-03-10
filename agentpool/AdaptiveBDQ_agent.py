import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers

# import agent_config
from configs import agent_config
import copy

tf.config.experimental_run_functions_eagerly(True)


def build_network(state_dim, action_dim, sub_action_num, comm_context_dim=0):
    """
    Build Branching Dueling DQ network.

    Args:
        state_dim:        Flattened state dimension
        action_dim:       Number of action branches (intersections in region)
        sub_action_num:   Number of sub-actions per branch
        comm_context_dim: Communication context dimension from RegionCoordinator
                          (0 = no hierarchical communication)
    """
    print(f"build Branching DQ network (comm_context_dim={comm_context_dim})")
    state_input = layers.Input(shape=(state_dim,), name='state_input')
    model_inputs = [state_input]

    # Optional communication context input (Level 1 → Level 2)
    if comm_context_dim > 0:
        context_input = layers.Input(shape=(comm_context_dim,), name='comm_context_input')
        model_inputs.append(context_input)

    # Level 0 – build shared representation
    shared_representation = layers.Dense(512, activation="relu", name='encoder_fc1')(state_input)
    shared_representation = layers.Dense(256, activation="relu", name='encoder_fc2')(shared_representation)

    # Level 2 – fuse communication context (Hierarchical Region Communication)
    if comm_context_dim > 0:
        # Project comm context to match shared representation dimension
        context_processed = layers.Dense(256, activation='relu', name='comm_proj')(context_input)
        # Gate: learn how much to trust comm vs local info
        gate_input = layers.Concatenate(name='comm_gate_concat')(
            [shared_representation, context_processed]
        )
        gate = layers.Dense(256, activation='sigmoid', name='comm_gate')(gate_input)
        shared_representation = gate * shared_representation + (1.0 - gate) * context_processed

    # build common state value layer
    common_state_value = layers.Dense(128, activation="relu")(shared_representation)
    common_state_value = layers.Dense(1)(common_state_value)
    # build action branch q value layer iteratively
    subaction_q_layers = []
    for _ in range(action_dim):
        action_branch_layer = layers.Dense(128, activation="relu")(shared_representation)
        action_branch_layer = layers.Dense(sub_action_num)(action_branch_layer)
        subaction_q_value = common_state_value + (action_branch_layer - tf.reduce_mean(action_branch_layer))
        subaction_q_layers.append(subaction_q_value)
    subaction_q_layers = tf.stack(subaction_q_layers, axis=1)

    if len(model_inputs) > 1:
        model = tf.keras.Model(model_inputs, subaction_q_layers)
    else:
        model = tf.keras.Model(state_input, subaction_q_layers)
    return model

class AdaptiveBDQ_agent:
    def __init__(self,
                 env_config,
                 coordinator=None,
                 region_id=0
                 ):

        self.state_dim = env_config["ITSX_STATE_DIM"] * env_config["ACTION_DIM"]
        self.action_dim = env_config["ACTION_DIM"]
        self.subaction_num = env_config["ITSX_ACTION_DIM"]

        # Hierarchical Region Communication config
        COMM_CONFIG = agent_config.BDQ_AGENT_CONFIG.get("COMM_CONFIG", {})
        self.use_comm = COMM_CONFIG.get("ENABLED", False)
        self.comm_context_dim = COMM_CONFIG.get("COMM_HIDDEN_DIM", 64) if self.use_comm else 0
        self.e2e_freq = COMM_CONFIG.get("E2E_FREQ", 1)  # only do E2E backward every N learn steps

        # End-to-end: reference to shared coordinator and this agent's region index
        self.coordinator = coordinator if self.use_comm else None
        self.region_id = region_id

        # build model
        self.eval_model = build_network(self.state_dim, self.action_dim, self.subaction_num,
                                        comm_context_dim=self.comm_context_dim)
        self.target_model = build_network(self.state_dim, self.action_dim, self.subaction_num,
                                          comm_context_dim=self.comm_context_dim)
        self.target_model.set_weights(self.eval_model.get_weights())
        AGENT_CONFIG = copy.deepcopy(agent_config.AGENT_CONFIG)
        AGENT_CONFIG.update(agent_config.BDQ_AGENT_CONFIG)
        print(AGENT_CONFIG)
        self.td_operator_type = AGENT_CONFIG["TD_OPERATOR"]

        # policy parameters
        self.max_epsilon = AGENT_CONFIG["MAX_EPSILON"]
        self.epsilon = self.max_epsilon
        self.min_epsilon = AGENT_CONFIG["MIN_EPSILON"]
        self.decay_steps = AGENT_CONFIG["DECAY_STEPS"]

        self.gamma = AGENT_CONFIG["GAMMA"]
        self.learn_count = 0


        # memory replay
        self.memory_counter = 0
        self.memory_size = AGENT_CONFIG["MEMORY_SIZE"]
        self.batch_size = AGENT_CONFIG["BATCH_SIZE"]
        self.state_memory = np.zeros((self.memory_size, self.state_dim))
        self.action_memory = np.zeros((self.memory_size, self.action_dim))
        self.reward_memory = np.zeros((self.memory_size, 1))
        self.next_state_memory = np.zeros((self.memory_size, self.state_dim))
        # End-to-end: store step index into coordinator's shared observation buffer
        if self.use_comm:
            self.step_idx_memory = np.full((self.memory_size,), -1, dtype=np.int64)
            self.region_id_memory = np.full((self.memory_size,), self.region_id, dtype=np.int32)

        # target network update setting
        self.replace_target_iter = AGENT_CONFIG["REPLACE_INTERVAL"]
        self.replace_count = 0
        self.replace_mode = AGENT_CONFIG["NET_REPLACE_TYPE"]
        if self.replace_mode == 'HARD':
            self.tau = 1
        else:
            self.tau = AGENT_CONFIG["TAU"]
        self.loss_his = []
        #
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=AGENT_CONFIG["LEARNING_RATE"])
        # print()

    def _build_inputs(self, state_tensor, comm_context_tensor=None):
        """Construct model input(s) depending on whether comm is enabled."""
        if self.use_comm:
            if comm_context_tensor is None:
                B = tf.shape(state_tensor)[0]
                comm_context_tensor = tf.zeros((B, self.comm_context_dim), dtype=tf.float32)
            return [state_tensor, comm_context_tensor]
        return state_tensor

    def choose_action(self, state, idle_id=None, comm_context=None):
        """
        Choose action given current state (and optional comm context).

        :param state:        1-D numpy array of shape (state_dim,)
        :param idle_id:      list of branch indices that are idle (action=-1)
        :param comm_context: 1-D numpy array (comm_context_dim,) from RegionCoordinator
        :return: numpy array of joint actions
        """
        state = state[np.newaxis, :]  # (1, state_dim)
        ctx = None
        if self.use_comm and comm_context is not None:
            ctx = tf.convert_to_tensor(comm_context[np.newaxis, :], dtype=tf.float32)
        inputs = self._build_inputs(tf.convert_to_tensor(state, dtype=tf.float32), ctx)
        action_branch_value = self.eval_model(inputs)[0]
        joint_action = []
        if np.random.random() > self.epsilon:
            # greedy choice
            for i in range(self.action_dim):
                joint_action.append(np.argmax(action_branch_value[i]))
        else:
            # random
            for i in range(self.action_dim):
                joint_action.append(np.random.randint(0, self.subaction_num))
        if idle_id is not None:
            #
            for id in idle_id:
                joint_action[id] = -1
        return np.array(joint_action)

    def learn(self):
        available_count = min(self.memory_size, self.memory_counter)
        self.learn_count += 1
        
        # 這裡保持與 origin 一致的門檻（10000 筆才開始練）
        if available_count > 10000:
            self.replace_count += 1
            batch_indices = np.random.choice(available_count, self.batch_size)

            # 改進點：轉換 Tensor 的方式與 bk 對齊，確保 float32 型態
            s = tf.convert_to_tensor(self.state_memory[batch_indices], dtype=tf.float32)
            a = tf.convert_to_tensor(self.action_memory[batch_indices], dtype=tf.float32)
            r = tf.convert_to_tensor(self.reward_memory[batch_indices], dtype=tf.float32)
            s_ = tf.convert_to_tensor(self.next_state_memory[batch_indices], dtype=tf.float32)

            # End-to-end: fetch all_obs from coordinator's shared buffer
            all_obs_batch = None
            all_next_obs_batch = None
            region_ids = None
            if self.use_comm and self.coordinator is not None:
                step_indices = self.step_idx_memory[batch_indices]
                all_obs_batch, all_next_obs_batch = self.coordinator.get_step_obs(step_indices)
                region_ids = self.region_id_memory[batch_indices]

            # 呼叫 update_gradient (這裡傳入全 1 的權重，因為不使用 PER)
            is_weights = tf.ones(self.batch_size, dtype=tf.float32)
            do_e2e = (self.learn_count % self.e2e_freq == 0)
            self.update_gradient(s, a, r, s_, is_weights,
                                all_obs_batch=all_obs_batch,
                                all_next_obs_batch=all_next_obs_batch,
                                region_ids=region_ids,
                                e2e=do_e2e)

            # 目標網路更新
            if self.replace_count % self.replace_target_iter == 0:
                print("replace para")
                self.replace_para(self.target_model.variables, self.eval_model.variables)
                
        # Epsilon 更新邏輯保持 origin 的線性衰減
        fraction = min(float(self.learn_count) / self.decay_steps, 1)
        self.epsilon = self.max_epsilon + fraction * (self.min_epsilon - self.max_epsilon)
        self.epsilon = max(self.epsilon, 0.001)

    @tf.function
    def replace_para(self, target_var, source_var):
        for (a, b) in zip(target_var, source_var):
            a.assign(a * (1 - self.tau) + b * self.tau)

    def update_gradient(self, s, a, r, s_, is_weights,
                        all_obs_batch=None, all_next_obs_batch=None,
                        region_ids=None, e2e=True):
        """
        Compute and apply gradients with end-to-end communication.

        When comm is enabled, context is recomputed from all_obs within the
        GradientTape so that TD loss gradients flow back to the coordinator's
        state_encoder and communication module (GAT or MeanField).
        Uses context caching to avoid redundant computation in DECENTRAL mode.

        Args:
            e2e: If True, compute context inside tape (full E2E gradient).
                 If False, compute context outside tape (frozen, faster).
        """
        # Determine trainable variables (BDQ + coordinator if end-to-end)
        train_vars = list(self.eval_model.trainable_variables)
        if self.use_comm and self.coordinator is not None and e2e:
            train_vars += self.coordinator.get_comm_trainable_variables()

        # Pre-compute frozen context when not doing E2E
        comm_ctx_frozen = None
        next_comm_ctx_frozen = None
        if (self.use_comm and self.coordinator is not None
                and all_obs_batch is not None and not e2e):
            all_obs_tf = tf.convert_to_tensor(all_obs_batch, dtype=tf.float32)
            all_next_obs_tf = tf.convert_to_tensor(all_next_obs_batch, dtype=tf.float32)
            all_context = tf.stop_gradient(
                self.coordinator.compute_context_tf(all_obs_tf, training=False))
            next_all_context = tf.stop_gradient(
                self.coordinator.compute_context_tf(all_next_obs_tf, training=False))
            batch_idx = tf.range(tf.shape(all_context)[0])
            region_ids_tf = tf.cast(tf.convert_to_tensor(region_ids), tf.int32)
            indices = tf.stack([batch_idx, region_ids_tf], axis=1)
            comm_ctx_frozen = tf.gather_nd(all_context, indices)
            next_comm_ctx_frozen = tf.gather_nd(next_all_context, indices)

        with tf.GradientTape() as tape:
            # ---- Compute comm context ----
            comm_ctx = None
            next_comm_ctx = None
            if self.use_comm and self.coordinator is not None and all_obs_batch is not None:
                if e2e:
                    # Full E2E: recompute inside tape for gradient flow
                    all_obs_tf = tf.convert_to_tensor(all_obs_batch, dtype=tf.float32)
                    all_context = self.coordinator.compute_context_tf(
                        all_obs_tf, training=True)
                    batch_idx = tf.range(tf.shape(all_context)[0])
                    region_ids_tf = tf.cast(
                        tf.convert_to_tensor(region_ids), tf.int32)
                    indices = tf.stack([batch_idx, region_ids_tf], axis=1)
                    comm_ctx = tf.gather_nd(all_context, indices)

                    all_next_obs_tf = tf.convert_to_tensor(
                        all_next_obs_batch, dtype=tf.float32)
                    next_all_context = self.coordinator.compute_context_tf(
                        all_next_obs_tf, training=False)
                    next_comm_ctx = tf.stop_gradient(
                        tf.gather_nd(next_all_context, indices))
                else:
                    # Frozen context (pre-computed outside tape)
                    comm_ctx = comm_ctx_frozen
                    next_comm_ctx = next_comm_ctx_frozen

            # Build model inputs (with fresh, differentiable comm context)
            eval_inputs = self._build_inputs(s, comm_ctx)
            next_inputs_eval = self._build_inputs(s_, next_comm_ctx)
            next_inputs_target = self._build_inputs(s_, next_comm_ctx)

            # 1. 取得 Eval 網路輸出，指定 training=True
            q_eval = self.eval_model(eval_inputs, training=True)
            
            # 2. 構建 Mask
            eval_act_index = tf.cast(a, tf.int32)
            eval_act_index_mask = tf.one_hot(eval_act_index, self.subaction_num)
            
            # 處理閒置分支 (Idle branches)
            idle_mask = tf.where(eval_act_index == -1, 0.0, 1.0)

            # 3. 計算 Target Q (Double DQN)
            q_next_eval = self.eval_model(next_inputs_eval, training=False)
            greedy_next_action = tf.argmax(q_next_eval, axis=2)
            greedy_next_action_mask = tf.one_hot(greedy_next_action, self.subaction_num)

            q_target_s_next = self.target_model(next_inputs_target, training=False)
            masked_q_target_s_next = tf.multiply(q_target_s_next, greedy_next_action_mask)
            
            # 根據 TD 算子類型計算 V(s')
            if self.td_operator_type == 'MEAN':
                branch_v = tf.reduce_sum(masked_q_target_s_next, 2)
                branch_v = tf.multiply(branch_v, idle_mask)
                v_sum = tf.reduce_sum(branch_v, axis=1, keepdims=True)
                v_count = tf.reduce_sum(idle_mask, axis=1, keepdims=True)
                operator = v_sum / (v_count + 1e-8) # 避免除以零
            elif self.td_operator_type == "MAX":
                branch_v = tf.reduce_sum(masked_q_target_s_next, 2)
                operator = tf.reduce_max(branch_v, axis=1, keepdims=True)
            elif self.td_operator_type == "NAIVE":
                operator = tf.reduce_sum(masked_q_target_s_next, 2)
            
            # 4. 計算 TD Error 與 Loss
            # 這裡與 bk 一致：只計算特定 action 分支的 q_eval
            q_eval_selected = tf.reduce_sum(tf.multiply(q_eval, eval_act_index_mask), axis=[1, 2])
            
            # 計算 Target (R + Gamma * V)
            # 注意：若為多分支，r 通常是共用的
            target_val = tf.squeeze(r) + self.gamma * tf.squeeze(operator)
            
            # 計算 MSE Loss
            td_errors = target_val - q_eval_selected
            loss = tf.reduce_mean(tf.square(td_errors)) 
            
            self.loss_his.append(loss.numpy())

        # 5. 套用梯度 (含 coordinator 的 end-to-end 梯度)
        gradients = tape.gradient(loss, train_vars)
        grads_and_vars = [(g, v) for g, v in zip(gradients, train_vars)
                          if g is not None]
        if grads_and_vars:
            g_list, v_list = zip(*grads_and_vars)
            self.optimizer.apply_gradients(zip(g_list, v_list))

        # Invalidate coordinator cache after weights change
        if self.use_comm and self.coordinator is not None:
            self.coordinator.invalidate_cache()

    def store_transition(self, s, a, r, s_, step_idx=-1, region_id=None):
        """
        Store a transition.

        For end-to-end comm: step_idx is the index into coordinator's shared
        observation buffer; region_id identifies which region this transition
        belongs to.
        """
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = s
        self.action_memory[index] = a
        self.reward_memory[index] = r
        self.next_state_memory[index] = s_
        if self.use_comm:
            self.step_idx_memory[index] = step_idx
            if region_id is not None:
                self.region_id_memory[index] = region_id
        self.memory_counter += 1


    def save_model(self, model_folder):
        """
        save critic and actor model in model_folder
        critic model is saved in model_folder/critic.h5
        actor model is saved in model_folder/actor.h5
        model file type is .h5
        :param model_folder: relative path to model folder
        :return:
        """
        # save critic model
        self.eval_model.save(os.path.join(model_folder, "eval_model.h5"))

