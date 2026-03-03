"""
CommBDQ Agent: Branching DQN with Cross-Region Attention Communication
======================================================================

基於 AdaptiveBDQ_agent 擴展，加入跨 Region 的 Attention 溝通機制。

網路架構：
    State Input (state_dim)
        → Dense(512, ReLU) → Dense(256, ReLU)  [Shared Representation]
        → Dense(comm_dim, ReLU)                 [Message Encoder]
            ↓
    Neighbor Messages (max_neighbors, comm_dim)
    Neighbor Mask (max_neighbors)
        ↓
    Scaled Dot-Product Attention (Q=own_msg, K=V=neighbor_msgs)
        → LayerNorm
        → Gating Mechanism (自適應控制通訊信任度)
        ↓
    Concatenate([Shared Repr, Gated Comm Output])
        → State Value: Dense(128, ReLU) → Dense(1)
        → Action Branch ×N: Dense(128, ReLU) → Dense(sub_action_num)
        → Q = V + (A - mean(A))

關鍵創新：
    1. Message Encoder 與 Shared Representation 共享前層，端到端可訓練
    2. Attention 讓 Agent 選擇性聚合有用的鄰居資訊
    3. Gating 機制讓 Agent 自適應調整對鄰居通訊的依賴程度
    4. LayerNorm 穩定通訊表示的分布

Replay Buffer 擴展：
    除了 (s, a, r, s')，額外儲存 (neighbor_msgs, neighbor_mask) 用於訓練
"""

import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers

from configs import agent_config
import copy

tf.config.experimental_run_functions_eagerly(True)


def build_comm_network(state_dim, comm_dim, action_dim, sub_action_num, max_neighbors):
    """
    建構帶有跨 Region Attention 通訊的 Branching DQN 網路。

    Inputs:
        state:          (batch, state_dim)           - 本地 Region 觀測
        neighbor_msgs:  (batch, max_neighbors, comm_dim) - 鄰居 Region 的訊息
        neighbor_mask:  (batch, max_neighbors)       - 有效鄰居遮罩 (1=valid, 0=pad)

    Outputs:
        q_values:       (batch, action_dim, sub_action_num) - 各分支 Q 值
        own_msg:        (batch, comm_dim)            - 本 Agent 的訊息（供鄰居使用）

    Args:
        state_dim:       local state 維度
        comm_dim:        通訊 embedding 維度
        action_dim:      action 分支數（Region 內 intersection 數）
        sub_action_num:  每個分支的動作數
        max_neighbors:   最大鄰居數（用於 padding）
    """
    print(f"Build CommBDQ network: state_dim={state_dim}, comm_dim={comm_dim}, "
          f"action_dim={action_dim}, sub_actions={sub_action_num}, max_neighbors={max_neighbors}")

    # ====== Inputs ======
    state_input = layers.Input(shape=(state_dim,), name='state')
    neighbor_msgs_input = layers.Input(shape=(max_neighbors, comm_dim), name='neighbor_msgs')
    neighbor_mask_input = layers.Input(shape=(max_neighbors,), name='neighbor_mask')

    # ====== Shared Representation (與原 BDQ 相同) ======
    shared_repr = layers.Dense(512, activation='relu', name='shared_1')(state_input)
    shared_repr = layers.Dense(256, activation='relu', name='shared_2')(shared_repr)

    # ====== Message Encoder ======
    # 從 shared representation 生成一個 compact message vector
    # 這個 message 會被發送給鄰居 Region
    own_msg = layers.Dense(comm_dim, activation='relu', name='msg_encoder')(shared_repr)

    # ====== Cross-Region Scaled Dot-Product Attention ======
    # Query: 由本 Agent 的 message 生成
    # Key, Value: 由鄰居的 messages 生成
    query = layers.Dense(comm_dim, use_bias=False, name='attn_Wq')(own_msg)       # (batch, comm_dim)
    key = layers.Dense(comm_dim, use_bias=False, name='attn_Wk')(neighbor_msgs_input)   # (batch, N, comm_dim)
    value = layers.Dense(comm_dim, use_bias=False, name='attn_Wv')(neighbor_msgs_input)  # (batch, N, comm_dim)

    # Expand query for matrix multiplication
    query_exp = tf.expand_dims(query, axis=1)   # (batch, 1, comm_dim)

    # Attention scores
    scale = tf.math.sqrt(tf.cast(comm_dim, tf.float32))
    scores = tf.matmul(query_exp, key, transpose_b=True) / scale  # (batch, 1, N)

    # Apply neighbor mask: padding 位置設為 -1e9（softmax 後趨近 0）
    mask_exp = tf.expand_dims(neighbor_mask_input, axis=1)  # (batch, 1, N)
    scores = scores + (1.0 - mask_exp) * (-1e9)

    # Softmax attention weights
    attn_weights = tf.nn.softmax(scores, axis=-1)  # (batch, 1, N)

    # Weighted aggregation of neighbor values
    comm_raw = tf.matmul(attn_weights, value)    # (batch, 1, comm_dim)
    comm_raw = tf.squeeze(comm_raw, axis=1)      # (batch, comm_dim)

    # 無有效鄰居時（全 padding），使用零向量
    has_neighbor = tf.reduce_any(tf.cast(neighbor_mask_input, tf.bool), axis=-1, keepdims=True)
    comm_raw = tf.where(has_neighbor, comm_raw, tf.zeros_like(comm_raw))

    # ====== Layer Normalization (穩定通訊表示) ======
    comm_normed = layers.LayerNormalization(name='comm_layernorm')(comm_raw)

    # ====== Gating Mechanism (自適應通訊信任度) ======
    # gate ∈ (0, 1)：控制 Agent 多大程度依賴鄰居資訊
    # 初期（messages 尚不可靠）gate 可以學到較小值；trainig 成熟後逐漸打開
    gate_input = layers.Concatenate()([shared_repr, comm_normed])
    gate = layers.Dense(comm_dim, activation='sigmoid', name='comm_gate')(gate_input)
    gated_comm = gate * comm_normed  # element-wise gating

    # ====== Augmented Representation ======
    # 結合 local representation 與 gated 鄰居通訊
    augmented = layers.Concatenate(name='augmented_repr')([shared_repr, gated_comm])

    # ====== State Value (from augmented) ======
    common_state_value = layers.Dense(128, activation='relu', name='value_hidden')(augmented)
    common_state_value = layers.Dense(1, name='value_output')(common_state_value)

    # ====== Action Branches (from augmented) ======
    subaction_q_layers = []
    for i in range(action_dim):
        branch = layers.Dense(128, activation='relu', name=f'branch_{i}_hidden')(augmented)
        branch = layers.Dense(sub_action_num, name=f'branch_{i}_output')(branch)
        # Dueling: Q = V + (A - mean(A))
        q_branch = common_state_value + (branch - tf.reduce_mean(branch))
        subaction_q_layers.append(q_branch)

    q_values = tf.stack(subaction_q_layers, axis=1)  # (batch, action_dim, sub_action_num)

    model = tf.keras.Model(
        inputs=[state_input, neighbor_msgs_input, neighbor_mask_input],
        outputs=[q_values, own_msg],
        name='CommBDQ_Network'
    )
    return model


class CommBDQ_agent:
    """
    帶有跨 Region Attention 通訊的 Branching DQN Agent。

    相比 AdaptiveBDQ_agent 的主要改動：
        1. 網路增加 Attention Communication Module
        2. Replay Buffer 額外儲存鄰居 messages
        3. choose_action 接收鄰居資訊
        4. generate_message 方法供 RegionCommCoordinator 調用
    """

    def __init__(self, env_config, comm_config=None):
        """
        Args:
            env_config: dict with ITSX_STATE_DIM, ACTION_DIM, ITSX_ACTION_DIM
            comm_config: dict with COMM_DIM, MAX_NEIGHBORS (optional)
        """
        self.state_dim = env_config["ITSX_STATE_DIM"] * env_config["ACTION_DIM"]
        self.action_dim = env_config["ACTION_DIM"]
        self.subaction_num = env_config["ITSX_ACTION_DIM"]

        # Communication parameters
        if comm_config is None:
            comm_config = {}
        self.comm_dim = comm_config.get("COMM_DIM", 64)
        self.max_neighbors = comm_config.get("MAX_NEIGHBORS", 4)

        # Build communication-enabled model
        self.eval_model = build_comm_network(
            self.state_dim, self.comm_dim, self.action_dim,
            self.subaction_num, self.max_neighbors
        )
        self.target_model = build_comm_network(
            self.state_dim, self.comm_dim, self.action_dim,
            self.subaction_num, self.max_neighbors
        )
        self.target_model.set_weights(self.eval_model.get_weights())

        # Load agent config
        AGENT_CONFIG = copy.deepcopy(agent_config.AGENT_CONFIG)
        AGENT_CONFIG.update(agent_config.BDQ_AGENT_CONFIG)
        # Update with COMM_BDQ specific config if available
        if hasattr(agent_config, 'COMM_BDQ_AGENT_CONFIG'):
            AGENT_CONFIG.update(agent_config.COMM_BDQ_AGENT_CONFIG)
        print(AGENT_CONFIG)

        self.td_operator_type = AGENT_CONFIG["TD_OPERATOR"]

        # Policy parameters (epsilon-greedy)
        self.max_epsilon = AGENT_CONFIG["MAX_EPSILON"]
        self.epsilon = self.max_epsilon
        self.min_epsilon = AGENT_CONFIG["MIN_EPSILON"]
        self.decay_steps = AGENT_CONFIG["DECAY_STEPS"]
        self.gamma = AGENT_CONFIG["GAMMA"]
        self.learn_count = 0

        # ====== Extended Replay Buffer ======
        # 額外儲存 neighbor messages 與 mask
        self.memory_counter = 0
        self.memory_size = AGENT_CONFIG["MEMORY_SIZE"]
        self.batch_size = AGENT_CONFIG["BATCH_SIZE"]

        self.state_memory = np.zeros((self.memory_size, self.state_dim))
        self.neighbor_msgs_memory = np.zeros((self.memory_size, self.max_neighbors, self.comm_dim))
        self.neighbor_mask_memory = np.zeros((self.memory_size, self.max_neighbors))
        self.action_memory = np.zeros((self.memory_size, self.action_dim))
        self.reward_memory = np.zeros((self.memory_size, 1))
        self.next_state_memory = np.zeros((self.memory_size, self.state_dim))
        self.next_neighbor_msgs_memory = np.zeros((self.memory_size, self.max_neighbors, self.comm_dim))
        self.next_neighbor_mask_memory = np.zeros((self.memory_size, self.max_neighbors))

        # Target network update
        self.replace_target_iter = AGENT_CONFIG["REPLACE_INTERVAL"]
        self.replace_count = 0
        self.replace_mode = AGENT_CONFIG["NET_REPLACE_TYPE"]
        if self.replace_mode == 'HARD':
            self.tau = 1
        else:
            self.tau = AGENT_CONFIG["TAU"]

        self.loss_his = []
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=AGENT_CONFIG["LEARNING_RATE"])

        # Print model summary
        print(f"\n[CommBDQ] Model parameter count: "
              f"{sum(p.numpy().size for p in self.eval_model.trainable_weights):,}")
        print(f"[CommBDQ] Comm dim: {self.comm_dim}, Max neighbors: {self.max_neighbors}")

    def generate_message(self, state):
        """
        從 local state 生成 message embedding，供 RegionCommCoordinator 分發給鄰居。

        Message 是 shared representation 的 compact projection，
        其權重透過 Q-loss 端到端訓練。

        Args:
            state: np.array of shape (state_dim,)

        Returns:
            message: np.array of shape (comm_dim,)
        """
        state_tensor = state[np.newaxis, :].astype(np.float32)
        # 使用 dummy neighbor data（message 只依賴 state，不依賴鄰居輸入）
        dummy_msgs = np.zeros((1, self.max_neighbors, self.comm_dim), dtype=np.float32)
        dummy_mask = np.zeros((1, self.max_neighbors), dtype=np.float32)
        _, msg = self.eval_model([state_tensor, dummy_msgs, dummy_mask], training=False)
        return msg[0].numpy()

    def choose_action(self, state, neighbor_info, idle_id=None):
        """
        結合 local state 與鄰居通訊資訊選擇動作。

        透過 Attention 機制，Agent 選擇性地聚合鄰居 Region 的有用資訊，
        提升邊界路口的協調能力。

        Args:
            state: np.array of shape (state_dim,)，本地 Region 觀測
            neighbor_info: dict with:
                'msgs': (max_neighbors, comm_dim) 鄰居 messages
                'mask': (max_neighbors,) 有效鄰居遮罩
            idle_id: list of idle branch indices（動態分支邏輯）

        Returns:
            joint_action: np.array of shape (action_dim,)
        """
        state_input = state[np.newaxis, :].astype(np.float32)
        n_msgs = neighbor_info['msgs'][np.newaxis, :].astype(np.float32)
        n_mask = neighbor_info['mask'][np.newaxis, :].astype(np.float32)

        q_values, _ = self.eval_model([state_input, n_msgs, n_mask], training=False)
        action_branch_value = q_values[0]  # (action_dim, sub_action_num)

        joint_action = []
        if np.random.random() > self.epsilon:
            # Greedy
            for i in range(self.action_dim):
                joint_action.append(np.argmax(action_branch_value[i]))
        else:
            # Random
            for i in range(self.action_dim):
                joint_action.append(np.random.randint(0, self.subaction_num))

        if idle_id is not None:
            for id in idle_id:
                joint_action[id] = -1

        return np.array(joint_action)

    def store_transition(self, s, n_msgs, n_mask, a, r, s_, n_msgs_, n_mask_):
        """
        儲存一筆 transition 至 extended replay buffer。

        除了標準 (s, a, r, s')，額外儲存通訊上下文 (neighbor_msgs, neighbor_mask)
        以便訓練時重建通訊決策過程。

        Args:
            s:       local state
            n_msgs:  neighbor messages at current step
            n_mask:  neighbor mask at current step
            a:       joint action
            r:       reward
            s_:      next local state
            n_msgs_: neighbor messages at next step
            n_mask_: neighbor mask at next step
        """
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = s
        self.neighbor_msgs_memory[index] = n_msgs
        self.neighbor_mask_memory[index] = n_mask
        self.action_memory[index] = a
        self.reward_memory[index] = r
        self.next_state_memory[index] = s_
        self.next_neighbor_msgs_memory[index] = n_msgs_
        self.next_neighbor_mask_memory[index] = n_mask_
        self.memory_counter += 1

    def learn(self):
        """
        從 replay buffer 取樣並更新網路參數。

        通訊模組（Message Encoder + Attention）透過 Q-loss 端到端訓練：
            Gradient path: Loss → Q-values → Augmented Repr → Gated Comm
                → Attention → Query(own_msg) → Message Encoder → Shared Repr

        Epsilon 更新保持與原 origin 一致的線性衰減。
        """
        available_count = min(self.memory_size, self.memory_counter)
        self.learn_count += 1

        # 10000 筆經驗後才開始訓練（與 origin 一致）
        if available_count > 10000:
            self.replace_count += 1
            batch_indices = np.random.choice(available_count, self.batch_size)

            s = tf.convert_to_tensor(self.state_memory[batch_indices], dtype=tf.float32)
            n_msgs = tf.convert_to_tensor(self.neighbor_msgs_memory[batch_indices], dtype=tf.float32)
            n_mask = tf.convert_to_tensor(self.neighbor_mask_memory[batch_indices], dtype=tf.float32)
            a = tf.convert_to_tensor(self.action_memory[batch_indices], dtype=tf.float32)
            r = tf.convert_to_tensor(self.reward_memory[batch_indices], dtype=tf.float32)
            s_ = tf.convert_to_tensor(self.next_state_memory[batch_indices], dtype=tf.float32)
            n_msgs_ = tf.convert_to_tensor(self.next_neighbor_msgs_memory[batch_indices], dtype=tf.float32)
            n_mask_ = tf.convert_to_tensor(self.next_neighbor_mask_memory[batch_indices], dtype=tf.float32)

            is_weights = tf.ones(self.batch_size, dtype=tf.float32)
            self.update_gradient(s, n_msgs, n_mask, a, r, s_, n_msgs_, n_mask_, is_weights)

            # Target network update
            if self.replace_count % self.replace_target_iter == 0:
                print("replace para")
                self.replace_para(self.target_model.variables, self.eval_model.variables)

        # Epsilon: 線性衰減
        fraction = min(float(self.learn_count) / self.decay_steps, 1)
        self.epsilon = self.max_epsilon + fraction * (self.min_epsilon - self.max_epsilon)
        self.epsilon = max(self.epsilon, 0.001)

    @tf.function
    def replace_para(self, target_var, source_var):
        for (a, b) in zip(target_var, source_var):
            a.assign(a * (1 - self.tau) + b * self.tau)

    @tf.function
    def update_gradient(self, s, n_msgs, n_mask, a, r, s_, n_msgs_, n_mask_, is_weights):
        """
        計算 TD Loss 並更新網路參數（含通訊模組）。

        梯度流向：
            Loss → Q-values
                → Augmented Representation
                    → Shared Repr (直接路徑)
                    → Gated Comm → Attention → Query(own_msg) → Msg Encoder → Shared Repr
                → Value/Advantage branches

        這確保 Message Encoder 和 Attention 的權重都透過 Q-loss 端到端更新。
        """
        with tf.GradientTape() as tape:
            # 1. Eval network forward pass（含通訊）
            q_eval, _ = self.eval_model([s, n_msgs, n_mask], training=True)

            # 2. Action mask
            eval_act_index = tf.cast(a, tf.int32)
            eval_act_index_mask = tf.one_hot(eval_act_index, self.subaction_num)

            # Idle branch mask
            idle_mask = tf.where(eval_act_index == -1, 0.0, 1.0)

            # 3. Double DQN: 用 eval 選 action，用 target 算 Q
            q_next_eval, _ = self.eval_model([s_, n_msgs_, n_mask_], training=False)
            greedy_next_action = tf.argmax(q_next_eval, axis=2)
            greedy_next_action_mask = tf.one_hot(greedy_next_action, self.subaction_num)

            q_target_s_next, _ = self.target_model([s_, n_msgs_, n_mask_], training=False)
            masked_q_target_s_next = tf.multiply(q_target_s_next, greedy_next_action_mask)

            # 4. TD operator (MEAN / MAX / NAIVE)
            if self.td_operator_type == 'MEAN':
                branch_v = tf.reduce_sum(masked_q_target_s_next, 2)
                branch_v = tf.multiply(branch_v, idle_mask)
                v_sum = tf.reduce_sum(branch_v, axis=1, keepdims=True)
                v_count = tf.reduce_sum(idle_mask, axis=1, keepdims=True)
                operator = v_sum / (v_count + 1e-8)
            elif self.td_operator_type == "MAX":
                branch_v = tf.reduce_sum(masked_q_target_s_next, 2)
                operator = tf.reduce_max(branch_v, axis=1, keepdims=True)
            elif self.td_operator_type == "NAIVE":
                operator = tf.reduce_sum(masked_q_target_s_next, 2)

            # 5. TD error and loss
            q_eval_selected = tf.reduce_sum(
                tf.multiply(q_eval, eval_act_index_mask), axis=[1, 2]
            )
            target_val = tf.squeeze(r) + self.gamma * tf.squeeze(operator)
            td_errors = target_val - q_eval_selected
            loss = tf.reduce_mean(tf.square(td_errors))

            self.loss_his.append(loss.numpy())

        # 6. Gradient update（含通訊模組參數）
        gradients = tape.gradient(loss, self.eval_model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.eval_model.trainable_variables)
        )

    def save_model(self, model_folder):
        """
        儲存模型權重。
        使用 save_weights 以相容多輸出模型中的自定義 TF 操作。
        """
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        self.eval_model.save_weights(
            os.path.join(model_folder, "eval_model_weights.h5")
        )
        # 同時保存完整模型（SavedModel 格式，更穩定）
        self.eval_model.save(
            os.path.join(model_folder, "eval_model_savedmodel"),
            save_format='tf'
        )
        print(f"[CommBDQ] Model saved to {model_folder}")
