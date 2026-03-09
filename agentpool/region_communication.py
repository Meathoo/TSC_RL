"""
Hierarchical Region Communication for RegionLight (TSC_RL)

Implements inter-region message passing using Graph Attention Networks (GAT)
to enable coordinated traffic signal control across region boundaries.

Architecture:
  Level 0  (Intra-Region)  : Each agent's BDQ network processes intersection states
  Level 1  (Inter-Region)  : RegionCoordinator runs GAT message passing on region graph
  Level 2  (Decision)      : Agent fuses comm context with local representation for Q-values

Reference: GAT (Veličković et al., 2018), CommNet (Sukhbaatar et al., 2016),
           TarMAC (Das et al., 2019)
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


# ============================================================================
#  GAT Layer for Region Graph
# ============================================================================

class RegionGATLayer(layers.Layer):
    """
    Graph Attention Network layer specialized for region-level communication.

    Attention scores are masked so only neighbouring regions (as defined by
    the adjacency matrix) contribute to each region's aggregation.

        e_ij = LeakyReLU( a_left^T · W·h_i  +  a_right^T · W·h_j )
        α_ij = softmax_j(e_ij)      (only over neighbours j ∈ N(i))
        h_i' = ELU( Σ_j  α_ij · W · h_j + bias )

    Multi-head: concatenates head outputs when concat=True.

    Args:
        head_dim:     Output dimension per head
        num_heads:    Number of attention heads
        dropout_rate: Dropout on attention coefficients
        concat:       True → output = num_heads * head_dim (concat heads)
    """

    def __init__(self, head_dim=8, num_heads=4, dropout_rate=0.1,
                 concat=True, **kwargs):
        super().__init__(**kwargs)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.concat = concat

    def build(self, input_shape):
        F_in = int(input_shape[-1])

        # Shared linear transform per head: (num_heads, F_in, head_dim)
        self.W = self.add_weight(
            name='W',
            shape=(self.num_heads, F_in, self.head_dim),
            initializer='glorot_uniform', trainable=True)

        # Attention vectors – separate left (query) and right (key)
        self.a_left = self.add_weight(
            name='a_left',
            shape=(self.num_heads, self.head_dim, 1),
            initializer='glorot_uniform', trainable=True)
        self.a_right = self.add_weight(
            name='a_right',
            shape=(self.num_heads, self.head_dim, 1),
            initializer='glorot_uniform', trainable=True)

        # Per-head bias
        self.bias = self.add_weight(
            name='bias',
            shape=(self.num_heads, self.head_dim),
            initializer='zeros', trainable=True)

        self.leaky_relu = layers.LeakyReLU(alpha=0.2)
        self.dropout_layer = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'head_dim': self.head_dim,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'concat': self.concat,
        })
        return config

    def call(self, inputs, adjacency_matrix, training=None):
        """
        Args:
            inputs:           (batch, N, F_in)
            adjacency_matrix: (N, N) float32 — 1.0 for connected, 0.0 otherwise
            training:         bool
        Returns:
            output: (batch, N, num_heads*head_dim) if concat else (batch, N, head_dim)
        """
        # inputs: (B, N, F_in) → (B, H, N, D) via einsum
        h = tf.einsum('bnf,hfd->bhnd', inputs, self.W)  # (B, H, N, D)

        # Attention scores:  e_ij = LeakyReLU( a_left^T h_i + a_right^T h_j )
        attn_left = tf.einsum('bhnd,hdk->bhnk', h, self.a_left)    # (B, H, N, 1)
        attn_right = tf.einsum('bhnd,hdk->bhnk', h, self.a_right)  # (B, H, N, 1)
        # Broadcast:  (B,H,N,1) + (B,H,1,N)  → (B,H,N,N)
        e = self.leaky_relu(attn_left + tf.transpose(attn_right, perm=[0, 1, 3, 2]))

        # Mask non-neighbours with -1e9 before softmax
        adj = tf.cast(adjacency_matrix, tf.float32)           # (N, N)
        mask = tf.reshape(adj, [1, 1, adj.shape[0], adj.shape[1]])  # (1,1,N,N)
        e = tf.where(mask > 0.5, e, tf.constant(-1e9, dtype=tf.float32))

        alpha = tf.nn.softmax(e, axis=-1)  # (B, H, N, N)
        alpha = self.dropout_layer(alpha, training=training)

        # Weighted aggregation:  h_i' = Σ_j α_ij · h_j + bias
        out = tf.einsum('bhmn,bhnd->bhmd', alpha, h)  # (B, H, N, D)
        out = out + tf.reshape(self.bias, [1, self.num_heads, 1, self.head_dim])
        out = tf.nn.elu(out)

        if self.concat:
            # (B, H, N, D) → (B, N, H*D)
            out = tf.transpose(out, perm=[0, 2, 1, 3])
            B = tf.shape(out)[0]
            out = tf.reshape(out, [B, out.shape[1], self.num_heads * self.head_dim])
        else:
            out = tf.reduce_mean(out, axis=1)  # average heads → (B, N, D)

        return out


# ============================================================================
#  Inter-Region Communication Layer  (GAT-based)
# ============================================================================

class InterRegionCommunication(layers.Layer):
    """
    Multi-round gated GAT message passing between region agents.

    Each round:
      1. GAT-based neighbour aggregation (respects adjacency graph)
      2. Gated update: agent decides how much to incorporate neighbor info
      3. LayerNorm for stability

    Args:
        message_dim:  Dimension of inter-region messages
        hidden_dim:   Output context vector dimension
        num_heads:    Number of GAT attention heads
        num_rounds:   Number of communication rounds (multi-hop propagation)
        dropout_rate: Dropout rate during training
    """

    def __init__(self, message_dim=32, hidden_dim=64, num_heads=4,
                 num_rounds=2, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.message_dim = message_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_rounds = num_rounds
        self.dropout_rate = dropout_rate

        # Message encoder: region features → compact message
        self.msg_encoder = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu', name='msg_enc_dense1'),
            layers.Dense(message_dim, activation='tanh', name='msg_enc_dense2')
        ], name='message_encoder')

        # Multi-round GAT communication layers
        head_dim = max(message_dim // num_heads, 1)

        self.gat_layers = []
        self.norm_layers = []
        self.gate_layers = []
        self.update_layers = []
        self.dropout_layers = []
        self.proj_layers = []

        for r in range(num_rounds):
            self.gat_layers.append(
                RegionGATLayer(
                    head_dim=head_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    concat=True,
                    name=f'region_gat_round_{r}'
                )
            )
            gat_out_dim = num_heads * head_dim
            # Project GAT output back to message_dim if dimensions differ
            self.proj_layers.append(
                layers.Dense(message_dim, activation='relu', name=f'gat_proj_round_{r}')
                if gat_out_dim != message_dim else None
            )
            self.norm_layers.append(
                layers.LayerNormalization(name=f'comm_norm_round_{r}')
            )
            self.gate_layers.append(
                layers.Dense(message_dim, activation='sigmoid', name=f'gate_round_{r}')
            )
            self.update_layers.append(
                layers.Dense(message_dim, activation='relu', name=f'update_round_{r}')
            )
            self.dropout_layers.append(
                layers.Dropout(dropout_rate, name=f'dropout_round_{r}')
            )

        # Message decoder: enhanced message → context vector for agent
        self.msg_decoder = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu', name='msg_dec_dense1'),
            layers.Dense(hidden_dim, name='msg_dec_dense2')
        ], name='message_decoder')

    def call(self, region_features, region_adj_matrix, training=None):
        """
        Run multi-round GAT message passing between regions.

        Args:
            region_features:   (batch, num_regions, feat_dim)
            region_adj_matrix: (num_regions, num_regions) float32
            training: bool

        Returns:
            context_vectors: (batch, num_regions, hidden_dim)
        """
        messages = self.msg_encoder(region_features)  # (B, N, message_dim)

        for r in range(self.num_rounds):
            neighbor_agg = self.gat_layers[r](
                messages, adjacency_matrix=region_adj_matrix, training=training
            )
            if self.proj_layers[r] is not None:
                neighbor_agg = self.proj_layers[r](neighbor_agg)
            neighbor_agg = self.dropout_layers[r](neighbor_agg, training=training)

            gate_input = tf.concat([messages, neighbor_agg], axis=-1)
            gate = self.gate_layers[r](gate_input)          # (B, N, message_dim)
            update = self.update_layers[r](neighbor_agg)    # (B, N, message_dim)
            messages = self.norm_layers[r](gate * update + (1.0 - gate) * messages)

        context = self.msg_decoder(messages)  # (B, N, hidden_dim)
        return context


# ============================================================================
#  Region Adjacency Graph Construction
# ============================================================================

def build_region_adjacency_matrix(itsx_assignments, proximity_threshold=2,
                                  include_self_loop=True):
    """
    Build a region-level adjacency matrix.

    Two regions are adjacent if:
      (a) They share an intersection, OR
      (b) Their center intersections have Manhattan distance ≤ proximity_threshold

    Args:
        itsx_assignments:    List[List[str]] — each inner list is intersection IDs
                             for one region (non-dummy first entry is center)
        proximity_threshold: Max Manhattan distance for spatial adjacency
        include_self_loop:   Whether each region connects to itself

    Returns:
        adj_matrix: np.ndarray (num_regions, num_regions) float32
    """
    num_regions = len(itsx_assignments)
    adj_matrix = np.zeros((num_regions, num_regions), dtype=np.float32)

    if include_self_loop:
        np.fill_diagonal(adj_matrix, 1.0)

    # --- Criterion (a): shared intersections ---
    itsx_to_regions = {}
    for region_idx, assignment in enumerate(itsx_assignments):
        for itsx_id in assignment:
            if itsx_id != 'dummy':
                itsx_to_regions.setdefault(itsx_id, []).append(region_idx)

    for itsx_id, region_indices in itsx_to_regions.items():
        for i in range(len(region_indices)):
            for j in range(i + 1, len(region_indices)):
                adj_matrix[region_indices[i], region_indices[j]] = 1.0
                adj_matrix[region_indices[j], region_indices[i]] = 1.0

    # --- Criterion (b): spatial proximity of region centers ---
    def _parse_center(assignment):
        """Extract (x, y) from first non-dummy intersection ID like 'intersection_1_2'."""
        for itsx_id in assignment:
            if itsx_id != 'dummy':
                parts = itsx_id.split('_')
                if len(parts) >= 3:
                    try:
                        return (int(parts[-2]), int(parts[-1]))
                    except ValueError:
                        pass
        return None

    centers = [_parse_center(a) for a in itsx_assignments]

    for i in range(num_regions):
        for j in range(i + 1, num_regions):
            if centers[i] is not None and centers[j] is not None:
                dist = (abs(centers[i][0] - centers[j][0]) +
                        abs(centers[i][1] - centers[j][1]))
                if dist <= proximity_threshold:
                    adj_matrix[i, j] = 1.0
                    adj_matrix[j, i] = 1.0

    return adj_matrix


# ============================================================================
#  Region Coordinator — Orchestrates Communication in Pipeline
# ============================================================================

class RegionCoordinator:
    """
    Manages inter-region communication during training and inference.

    Computes communication context (hidden_dim per region) from stacked raw
    observations, trains the communication module with a self-supervised dual
    loss (predict own reward + predict mean neighbour reward).

    Usage in the pipeline
    ---------------------
        coordinator = RegionCoordinator(num_regions, state_dim, adj_matrix, config)

        # Every step:
        all_obs = np.stack([obs[i] for i in range(num_agents)])
        context = coordinator.get_context(all_obs, training=True)
        # context: np.ndarray (num_regions, hidden_dim)

        # Pass to agent:
        action = agent.choose_action(obs[aid], idle_id, comm_context=context[aid])

        # Periodic training:
        coordinator.store(all_obs, all_rewards)
        if global_step % COMM_TRAIN_INTERVAL == 0:
            coordinator.train()
    """

    def __init__(self, num_regions, state_dim, region_adj_matrix, config=None):
        if config is None:
            config = {}

        self.num_regions = num_regions
        self.state_dim = state_dim
        self.region_adj_matrix = tf.constant(region_adj_matrix, dtype=tf.float32)

        # Hyperparameters
        self.message_dim = config.get("COMM_MESSAGE_DIM", 32)
        self.hidden_dim = config.get("COMM_HIDDEN_DIM", 64)
        self.num_heads = config.get("COMM_NUM_HEADS", 4)
        self.num_rounds = config.get("COMM_NUM_ROUNDS", 2)
        self.dropout_rate = config.get("COMM_DROPOUT_RATE", 0.1)
        self.learning_rate = config.get("COMM_LEARNING_RATE", 0.0001)
        self.comm_grad_clip = config.get("COMM_GRAD_CLIP_NORM", 5.0)
        self.comm_train_interval = config.get("COMM_TRAIN_INTERVAL", 10)
        self.comm_batch_size = config.get("COMM_BATCH_SIZE", 64)
        self.comm_buffer_size = config.get("COMM_BUFFER_SIZE", 50000)

        # State encoder: raw obs → region feature vector
        self.state_encoder = tf.keras.Sequential([
            layers.Dense(128, activation='relu', name='coord_enc_dense1'),
            layers.Dense(self.hidden_dim, activation='relu', name='coord_enc_dense2')
        ], name='region_state_encoder')

        # Communication module (State encoder output → context vectors)
        self.comm_module = InterRegionCommunication(
            message_dim=self.message_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_rounds=self.num_rounds,
            dropout_rate=self.dropout_rate
        )

        # Self-supervised heads
        # Task 1: predict own reward from comm context
        self.reward_predictor = tf.keras.Sequential([
            layers.Dense(32, activation='relu', name='rew_pred_h1'),
            layers.Dense(1, name='rew_pred_out'),
        ], name='reward_predictor')

        # Task 2: predict mean neighbour reward (forces comm to aggregate neighbour info)
        self.neighbor_reward_predictor = tf.keras.Sequential([
            layers.Dense(32, activation='relu', name='neigh_pred_h1'),
            layers.Dense(1, name='neigh_pred_out'),
        ], name='neighbor_reward_predictor')

        # Warm up all layers with a dummy forward pass
        dummy_obs = tf.zeros((1, num_regions, state_dim), dtype=tf.float32)
        dummy_features = self.state_encoder(dummy_obs)
        dummy_context = self.comm_module(dummy_features, self.region_adj_matrix, training=False)
        self.reward_predictor(dummy_context)
        self.neighbor_reward_predictor(dummy_context)

        # Row-normalized adjacency (no self-loop) for neighbour reward target
        adj_np = np.array(region_adj_matrix, dtype=np.float32)
        np.fill_diagonal(adj_np, 0.0)
        row_sum = np.maximum(adj_np.sum(axis=1, keepdims=True), 1e-6)
        self.neighbor_adj_norm = tf.constant(adj_np / row_sum, dtype=tf.float32)

        # Optimizer for communication module
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Replay buffer for self-supervised training
        self._buf_obs = np.zeros((self.comm_buffer_size, num_regions, state_dim),
                                 dtype=np.float32)
        self._buf_rewards = np.zeros((self.comm_buffer_size, num_regions), dtype=np.float32)
        self._buf_ptr = 0
        self._buf_size = 0

        self.comm_loss_his = []
        self.train_step_count = 0

        print(f"[RegionCoordinator] Initialized: {num_regions} regions, "
              f"msg_dim={self.message_dim}, hidden={self.hidden_dim}, "
              f"heads={self.num_heads}, rounds={self.num_rounds}")
        print(f"[RegionCoordinator] adj_matrix:\n{region_adj_matrix}")

    # ------------------------------------------------------------------
    #  Context computation
    # ------------------------------------------------------------------

    def get_context(self, all_observations, training=False):
        """
        Compute communication context for all regions.

        Args:
            all_observations: np.ndarray (num_regions, state_dim)
            training: bool

        Returns:
            context_vectors: np.ndarray (num_regions, hidden_dim)
        """
        obs_t = tf.convert_to_tensor(all_observations[np.newaxis, :], dtype=tf.float32)
        region_features = self.state_encoder(obs_t, training=training)
        context = self.comm_module(region_features, self.region_adj_matrix, training=training)
        return context.numpy()[0]  # (num_regions, hidden_dim)

    # ------------------------------------------------------------------
    #  Buffer & training
    # ------------------------------------------------------------------

    def store(self, all_observations, all_rewards):
        """
        Store a step's observations and rewards for self-supervised training.

        Args:
            all_observations: np.ndarray (num_regions, state_dim)
            all_rewards:      list or np.ndarray of length num_regions
        """
        idx = self._buf_ptr % self.comm_buffer_size
        self._buf_obs[idx] = all_observations
        self._buf_rewards[idx] = np.array(all_rewards, dtype=np.float32)
        self._buf_ptr += 1
        self._buf_size = min(self._buf_size + 1, self.comm_buffer_size)

    def train(self):
        """
        Run one self-supervised training step if enough data is available.

        Loss = L_self + L_neighbor
          L_self:     Predict own reward from comm context
          L_neighbor: Predict mean neighbour reward from comm context
        """
        if self._buf_size < self.comm_batch_size:
            return None

        batch_idx = np.random.choice(self._buf_size, self.comm_batch_size, replace=False)
        obs_batch = tf.convert_to_tensor(self._buf_obs[batch_idx], dtype=tf.float32)
        rew_batch = tf.convert_to_tensor(self._buf_rewards[batch_idx], dtype=tf.float32)

        train_vars = (self.state_encoder.trainable_variables +
                      self.comm_module.trainable_variables +
                      self.reward_predictor.trainable_variables +
                      self.neighbor_reward_predictor.trainable_variables)

        with tf.GradientTape() as tape:
            region_features = self.state_encoder(obs_batch, training=True)
            context = self.comm_module(region_features, self.region_adj_matrix,
                                       training=True)  # (batch, N, hidden_dim)

            # Task 1: predict own reward
            self_pred = tf.squeeze(self.reward_predictor(context, training=True), axis=-1)
            loss_self = tf.reduce_mean(tf.square(self_pred - rew_batch))

            # Task 2: predict mean neighbour reward
            mean_neigh_rew = tf.matmul(rew_batch, self.neighbor_adj_norm, transpose_b=True)
            neigh_pred = tf.squeeze(
                self.neighbor_reward_predictor(context, training=True), axis=-1)
            loss_neighbor = tf.reduce_mean(tf.square(neigh_pred - mean_neigh_rew))

            loss = loss_self + loss_neighbor

        grads = tape.gradient(loss, train_vars)
        grads_and_vars = [(g, v) for g, v in zip(grads, train_vars) if g is not None]
        if grads_and_vars:
            g_list, v_list = zip(*grads_and_vars)
            clipped_grads, _ = tf.clip_by_global_norm(list(g_list), self.comm_grad_clip)
            self.optimizer.apply_gradients(zip(clipped_grads, v_list))

        loss_val = float(loss.numpy())
        self.comm_loss_his.append(loss_val)
        self.train_step_count += 1
        return loss_val

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------

    def save(self, folder):
        """Save communication module weights to folder."""
        import os
        os.makedirs(folder, exist_ok=True)
        self.state_encoder.save_weights(os.path.join(folder, 'comm_state_encoder.weights.h5'))
        self.comm_module.save_weights(os.path.join(folder, 'comm_module.weights.h5'))
        self.reward_predictor.save_weights(os.path.join(folder, 'reward_predictor.weights.h5'))
        self.neighbor_reward_predictor.save_weights(
            os.path.join(folder, 'neighbor_reward_predictor.weights.h5'))
        if self.comm_loss_his:
            np.save(os.path.join(folder, 'comm_loss_history.npy'),
                    np.array(self.comm_loss_his))
        print(f"[RegionCoordinator] Saved to {folder} "
              f"(train_steps={self.train_step_count})")
