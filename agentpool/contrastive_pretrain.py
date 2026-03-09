"""
Contrastive Pre-training Module (#5)

SimCLR-style contrastive learning to pre-train the BDQ shared encoder
before RL training begins.

Two augmented views of the same traffic observation form positive pairs;
different observations in the batch serve as negative pairs.

Pipeline:
  1. Collect observations via random policy (COLLECT_EPISODES)
  2. Pre-train encoder + projection head with NT-Xent loss
  3. Discard projection head; encoder weights already updated in eval_model
  4. Sync target_model, then start RL training

Optional auxiliary loss during RL:
  - Cosine consistency regularization on each RL batch
  - Encourages stable, meaningful encoder representations
  - Controlled by AUX_LOSS_WEIGHT (0 = disabled)

Benefits for traffic signal control:
  - Better initial state representations → faster RL convergence
  - Encoder learns traffic-invariant features (noise/occlusion robust)
  - Improved sample efficiency for multi-intersection coordination
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# ============================================================================
#  State Augmentation
# ============================================================================

class StateAugmentor:
    """
    Traffic-aware state augmentation for contrastive learning.

    Intersection state structure: [queue(12), wave(12), phase(1)] = 25 dims.
    Augmentations preserve semantic meaning while creating diverse views:
      1. Gaussian noise on continuous features (phase preserved)
      2. Random intersection-level masking (simulates sensor dropout)
      3. Per-intersection random scaling (traffic volume variance)
    """

    def __init__(self, itsx_state_dim=25, noise_std=0.1, mask_ratio=0.15):
        self.itsx_state_dim = itsx_state_dim
        self.noise_std = noise_std
        self.mask_ratio = mask_ratio

    def augment_np(self, states):
        """
        NumPy augmentation for pre-training batch.

        Args:
            states: np.ndarray (B, state_dim)
        Returns:
            augmented: np.ndarray (B, state_dim)
        """
        augmented = states.copy()
        B, D = states.shape
        num_itsx = D // self.itsx_state_dim

        # 1. Gaussian noise (skip phase feature at last dim of each intersection)
        noise = np.random.normal(0, self.noise_std, states.shape).astype(np.float32)
        for i in range(num_itsx):
            phase_idx = i * self.itsx_state_dim + (self.itsx_state_dim - 1)
            if phase_idx < D:
                noise[:, phase_idx] = 0
        augmented = augmented + noise

        # 2. Random intersection-level masking
        n_mask = max(1, int(num_itsx * self.mask_ratio))
        for b in range(B):
            mask_ids = np.random.choice(num_itsx, n_mask, replace=False)
            for mid in mask_ids:
                start = mid * self.itsx_state_dim
                end = start + self.itsx_state_dim
                augmented[b, start:end] = 0

        # 3. Per-intersection random scaling (0.8 ~ 1.2), skip phase
        for b in range(B):
            for i in range(num_itsx):
                scale = np.random.uniform(0.8, 1.2)
                start = i * self.itsx_state_dim
                end = start + self.itsx_state_dim - 1  # exclude phase
                augmented[b, start:end] *= scale

        return augmented

    @staticmethod
    def tf_augment(states, noise_std=0.1, mask_ratio=0.15):
        """
        TF-based augmentation for use inside GradientTape (auxiliary loss).

        Uses only TF ops:
          1. Gaussian noise
          2. Element-wise random dropout (inverted, preserves scale)

        Args:
            states: tf.Tensor (B, state_dim)
        Returns:
            augmented: tf.Tensor (B, state_dim)
        """
        aug = states + tf.random.normal(tf.shape(states), stddev=noise_std)
        keep_mask = tf.cast(
            tf.random.uniform(tf.shape(states)) > mask_ratio, tf.float32
        )
        aug = aug * keep_mask / tf.maximum(1.0 - mask_ratio, 1e-6)
        return aug


# ============================================================================
#  Projection Head
# ============================================================================

class ProjectionHead(layers.Layer):
    """
    MLP projection head: encoder output → contrastive embedding space.
    Architecture: Dense(hidden, ReLU) → Dense(output) → L2-normalize
    Discarded after pre-training.
    """

    def __init__(self, hidden_dim=256, output_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = layers.Dense(hidden_dim, activation='relu', name='proj_fc1')
        self.fc2 = layers.Dense(output_dim, name='proj_fc2')

    def call(self, x, training=None):
        h = self.fc1(x, training=training)
        z = self.fc2(h, training=training)
        return tf.math.l2_normalize(z, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
        })
        return config


# ============================================================================
#  NT-Xent Loss (SimCLR)
# ============================================================================

def nt_xent_loss(z_i, z_j, temperature=0.1):
    """
    Normalized Temperature-scaled Cross-Entropy Loss.

    For a batch of B samples, creates 2B embeddings:
      z_i[k] and z_j[k] are positive pairs (same observation, different augmentations)
      all other combinations are negative pairs.

    Args:
        z_i: (B, D) L2-normalized embeddings from augmentation 1
        z_j: (B, D) L2-normalized embeddings from augmentation 2
        temperature: scaling factor (lower = sharper similarity distribution)

    Returns:
        loss: scalar
    """
    batch_size = tf.shape(z_i)[0]
    z = tf.concat([z_i, z_j], axis=0)  # (2B, D)

    # Cosine similarity matrix (2B × 2B), scaled by temperature
    sim = tf.matmul(z, z, transpose_b=True) / temperature

    # Mask self-similarity (diagonal) with large negative value
    diag_mask = tf.eye(2 * batch_size) * 1e9
    sim = sim - diag_mask

    # Labels: positive pair of sample i is at position i+B, and vice versa
    labels = tf.concat([
        tf.range(batch_size, 2 * batch_size),
        tf.range(batch_size),
    ], axis=0)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=sim
    )
    return tf.reduce_mean(loss)


# ============================================================================
#  Contrastive Pre-trainer
# ============================================================================

class ContrastivePretrainer:
    """
    SimCLR-style contrastive pre-trainer for the BDQ encoder.

    Extracts encoder layers from eval_model (shared weights),
    pairs them with a temporary projection head, and trains
    using NT-Xent loss on augmented observation pairs.

    After pre-training:
      - Encoder weights in eval_model are already updated (shared layer references)
      - Projection head is discarded
      - Caller should sync target_model via set_weights()
    """

    def __init__(self, eval_model, state_dim, config, itsx_state_dim=25):
        self.state_dim = state_dim
        self.temperature = config.get("TEMPERATURE", 0.1)
        self.pretrain_lr = config.get("PRETRAIN_LR", 0.001)
        self.pretrain_epochs = config.get("PRETRAIN_EPOCHS", 50)
        self.pretrain_batch_size = config.get("PRETRAIN_BATCH_SIZE", 256)

        # Shared encoder layers (references to eval_model's layers)
        self.encoder_fc1 = eval_model.get_layer('encoder_fc1')
        self.encoder_fc2 = eval_model.get_layer('encoder_fc2')

        # Augmentor
        self.augmentor = StateAugmentor(
            itsx_state_dim=itsx_state_dim,
            noise_std=config.get("NOISE_STD", 0.1),
            mask_ratio=config.get("MASK_RATIO", 0.15),
        )

        # Temporary projection head (discarded after pre-training)
        self.projector = ProjectionHead(
            hidden_dim=256,
            output_dim=config.get("PROJECTION_DIM", 128),
            name='contrastive_projector',
        )

        # Separate optimizer for pre-training
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.pretrain_lr
        )

        self.loss_history = []

    def encode(self, states, training=False):
        """Forward through shared encoder layers."""
        h = self.encoder_fc1(states, training=training)
        return self.encoder_fc2(h, training=training)

    def train_step(self, batch_states):
        """One contrastive pre-training step."""
        view1 = tf.convert_to_tensor(
            self.augmentor.augment_np(batch_states), dtype=tf.float32
        )
        view2 = tf.convert_to_tensor(
            self.augmentor.augment_np(batch_states), dtype=tf.float32
        )

        with tf.GradientTape() as tape:
            z1 = self.projector(self.encode(view1, training=True), training=True)
            z2 = self.projector(self.encode(view2, training=True), training=True)
            loss = nt_xent_loss(z1, z2, self.temperature)

        trainable_vars = (
            self.encoder_fc1.trainable_variables
            + self.encoder_fc2.trainable_variables
            + self.projector.trainable_variables
        )
        gradients = tape.gradient(loss, trainable_vars)
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return loss.numpy()

    def pretrain(self, observation_buffer):
        """
        Run contrastive pre-training epochs.

        Args:
            observation_buffer: np.ndarray (N, state_dim)
        Returns:
            list of per-epoch average losses
        """
        N = len(observation_buffer)
        batch_size = min(self.pretrain_batch_size, N)
        steps_per_epoch = max(1, N // batch_size)

        print(
            f"  [Contrastive] Config: {N} samples, "
            f"batch={batch_size}, epochs={self.pretrain_epochs}, "
            f"steps/epoch={steps_per_epoch}, "
            f"temp={self.temperature}, lr={self.pretrain_lr}"
        )

        for epoch in range(self.pretrain_epochs):
            epoch_losses = []
            indices = np.random.permutation(N)

            for step in range(steps_per_epoch):
                start = step * batch_size
                batch_idx = indices[start:start + batch_size]
                batch = observation_buffer[batch_idx]
                loss = self.train_step(batch)
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            self.loss_history.append(avg_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  [Contrastive] Epoch {epoch + 1}/{self.pretrain_epochs}: "
                    f"loss = {avg_loss:.4f}"
                )

        print(
            f"  [Contrastive] Pre-training done. "
            f"Final loss: {self.loss_history[-1]:.4f}"
        )
        return self.loss_history
