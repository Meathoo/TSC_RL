# Hierarchical Region Communication

**跨區域通訊模組 — GAT / MeanField**

為 RegionLight 交通號誌控制系統實現的層級式區域通訊機制。支援兩種通訊方式：

| 模式 | 說明 | 速度 |
|------|------|------|
| **MeanField** (預設) | 以鄰接矩陣做加權平均聚合，無 attention | 快 (5-10×) |
| **GAT** | Multi-head Graph Attention Network | 慢 |

透過 `COMM_CONFIG["COMM_TYPE"]` 切換：`"MEANFIELD"` 或 `"GAT"`。

---

## 目錄

- [架構總覽](#架構總覽)
- [Level 0 — Intra-Region（區域內）](#level-0--intra-region區域內)
- [Level 1 — Inter-Region（跨區域 GAT 通訊）](#level-1--inter-region跨區域-gat-通訊)
  - [1. State Encoder](#1-state-encoder)
  - [2. Message Encoder](#2-message-encoder)
  - [3. Multi-Round GAT Communication](#3-multi-round-gat-communication)
  - [4. Message Decoder](#4-message-decoder)
- [Level 2 — Decision（融合決策）](#level-2--decision融合決策)
- [Region Adjacency Matrix](#region-adjacency-matrix)
- [自監督訓練機制](#自監督訓練機制)
- [資料流完整路徑](#資料流完整路徑)
- [超參數配置](#超參數配置)
- [程式碼結構](#程式碼結構)

---

## 架構總覽

系統採用三層層級式架構，從區域內獨立決策到跨區域協調：

```
┌─────────────────────────────────────────────────────────────────────┐
│  Level 2 — Decision (融合決策)                                      │
│    gate * local_repr + (1-gate) * comm_context → Q-values           │
├─────────────────────────────────────────────────────────────────────┤
│  Level 1 — Inter-Region (GAT / MeanField 通訊)                      │
│    State Encoder → Message Encoder → Aggregation × N rounds → Decoder │
├─────────────────────────────────────────────────────────────────────┤
│  Level 0 — Intra-Region (區域內)                                    │
│    state_input → encoder_fc1(512) → encoder_fc2(256) → shared_repr  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Level 0 — Intra-Region（區域內）

每個 region agent 的 BDQ 網路獨立處理自己負責的路口觀測：

```
state_input (state_dim)
    │
    ▼
encoder_fc1: Dense(512, ReLU)
    │
    ▼
encoder_fc2: Dense(256, ReLU)
    │
    ▼
shared_representation (256-dim)
```

- **輸入**：該 region 所有路口的拼接觀測（每個路口 25 維：12 queue + 12 wave + 1 phase）
- **輸出**：256 維的共享表示向量
- **程式碼位置**：`AdaptiveBDQ_agent.py` 的 `build_network()` 函式

每個 region agent 獨立擁有自己的 BDQ 網路，Level 0 不涉及任何跨區域資訊交換。

---

## Level 1 — Inter-Region（跨區域 GAT 通訊）

由 `RegionCoordinator` 統籌，負責讓相鄰區域的 agent 互相交換壓縮訊息。整個 Level 1 由 4 個子模組串接：

### 1. State Encoder

將每個 region 的原始觀測壓縮成緊湊的特徵向量。

```
raw_obs (state_dim)
    │
    ▼
Dense(128, ReLU)    ← coord_enc_dense1
    │
    ▼
Dense(64, ReLU)     ← coord_enc_dense2
    │
    ▼
region_features (64-dim)
```

- **輸入**：`(num_regions, state_dim)` — 所有 region 的原始觀測
- **輸出**：`(num_regions, 64)` — 壓縮後的區域特徵
- **程式碼位置**：`region_communication.py` — `RegionCoordinator.__init__()` 中的 `self.state_encoder`

### 2. Message Encoder

將區域特徵進一步壓縮為通訊訊息，`tanh` 激活確保訊息值域限制在 $[-1, 1]$。

```
region_features (64-dim)
    │
    ▼
Dense(64, ReLU)     ← msg_enc_dense1
    │
    ▼
Dense(32, tanh)     ← msg_enc_dense2
    │
    ▼
messages (32-dim)
```

- **輸入**：`(num_regions, 64)` — 區域特徵
- **輸出**：`(num_regions, 32)` — 壓縮的訊息向量
- **程式碼位置**：`region_communication.py` — `InterRegionCommunication.__init__()` 中的 `self.msg_encoder`

### 3. Multi-Round GAT Communication

**這是整個通訊機制的核心。** 預設執行 2 輪通訊，每一輪包含 GAT 聚合 + Gated Update + LayerNorm。

#### 單輪流程：

```
messages (32-dim)
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
RegionGATLayer                     (skip connection)
    │                                  │
    ▼                                  │
[Optional projection → 32-dim]        │
    │                                  │
    ▼                                  │
Dropout                                │
    │                                  │
    ▼                                  │
concat([messages, neighbor_agg])       │
    │                                  │
    ▼                                  │
Gate = Dense(32, sigmoid)              │
    │                                  │
    ▼                                  ▼
messages' = LayerNorm(gate × update + (1-gate) × messages)
```

#### 3.1 RegionGATLayer 詳細

標準 Graph Attention Network，核心公式：

$$e_{ij} = \text{LeakyReLU}\left( \mathbf{a}_L^\top \, \mathbf{W} \mathbf{h}_i \;+\; \mathbf{a}_R^\top \, \mathbf{W} \mathbf{h}_j \right)$$

其中 $\mathbf{W} \in \mathbb{R}^{F_{in} \times D}$ 是共享線性變換，$\mathbf{a}_L, \mathbf{a}_R \in \mathbb{R}^{D \times 1}$ 分別是 query（左）和 key（右）的注意力向量。

**Masked Softmax**：非鄰居的注意力分數設為 $-10^9$，再做 softmax，確保只聚合鄰居資訊。

$$\alpha_{ij} = \text{softmax}_j(e_{ij}) \quad \text{only over } j \in \mathcal{N}(i)$$

**加權聚合**：

$$\mathbf{h}_i' = \text{ELU}\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \, \mathbf{W} \mathbf{h}_j + \mathbf{b} \right)$$

**Multi-head**：4 個 attention head，每個 head 輸出 8 維，concat 後 = 32 維（等於 `message_dim`）。

| 參數 | 值 | 說明 |
|------|-----|------|
| `head_dim` | 8 | 每個 head 的輸出維度 |
| `num_heads` | 4 | 注意力頭數 |
| 輸出形狀 | 32-dim | `4 heads × 8 dim = 32 = message_dim` |
| LeakyReLU α | 0.2 | 標準 GAT 設定 |
| Dropout | 0.1 | 應用在 attention coefficients 上 |

**程式碼位置**：`region_communication.py` 的 `RegionGATLayer` 類別

#### 3.2 Gated Update

Agent 自主決定要聽取多少鄰居資訊。Gate 機制讓每個 region 根據當前情況動態調整信任比例：

```python
gate_input = concat([messages, neighbor_agg])       # (N, 64)
gate = Dense(32, activation='sigmoid')(gate_input)   # (N, 32) ∈ [0, 1]
update = Dense(32, activation='relu')(neighbor_agg)  # (N, 32)

messages = LayerNorm(gate * update + (1 - gate) * messages)
```

- **`gate → 1`**：完全採納鄰居更新（信任跨區域資訊）
- **`gate → 0`**：保留自己原本的訊息（依賴本地資訊）

#### 3.3 多輪通訊（Multi-hop）

預設 `COMM_NUM_ROUNDS = 2`，表示執行 2 輪以上流程。

- **第 1 輪**：每個 region 接收直接鄰居（1-hop）的訊息
- **第 2 輪**：接收到經過鄰居整合後的訊息，等效於 2-hop 傳播

```
Region A ──(round 1)──► Region B ──(round 2)──► Region C
```

即使 A 和 C 不直接相鄰，B 在第 1 輪聽取了 A 的訊息後，在第 2 輪會將整合過的資訊傳給 C。

### 4. Message Decoder

將經過多輪 GAT 通訊的增強訊息轉換為可注入 agent 決策的 context 向量。

```
final_messages (32-dim)
    │
    ▼
Dense(64, ReLU)     ← msg_dec_dense1
    │
    ▼
Dense(64)           ← msg_dec_dense2
    │
    ▼
context_vectors (64-dim)
```

- **輸入**：`(num_regions, 32)` — 經過多輪通訊的訊息
- **輸出**：`(num_regions, 64)` — 通訊 context 向量
- **程式碼位置**：`region_communication.py` — `InterRegionCommunication.__init__()` 中的 `self.msg_decoder`

---

## Level 2 — Decision（融合決策）

在 `build_network()` 中，通訊 context 與 Level 0 的 local representation 融合。

```
shared_representation (256-dim)         context_vectors (64-dim)
        │                                       │
        │                               comm_proj: Dense(256, ReLU)
        │                                       │
        │                               context_processed (256-dim)
        │                                       │
        ├───────── concat ──────────────────────┤
        │                                       │
        ▼                                       │
comm_gate: Dense(256, sigmoid)                  │
        │                                       │
        ▼                                       ▼
output = gate × shared_representation + (1 - gate) × context_processed
        │
        ▼
     (256-dim) → State Value + Action Branches → Q-values
```

Gate 機制讓 agent 自主學習：在什麼情況下該依賴本地資訊（`gate → 1`）、什麼時候該信任跨區域通訊（`gate → 0`）。

**程式碼位置**：`AdaptiveBDQ_agent.py` — `build_network()` 中標有 `# Fuse communication context` 的區塊

---

## Region Adjacency Matrix

決定哪些 region 之間可以通訊。兩個 region 被視為鄰居的條件：

### 判定規則

**(a) 共享路口**：如果兩個 region 的路口分配中有相同的路口 ID，則互為鄰居。

**(b) 空間鄰近**：兩個 region 中心路口之間的曼哈頓距離 $\leq$ `PROXIMITY_THRESHOLD`（預設 = 2）。

$$d(R_i, R_j) = |x_i - x_j| + |y_i - y_j| \leq 2$$

**(c) 自迴圈**：每個 region 連接自己（`include_self_loop = True`），確保 GAT 在聚合時也考慮自身資訊。

### 範例：Hangzhou 4×4

對於 Hangzhou 4×4 路網（16 個路口分成 4 個 region），產生的鄰接矩陣為：

```
     R0  R1  R2  R3
R0 [  1   1   1   0 ]
R1 [  1   1   0   1 ]
R2 [  1   0   1   1 ]
R3 [  0   1   1   1 ]
```

**程式碼位置**：`region_communication.py` — `build_region_adjacency_matrix()` 函式

---

## 自監督訓練機制

`RegionCoordinator` 使用雙目標自監督損失函式來訓練通訊模組（作為 End-to-End 主訓練的輔助）：

### 訓練目標

$$\mathcal{L} = \mathcal{L}_{\text{self}} + \mathcal{L}_{\text{neighbor}}$$

| 目標 | 公式 | 目的 |
|------|------|------|
| $\mathcal{L}_{\text{self}}$ | 從 comm context 預測自己的 reward | 確保 context 編碼了有用的本地資訊 |
| $\mathcal{L}_{\text{neighbor}}$ | 從 comm context 預測鄰居平均 reward | **迫使 GAT 聚合鄰居資訊**（否則無法預測） |

### 訓練流程

1. 維護一個 `(obs, reward)` 的 replay buffer（容量 50,000）
2. 每 `COMM_TRAIN_INTERVAL = 10` 個 global step 訓練一次
3. 從 buffer 隨機抽取 `COMM_BATCH_SIZE = 64` 筆資料
4. 計算雙目標損失，gradient clipping（norm = 5.0），Adam 優化

**關鍵設計**：$\mathcal{L}_{\text{neighbor}}$ 使用 row-normalized adjacency matrix（移除 self-loop）計算鄰居平均 reward。這確保通訊模組必須透過 GAT 訊息傳遞來獲取鄰居資訊，而非僅依賴自身觀測。

**程式碼位置**：`region_communication.py` — `RegionCoordinator.train()`

---

## End-to-End 訓練機制

**核心改進**：通訊模組（State Encoder + GAT Comm Module）的梯度直接由 BDQ 的 TD loss 驅動，讓 communication context 對 Q-value 估算有直接影響。

### 問題動機

先前版本中，通訊 context 在 action selection 時計算後轉為 numpy 存入 replay buffer，BDQ 訓練時直接使用存好的 context 作為固定輸入。這導致：

1. **梯度斷裂**：TD loss 的梯度無法回傳到通訊模組（State Encoder / GAT）
2. **Context 過時**：隨著通訊模組持續更新，replay buffer 中的舊 context 與當前模組不一致
3. **Gate 退化**：BDQ 的 gate 傾向學成 `gate ≈ 1`，完全忽略 comm context

### 解決方案

在 `update_gradient()` 內部，**於 GradientTape 內重新計算 context**：

```
                    GradientTape
┌───────────────────────────────────────────────────────────┐
│  all_obs ──► State Encoder ──► GAT Comm Module ──► context │
│                                                     │      │
│  state ──► BDQ encoder ──► gate(local, context) ──► Q(s,a) │
│                                                     │      │
│                                    TD loss ◄────────┘      │
└───────────────────────────────────────────────────────────┘
                        │
             gradients flow to:
          eval_model + state_encoder + GAT
```

### 實作細節

1. **共享觀測 buffer**：`RegionCoordinator.store_step(all_obs, all_next_obs)` 將每個 step 的全域觀測存入共享 ring buffer，各 agent 只需記錄 `step_idx`
2. **Agent 持有 coordinator 引用**：`AdaptiveBDQ_agent(env_config, coordinator=coord, region_id=i)`
3. **Tape 內重算**：`update_gradient()` 從共享 buffer 取回 `all_obs`，透過 `coordinator.compute_context_tf()` 在 tape 內計算可微分的 context
4. **Target 梯度隔離**：next-state 的 context 使用 `tf.stop_gradient`，與 DQN 標準一致
5. **Joint 梯度更新**：`train_vars = eval_model.vars + coordinator.state_encoder.vars + coordinator.comm_module.vars`

### 梯度路徑

$$\nabla_{\theta_{\text{BDQ}}, \theta_{\text{enc}}, \theta_{\text{GAT}}} \; \mathcal{L}_{\text{TD}} = \nabla \left[ r + \gamma Q_{\text{target}}(s') - Q_{\text{eval}}(s, a; \text{context}(\theta_{\text{enc}}, \theta_{\text{GAT}})) \right]^2$$

**程式碼位置**：`AdaptiveBDQ_agent.py` — `update_gradient()`

---

## 資料流完整路徑

以下是一個 training step 的完整資料流：

```
Step 1: 環境觀測
    env.step() → next_states → assign_state() → obs[num_agents] (raw observations)

Step 2: Level 1 通訊 — Action Selection (numpy, no gradient)
    all_obs = stack(obs)                                    # (N, state_dim)
    context = coordinator.get_context(all_obs)              # (N, 64)
    agent.choose_action(obs[aid], comm_context=context[aid])

Step 3: 儲存 Transition
    step_idx = coordinator.store_step(all_obs, all_next_obs)  # 共享 buffer
    agent.store_transition(obs, action, reward, next_obs, step_idx=step_idx, region_id=aid)

Step 4: End-to-End 訓練 (每 LEARNING_INTERVAL 步)
    batch = sample(replay_buffer)
    all_obs_batch = coordinator.get_step_obs(batch.step_idx)  # 從共享 buffer 取回

    ┌─── GradientTape ──────────────────────────────────────────────┐
    │  context = coordinator.compute_context_tf(all_obs_batch)      │
    │  comm_ctx = context[:, region_id, :]                          │
    │  Q(s,a) = eval_model([state, comm_ctx])                      │
    │  TD_loss = (r + γ·Q_target(s') − Q(s,a))²                    │
    └───────────────────────────────────────────────────────────────┘
    gradients → eval_model + state_encoder + comm_module (GAT or MeanField)

Step 5: 自監督輔助訓練 (每 10 步)
    RegionCoordinator.train(obs_batch, reward_batch)
    → L_self + L_neighbor (額外正則化)
```

---

## 超參數配置

所有超參數在 `configs/agent_config.py` 的 `COMM_CONFIG` 中設定：

```python
"COMM_CONFIG": {
    "ENABLED": False,               # 是否啟用跨區域通訊
    "COMM_TYPE": "MEANFIELD",       # "MEANFIELD" (快速) 或 "GAT" (注意力)
    "COMM_MESSAGE_DIM": 32,         # 區域間訊息壓縮維度
    "COMM_HIDDEN_DIM": 64,          # context 向量維度 (也是 State Encoder 輸出維度)
    "COMM_NUM_HEADS": 4,            # GAT 注意力頭數 (僅 COMM_TYPE="GAT" 時使用)
    "COMM_NUM_ROUNDS": 2,           # 通訊輪次 (multi-hop 傳播距離)
    "COMM_DROPOUT_RATE": 0.1,       # GAT attention dropout (僅 GAT)
    "COMM_LEARNING_RATE": 0.0001,   # 通訊模組 Adam 學習率
    "PROXIMITY_THRESHOLD": 2,       # 區域鄰接的曼哈頓距離閾值
    # --- 自監督訓練 ---
    "COMM_GRAD_CLIP_NORM": 5.0,     # gradient clipping (global norm)
    "COMM_TRAIN_INTERVAL": 10,      # 每 N 個 step 訓練一次
    "COMM_BATCH_SIZE": 64,          # 訓練 batch size
    "COMM_BUFFER_SIZE": 50000,      # replay buffer 容量
}
```

### MeanField vs GAT

| 項目 | MeanField | GAT |
|------|-----------|-----|
| 聚合方式 | 鄰接矩陣正規化 × matmul | Multi-head attention (einsum) |
| 每輪計算量 | O(N²·D) — 一次矩陣乘法 | O(N²·D·H) — attention + softmax |
| 可學習參數 | Dense(agg) + Dense(gate) + Dense(update) | W_Q, W_K, W_V × H heads + a_left, a_right |
| 額外開銷 | 無 Dropout, 無 Softmax | Dropout + Masked Softmax |
| 推薦場景 | 大規模 grid (6×6+)，快速實驗 | 鄰接結構不對稱，需要自適應權重 |

### 參數量開銷（Hangzhou 4×4）

| 組件 | 參數量 | 說明 |
|------|--------|------|
| State Encoder | ~14K | Dense(128) + Dense(64) |
| Message Encoder | ~6K | Dense(64) + Dense(32) |
| GAT × 2 rounds | ~8K | W, a_left, a_right, bias per round |
| Gate + Update × 2 rounds | ~6K | Dense(32) × 4 |
| Message Decoder | ~6K | Dense(64) + Dense(64) |
| Level 2 Gate (in BDQ) | ~132K | comm_proj(256) + comm_gate(256) |
| 自監督 Predictors | ~4K | reward + neighbor predictor |
| **Total** | **~148K** | **+42.3% over base 395K** |

---

## 程式碼結構

```
agentpool/
├── region_communication.py          # Level 1 完整實作
│   ├── RegionGATLayer               # GAT 注意力層
│   ├── InterRegionCommunication     # GAT: Message Encoder + GAT rounds + Decoder
│   ├── MeanFieldCommunication       # MeanField: 鄰接平均聚合 (輕量替代 GAT)
│   ├── build_region_adjacency_matrix # 鄰接矩陣建構
│   └── RegionCoordinator            # 統籌通訊 + 自監督訓練 + context 快取
│
├── AdaptiveBDQ_agent.py             # Level 0 + Level 2
│   ├── build_network()              # 包含 comm context 融合 gate
│   └── AdaptiveBDQ_agent            # Agent 訓練邏輯
│
├── region_assignment.py             # Region 劃分 (minimum dominating set)
│
configs/
└── agent_config.py                  # COMM_CONFIG 超參數
```

---

## 參考文獻

- **GAT**: Veličković et al., *Graph Attention Networks*, ICLR 2018
- **CommNet**: Sukhbaatar et al., *Learning Multiagent Communication with Backpropagation*, NeurIPS 2016
- **TarMAC**: Das et al., *TarMAC: Targeted Multi-Agent Communication*, ICML 2019
- **RegionLight**: 本專案的區域化交通號誌控制框架
