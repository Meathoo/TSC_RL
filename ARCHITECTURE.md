# RegionLight 系統架構詳解

## 目錄

- [1. 專案定位](#1-專案定位)
- [2. 整體架構鳥瞰](#2-整體架構鳥瞰)
- [3. 核心模組詳解](#3-核心模組詳解)
  - [3.1 環境層：CityFlow 模擬器封裝](#31-環境層cityflow-模擬器封裝)
  - [3.2 區域劃分機制](#32-區域劃分機制)
  - [3.3 Agent 層：Branching DQN 系列](#33-agent-層branching-dqn-系列)
  - [3.4 跨區域注意力通訊機制（CRAC）](#34-跨區域注意力通訊機制crac)
  - [3.5 訓練管線 Pipeline](#35-訓練管線-pipeline)
  - [3.6 入口與配置系統](#36-入口與配置系統)
- [4. 資料流完整走向](#4-資料流完整走向)
- [5. 網路架構圖解](#5-網路架構圖解)
  - [5.1 ABDQ 基線網路](#51-abdq-基線網路)
  - [5.2 CommBDQ 通訊增強網路](#52-commbdq-通訊增強網路)
- [6. 跨區域通訊（CRAC）深入解析](#6-跨區域通訊crac深入解析)
  - [6.1 設計動機](#61-設計動機)
  - [6.2 Region Adjacency Graph 建構](#62-region-adjacency-graph-建構)
  - [6.3 Message 生成與交換流程](#63-message-生成與交換流程)
  - [6.4 Scaled Dot-Product Attention 聚合](#64-scaled-dot-product-attention-聚合)
  - [6.5 Gating Mechanism](#65-gating-mechanism)
  - [6.6 端到端梯度路徑](#66-端到端梯度路徑)
  - [6.7 擴展的 Replay Buffer](#67-擴展的-replay-buffer)
- [7. 超參數一覽](#7-超參數一覽)
- [8. 目錄結構與檔案說明](#8-目錄結構與檔案說明)
- [9. 使用方法](#9-使用方法)
- [10. 與相關工作的比較](#10-與相關工作的比較)

---

## 1. 專案定位

RegionLight 是一個基於深度強化學習的**區域級交通號誌控制**系統。與傳統「每個路口一個 Agent」的做法不同，RegionLight 將路網中的多個路口分群為若干 **Region**，每個 Region 由一個 **Branching DQN (BDQ)** Agent 統一控制，以利用局部空間關聯性降低動作空間維度並提升學習效率。

在此基礎上，我們進一步引入 **Cross-Region Attention Communication (CRAC)** 機制，讓相鄰 Region 的 Agent 能夠交換資訊，實現跨區域的協調決策。

**關鍵技術棧：**

| 組件 | 技術 |
|------|------|
| 模擬器 | [CityFlow](https://cityflow.readthedocs.io/) |
| 深度學習框架 | TensorFlow 2.4.1 |
| RL 演算法 | Branching Dueling DQN (Double DQN) |
| 通訊機制 | Scaled Dot-Product Attention + Gating |
| 區域劃分 | 最小支配集（Gurobi 求解） |

---

## 2. 整體架構鳥瞰

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (入口)                           │
│  解析參數 → 載入配置 → 建構環境 → 建構 Agent → 啟動 Pipeline    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PipeLine.py (訓練迴圈)                     │
│                                                                 │
│  for episode in episodes:                                       │
│    for step in steps:                                           │
│      ┌──────────────────────────────────────────┐               │
│      │  [若使用 CRAC]                            │               │
│      │   RegionCommCoordinator.exchange_messages │               │
│      │         ↕ messages ↕                      │               │
│      └──────────────────────────────────────────┘               │
│      Agent.choose_action(obs, neighbor_info)                    │
│      ──→ env.step(actions) ──→ state, reward                    │
│      Agent.store_transition(...)                                │
│      Agent.learn()                                              │
└────────┬──────────────────────────────────────┬─────────────────┘
         │                                      │
         ▼                                      ▼
┌─────────────────────┐              ┌─────────────────────────┐
│ cityflow_env_wrapper │              │   agentpool/             │
│ (CityFlow 模擬器)    │              │   ├─ AdaptiveBDQ_agent  │
│                      │              │   ├─ CommBDQ_agent      │
│ • get state          │              │   └─ region_comm        │
│ • execute actions    │              │       (通訊協調器)       │
│ • compute reward     │              └─────────────────────────┘
└─────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      configs/ (配置系統)                         │
│  agent_config.py · env_config.py · exp_config.py · region_config│
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心模組詳解

### 3.1 環境層：CityFlow 模擬器封裝

**檔案：** `cityflow_env_wrapper.py`

`CityflowEnvWrapper` 封裝了 CityFlow 微觀交通模擬器，提供統一的 RL 環境介面。

#### 3.1.1 State 表示

每個路口的 state 是一個 **25 維向量**：

| 維度 | 特徵 | 說明 |
|------|------|------|
| 0–11 | Waiting Queue Length | 12 條進入車道的等待車輛數 |
| 12–23 | Wave | 12 條進入車道的總車輛數（含移動中） |
| 24 | Current Phase | 當前相位（1/2/3/4） |

每個路口有 4 條進入道路 × 3 條車道 = 12 條進入車道。

#### 3.1.2 Action 空間

使用 `CHOOSE PHASE` 模式，每個路口從 4 個固定相位中選擇一個：
- Phase 1 → Phase 3 → Phase 2 → Phase 4（循環）

#### 3.1.3 Reward 設計

Reward = **負的等待車輛數**。等待越少，Reward 越高。

```
R(intersection) = -Σ(等待車輛數) across all incoming lanes
R(region) = Σ R(intersection) for all intersections in the region
```

#### 3.1.4 主要方法

| 方法 | 功能 |
|------|------|
| `reset()` | 重置環境，回傳初始 state |
| `step(actions)` | 執行一步（10 個模擬步），回傳 (state, reward, done, info) |
| `get_average_travel_time()` | 取得平均旅行時間（核心評估指標） |
| `get_throughput()` | 取得已完成旅程的車輛數 |
| `get_average_queue_length()` | 取得平均排隊長度 |

---

### 3.2 區域劃分機制

**檔案：** `region_assignment.py`, `configs/region_config.py`

#### 3.2.1 劃分原理

使用**最小支配集 (Minimum Dominating Set)** 演算法：
1. 將路網建模為圖（每個路口為節點，相鄰路口有邊）
2. 透過 Gurobi 求解整數線性規劃，找到最少的控制中心
3. 將每個非中心路口分配給最近的中心，形成 Region

#### 3.2.2 區域配置

以 Hangzhou 4×4 路網為例，`4_4_ADJACENCY1` 配置將 16 個路口分成 **4 個 Region**，每個 Region 包含 5 個槽位（含 dummy 填充）：

```
Region 0: [intersection_2_2, intersection_1_1, intersection_2_1, intersection_3_1, dummy]
Region 1: [intersection_1_4, dummy, intersection_1_3, intersection_2_3, intersection_1_2]
Region 2: [intersection_4_3, intersection_3_2, intersection_4_2, dummy, intersection_4_1]
Region 3: [dummy, intersection_2_4, intersection_3_4, intersection_4_4, intersection_3_3]
```

其中 `dummy` 表示填充位，用於對齊各 Region 的 action 分支數。

#### 3.2.3 State 與 Reward 映射

```python
# State 映射：將各路口 state (25-dim) 按 Region 分配拼接
# Region 觀測 = [itsx_0_state | itsx_1_state | ... | itsx_4_state] = 5×25 = 125 維
assign_state(global_state, itsx_assignment, itsx_state_dim=25)

# Reward 映射：Region reward = 所轄路口 reward 之和
assign_reward(itsx_rewards, itsx_assignment)
```

---

### 3.3 Agent 層：Branching DQN 系列

系統提供三種 Agent，複雜度遞增：

| Agent | 檔案 | 特點 |
|-------|------|------|
| **BDQ** | `BDQ_agent.py` | 基礎 Branching DQN |
| **ABDQ** | `AdaptiveBDQ_agent.py` | 自適應分支（支援 dummy/idle branch） |
| **CommBDQ** | `CommBDQ_agent.py` | ABDQ + 跨區域注意力通訊 |

#### 3.3.1 Branching DQN 原理

傳統 DQN 的 action 數量 = 所有路口動作的笛卡爾積，隨路口數指數增長。

**Branching DQN** 將動作空間分解為多個獨立分支，每個分支對應一個路口：

$$Q(s, a_1, a_2, ..., a_N) \approx V(s) + \frac{1}{N}\sum_{i=1}^{N} A_i(s, a_i)$$

其中：
- $V(s)$：共享的 State Value
- $A_i(s, a_i)$：第 $i$ 個分支的 Advantage
- $N$：Region 內路口數（action_dim）

$$Q_i = V(s) + \left(A_i(s, a_i) - \frac{1}{|A_i|}\sum_{a'} A_i(s, a')\right)$$

#### 3.3.2 ABDQ 網路架構

```
State Input (125-dim)
    ↓
Dense(512, ReLU) ─── Shared Representation
    ↓
Dense(256, ReLU)
    ├───────────────────────────────────────────┐
    ↓                                           ↓
Dense(128, ReLU) → Dense(1)              Branch ×5:
     V(s)                                  Dense(128, ReLU) → Dense(4)
                                              A_i(s, a)
    ↓                                           ↓
    └────── Q_i = V + (A_i - mean(A_i)) ───────┘
                        ↓
              Q-values: (batch, 5, 4)
```

- **可訓練參數：** ~260K / Agent
- **Shared Representation** 讓所有分支共享路網全局特徵
- **Dueling** 結構分離 State Value 與 Advantage
- **行動分支** 數量 = Region 內路口數（含 dummy）

#### 3.3.3 關鍵 RL 技巧

| 技巧 | 設定 |
|------|------|
| **Double DQN** | Eval net 選 action，Target net 估 Q |
| **TD Operator** | MEAN（跨分支平均）、MAX、NAIVE |
| **Target Network** | Soft update，τ=0.001 |
| **ε-greedy** | 1.0 → 0.001，線性衰減 20000 步 |
| **Replay Buffer** | 200,000 筆，batch_size=32 |
| **訓練門檻** | 累積 10,000 筆經驗後才開始學習 |
| **Idle Branch** | dummy 路口的分支 action 設為 -1，不計入 TD target |

---

### 3.4 跨區域注意力通訊機制（CRAC）

**檔案：** `agentpool/CommBDQ_agent.py`, `agentpool/region_comm.py`

這是本專案的核心創新，詳見 [第 6 節](#6-跨區域通訊crac深入解析)。

---

### 3.5 訓練管線 Pipeline

**檔案：** `PipeLine.py`

#### 3.5.1 主訓練迴圈

```python
for episode in range(EPISODE):        # e.g. 1000 episodes
    state = env.reset()
    obs = assign_state(state, itsx_assignment)
    
    for step in range(400):            # 4000 sim_sec / 10 action_interval
        # ── 通訊（僅 CommBDQ）──
        if use_comm:
            neighbor_data = comm_coordinator.exchange_messages(agents, obs)
        
        # ── 選動作 ──
        for agent in agents:
            action = agent.choose_action(obs, neighbor_data)
        
        # ── 環境步進 ──
        next_state, reward, done = env.step(actions)
        
        # ── 儲存經驗 ──
        agent.store_transition(s, [n_msgs, n_mask], a, r, s', [n_msgs', n_mask'])
        
        # ── 學習（每 10 步） ──
        if step % 10 == 0:
            agent.learn()
```

#### 3.5.2 訓練範式

| 範式 | 說明 |
|------|------|
| **DECENTRAL** | 各 Region Agent 擁有獨立網路與 Replay Buffer |
| **CLDE** | 所有 Agent 共享同一組網路參數（Centralized Learning Decentralized Execution） |

#### 3.5.3 日誌與監控

每個 episode 記錄：
- Wall time、累計時間、ETA
- Episode reward、Average Travel Time、Queue Length、Throughput
- Epsilon、Loss、Gradient norm（若可用）
- 訓練結束後寫入 `training_log.json `

---

### 3.6 入口與配置系統

#### 3.6.1 main.py 流程

```
parse_args()
    ↓
init_exp()
    ├→ 建構 CityflowEnvWrapper
    ├→ 讀取 Region 配置
    ├→ 建構 Agent（ABDQ / CommBDQ）
    ├→ [若 CommBDQ] 建構 RegionCommCoordinator
    └→ 寫入 hyperparameter.txt
    ↓
run_pipeline()
    ├→ pipeline() 訓練
    └→ 儲存曲線與模型
```

#### 3.6.2 配置檔案

| 檔案 | 內容 |
|------|------|
| `agent_config.py` | RL 超參數（lr, γ, ε, buffer size, batch size, τ, TD operator） |
| `env_config.py` | 環境設定（模擬時間、action interval、action type） |
| `exp_config.py` | 實驗設定（episodes、訓練範式、Agent 類別、Region 類型） |
| `region_config.py` | 各路網的 Region 劃分方案 |

---

## 4. 資料流完整走向

以 **CommBDQ + Hangzhou 4×4 + ADJACENCY1** 為例：

```
Step 1: 環境觀測
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CityFlow Simulator
    │ get_state()
    ▼
{intersection_1_1: [25-dim], intersection_1_2: [25-dim], ...}  (16 個路口)
    │ assign_state()
    ▼
[obs_R0: 125-dim, obs_R1: 125-dim, obs_R2: 125-dim, obs_R3: 125-dim]

Step 2: 跨區域通訊
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
obs_R0 ──→ Agent 0 ──→ msg_0 (64-dim)  ─┐
obs_R1 ──→ Agent 1 ──→ msg_1 (64-dim)  ─┤ RegionCommCoordinator
obs_R2 ──→ Agent 2 ──→ msg_2 (64-dim)  ─┤ .exchange_messages()
obs_R3 ──→ Agent 3 ──→ msg_3 (64-dim)  ─┘
                                          │
            根據 Region Adjacency Graph:  │
            R0 ↔ [R1, R2]                │
            R1 ↔ [R0, R3]                │
            R2 ↔ [R0, R3]                │
            R3 ↔ [R1, R2]                │
                                          ▼
Agent 0 收到: {msgs: [msg_1, msg_2], mask: [1, 1]}
Agent 1 收到: {msgs: [msg_0, msg_3], mask: [1, 1]}
Agent 2 收到: {msgs: [msg_0, msg_3], mask: [1, 1]}
Agent 3 收到: {msgs: [msg_1, msg_2], mask: [1, 1]}

Step 3: 動作選擇
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Agent 0: obs_R0 + neighbor_msgs ──→ Attention ──→ Q-values (5, 4) ──→ [2,3,1,-1,0]
Agent 1: obs_R1 + neighbor_msgs ──→ Attention ──→ Q-values (5, 4) ──→ [1,-1,3,2,0]
  ...
    │ convert_actions()（-1 = idle/dummy，其餘 +1 映射為相位 1~4）
    ▼
{intersection_1_1: 3, intersection_2_1: 4, intersection_2_2: 3, ...}

Step 4: 環境步進
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
env.step(joint_actions)
    │ 模擬 10 個時間步
    ▼
(next_state, itsx_rewards, done, log_metric)
    │ assign_reward()
    ▼
[reward_R0, reward_R1, reward_R2, reward_R3]

Step 5: 儲存與學習
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
store_transition(s, n_msgs, n_mask, a, r, s', n_msgs', n_mask')
    ↓
learn() ──→ sample batch ──→ update_gradient() ──→ 更新 Q-network + 通訊模組
```

---

## 5. 網路架構圖解

### 5.1 ABDQ 基線網路

```
                     State (125)
                         │
                    ┌────┴────┐
                    │Dense 512│ ReLU
                    └────┬────┘
                    ┌────┴────┐
                    │Dense 256│ ReLU          ← Shared Representation
                    └────┬────┘
              ┌──────────┼──────────┐
              │          │          │
         ┌────┴────┐     │     ┌────┴────┐ ×5 branches
         │Dense 128│     │     │Dense 128│ ReLU
         │Dense  1 │     │     │Dense  4 │
         └────┬────┘     │     └────┬────┘
           V(s)          │       A_i(s,a)
              │          │          │
              └────┬─────┘─────┬────┘
                   │           │
            Q_i = V + (A_i - mean(A_i))
                        │
               Q: (batch, 5, 4)

  Total trainable params: ~260,000
```

### 5.2 CommBDQ 通訊增強網路

```
    State (125)       Neighbor Messages (N, 64)    Neighbor Mask (N)
        │                       │                        │
   ┌────┴────┐                  │                        │
   │Dense 512│ ReLU             │                        │
   └────┬────┘                  │                        │
   ┌────┴────┐                  │                        │
   │Dense 256│ ReLU             │                        │
   └────┬────┘                  │                        │
        │                       │                        │
        ├──────────┐            │                        │
        │     ┌────┴────┐       │                        │
        │     │Dense  64│ ReLU  │                        │
        │     └────┬────┘       │                        │
        │    own_msg (64)       │                        │
        │          │            │                        │
        │     ┌────┴──┐    ┌────┴─────┐                  │
        │     │ Wq    │    │ Wk   Wv │                   │
        │     │(64×64)│    │(64×64)×2 │                   │
        │     └───┬───┘    └──┬────┬──┘                   │
        │         │           │    │                      │
        │     ┌───┴───────────┴────┘                      │
        │     │  Scaled Dot-Product Attention              │
        │     │  scores = (Q · K^T) / √64                 │
        │     │  scores += (1 - mask) × (-1e9)  ←─────────┘
        │     │  weights = softmax(scores)
        │     │  comm = weights · V
        │     └────────┬────────┘
        │              │
        │         ┌────┴────┐
        │         │LayerNorm│
        │         └────┬────┘
        │              │
        │    ┌─────────┴─────────┐
        ├────┤   Gating (σ)      │
        │    │ gate = σ(W·[shared;comm])
        │    │ gated_comm = gate ⊙ comm
        │    └─────────┬─────────┘
        │              │
   ┌────┴──────────────┴────┐
   │  Concatenate           │
   │  [shared_repr; gated]  │  = (256 + 64) = 320 dim
   └───────────┬────────────┘
               │
        ┌──────┼──────────┐
        │      │          │
   ┌────┴────┐ │     ┌────┴────┐ ×5 branches
   │Dense 128│ │     │Dense 128│ ReLU
   │Dense  1 │ │     │Dense  4 │
   └────┬────┘ │     └────┬────┘
     V(s)      │       A_i(s,a)
        │      │          │
        └──┬───┘───┬──────┘
           │       │
    Q_i = V + (A_i - mean(A_i))
                │
       Q: (batch, 5, 4)     own_msg: (batch, 64)

  Total trainable params: ~494,000 (原始 ~260K + 通訊模組 ~234K)
```

---

## 6. 跨區域通訊（CRAC）深入解析

### 6.1 設計動機

原始 ABDQ 中，每個 Region Agent **完全獨立**決策。這在邊界路口（Region 交界處）會產生問題：

```
┌─────────────────┐   ┌─────────────────┐
│    Region 0      │   │    Region 1      │
│                  │   │                  │
│  itsx_1_1  ─────►│◄──│── itsx_1_2      │
│                  │   │                  │
│  Agent 0 獨立決策 │   │ Agent 1 獨立決策  │
└─────────────────┘   └─────────────────┘
        ↑                       ↑
   不知道 Region 1 的          不知道 Region 0 的
   交通壓力與相位狀態          交通壓力與相位狀態
```

**問題：** `itsx_1_1` 和 `itsx_1_2` 相鄰，它們的號誌決策會互相影響（例如一方放行會增加另一方的進入流量），但兩個 Agent 彼此不知道對方的狀態和意圖。

**CRAC 解決方案：** 讓 Agent 之間透過可學習的 message passing 機制交換摘要資訊。

### 6.2 Region Adjacency Graph 建構

**自動推導**，不需要人工指定 Region 間的鄰接關係。

演算法（`build_region_adjacency`）：
1. 解析每個 Region 中所有路口的 `(x, y)` 座標
2. 若 Region $i$ 和 Region $j$ 存在路口對 Manhattan Distance = 1，則它們相鄰

以 `4_4_ADJACENCY1` 為例，自動推導的鄰接圖：

```
  R0 ──── R1
  │ ╲    ╱ │
  │  ╲  ╱  │
  │   ╲╱   │
  │   ╱╲   │
  │  ╱  ╲  │
  │ ╱    ╲ │
  R2 ──── R3

鄰接表：
  R0: [R1, R2]
  R1: [R0, R3]
  R2: [R0, R3]
  R3: [R1, R2]
```

### 6.3 Message 生成與交換流程

```
每個 decision step:

┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
│  Agent 0    │  │  Agent 1    │  │  Agent 2    │  │  Agent 3    │
│             │  │             │  │             │  │             │
│ obs_0 (125) │  │ obs_1 (125) │  │ obs_2 (125) │  │ obs_3 (125) │
│      │      │  │      │      │  │      │      │  │      │      │
│ Shared Repr │  │ Shared Repr │  │ Shared Repr │  │ Shared Repr │
│      │      │  │      │      │  │      │      │  │      │      │
│ Msg Encoder │  │ Msg Encoder │  │ Msg Encoder │  │ Msg Encoder │
│      │      │  │      │      │  │      │      │  │      │      │
│ msg_0 (64)  │  │ msg_1 (64)  │  │ msg_2 (64)  │  │ msg_3 (64)  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │                │
       └────────────────┼────────────────┼────────────────┘
                        │
              RegionCommCoordinator
              .exchange_messages()
                        │
       ┌────────────────┼────────────────┐
       │                │                │
       ▼                ▼                ▼
   Agent 0 收到      Agent 1 收到     Agent 2 收到    Agent 3 收到
   [msg_1, msg_2]   [msg_0, msg_3]  [msg_0, msg_3]  [msg_1, msg_2]
   mask=[1, 1]      mask=[1, 1]     mask=[1, 1]     mask=[1, 1]
```

**Message 內容：** 64 維向量，由 shared representation（256-dim）經 `Dense(64, ReLU)` 壓縮而來。Message 隱式編碼了 Region 的交通狀態、壓力水平等摘要資訊。

### 6.4 Scaled Dot-Product Attention 聚合

為什麼用 Attention 而非簡單 Mean Pooling？

- **選擇性聚合：** 不同鄰居的資訊重要程度不同。例如塞車方向的鄰居資訊更重要。
- **自適應權重：** Attention 權重由當前 Agent 的 query 與鄰居的 key 動態計算。

數學表達：

$$\text{Query} = W_Q \cdot \text{own\_msg} \quad (64\text{-dim})$$

$$\text{Key}_i = W_K \cdot \text{neighbor\_msg}_i \quad (64\text{-dim})$$

$$\text{Value}_i = W_V \cdot \text{neighbor\_msg}_i \quad (64\text{-dim})$$

$$\alpha_i = \text{softmax}\left(\frac{Q \cdot K_i^T}{\sqrt{d_k}} + M_i\right)$$

$$\text{comm} = \sum_i \alpha_i \cdot V_i$$

其中 $M_i = 0$ 若鄰居 $i$ 有效，$M_i = -10^9$ 若為 padding。

### 6.5 Gating Mechanism

通訊品質在訓練初期不可靠（messages 來自隨機權重的網路）。Gating 機制讓 Agent 自適應控制對通訊的依賴：

$$\text{gate} = \sigma(W_g \cdot [\text{shared\_repr}; \text{comm\_normed}])$$

$$\text{gated\_comm} = \text{gate} \odot \text{LayerNorm}(\text{comm})$$

- 訓練初期：gate 學到較小值 → 主要依賴 local state
- 訓練後期：gate 逐漸打開 → 開始利用鄰居通訊
- 不同通訊維度的 gate 可以獨立控制

### 6.6 端到端梯度路徑

通訊模組**完全透過 Q-loss 端到端訓練**，不需要額外的輔助損失。

```
Loss (MSE of TD error)
    │
    ▼ ∂Loss/∂Q
Q-values
    │
    ▼ ∂Q/∂augmented
Augmented Representation = [shared_repr; gated_comm]
    │                            │
    │ (直接路徑)                   │ (通訊路徑)
    │                            ▼
    │                     Gated Comm
    │                            │
    │                       ▼         ▼
    │                  Gate (σ)   LayerNorm
    │                    │            │
    │               ▼         ▼       ▼
    │         Shared Repr    Comm (Attention output)
    │               │              │
    │               ▼              ▼
    │         Concatenate    Weights · Values
    │               │              │
    │               ▼              ▼
    │          W_gate        Q · K^T / √d
    │                        │         │
    │                        ▼         ▼
    │                    own_msg   neighbor_msgs
    │                        │         │
    │                        ▼         ▼
    │                   Msg Encoder  W_k, W_v
    │                        │
    ▼                        ▼
Shared Representation (Dense 512 → Dense 256)
    │
    ▼
State Input
```

這個梯度路徑確保：
- **Message Encoder** 學會生成有幫助的 messages
- **Attention Q/K/V** 學會選擇有用的鄰居資訊
- **Gate** 學會何時信任通訊

### 6.7 擴展的 Replay Buffer

CommBDQ 的 Replay Buffer 比標準 BDQ 多儲存通訊上下文：

| 欄位 | Shape | 說明 |
|------|-------|------|
| `state_memory` | (200K, 125) | Local state |
| `neighbor_msgs_memory` | (200K, N, 64) | 當前步鄰居 messages |
| `neighbor_mask_memory` | (200K, N) | 當前步有效鄰居遮罩 |
| `action_memory` | (200K, 5) | Joint action |
| `reward_memory` | (200K, 1) | Region reward |
| `next_state_memory` | (200K, 125) | Next local state |
| `next_neighbor_msgs_memory` | (200K, N, 64) | 下一步鄰居 messages |
| `next_neighbor_mask_memory` | (200K, N) | 下一步有效鄰居遮罩 |

其中 N = `max_neighbors`（4×4 ADJACENCY1 中 N=2）。

**注意：** 訓練時重建 attention 是基於 replay buffer 中**儲存時的 messages**（Off-policy 的 communication context），而非根據 next state 重新生成。這是一種務實的近似，避免了需要多次 forward pass 的複雜度。

---

## 7. 超參數一覽

### Agent 通用配置 (`AGENT_CONFIG`)

| 參數 | 值 | 說明 |
|------|-----|------|
| `BATCH_SIZE` | 32 | Mini-batch 大小 |
| `MEMORY_SIZE` | 200,000 | Replay buffer 容量 |
| `MAX_EPSILON` | 1.0 | 初始探索率 |
| `MIN_EPSILON` | 0.001 | 最終探索率 |
| `DECAY_STEPS` | 20,000 | ε 衰減的 learn 步數 |
| `NET_REPLACE_TYPE` | SOFT | Target network 更新方式 |
| `TAU` | 0.001 | Soft update 係數 |
| `REPLACE_INTERVAL` | 200 | Target net 更新間隔 |
| `GAMMA` | 0.99 | 折扣因子 |

### BDQ/ABDQ 配置 (`BDQ_AGENT_CONFIG`)

| 參數 | 值 | 說明 |
|------|-----|------|
| `TD_OPERATOR` | MEAN | 跨分支 TD 聚合方式 |
| `LEARNING_RATE` | 0.0001 | Adam 學習率 |

### CommBDQ 配置 (`COMM_BDQ_AGENT_CONFIG`)

| 參數 | 值 | 說明 |
|------|-----|------|
| `TD_OPERATOR` | MEAN | 跨分支 TD 聚合方式 |
| `LEARNING_RATE` | 0.0001 | Adam 學習率 |
| `COMM_DIM` | 64 | Message embedding 維度 |

### 環境配置 (`ENV_CONFIG`)

| 參數 | 值 | 說明 |
|------|-----|------|
| `SIM_TIMESPAN` | 4000 | 模擬總秒數 |
| `ACTION_INTERVAL` | 10 | 動作執行間隔（秒） |
| `ACTION_TYPE` | CHOOSE PHASE | 動作類型 |
| `INTERVAL` | 1.0 | CityFlow 模擬步長（秒） |

### 實驗配置 (`EXP_CONFIG`)

| 參數 | 值 | 說明 |
|------|-----|------|
| `EPISODE` | 1000 | 訓練回合數 |
| `TRAINING_PARADIM` | DECENTRAL | 訓練範式 |
| `REGIONAL` | True | 是否使用區域劃分 |
| `REGION_TYPE` | ADJACENCY1 | Region 劃分方案 |
| `LEARNING_INTERVAL` | 10 | 每幾步學習一次 |

---

## 8. 目錄結構與檔案說明

```
RegionLight/
│
├── main.py                      # 入口：解析參數 → 初始化 → 啟動訓練
├── PipeLine.py                  # 訓練迴圈：Agent-Environment 互動 + 通訊
├── cityflow_env_wrapper.py      # CityFlow 模擬器的 RL 環境封裝
├── region_assignment.py         # 最小支配集 Region 劃分演算法
├── plot.py                      # 訓練曲線繪製
├── compare.py                   # 多實驗比較工具
├── ARCHITECTURE.md              # ← 本文件
│
├── agentpool/                   # RL Agent 實作
│   ├── BDQ_agent.py             #   基礎 Branching DQN
│   ├── AdaptiveBDQ_agent.py     #   自適應 BDQ（支援 idle branch）
│   ├── CommBDQ_agent.py         #   BDQ + 跨區域注意力通訊（CRAC）
│   └── region_comm.py           #   Region 鄰接圖建構 + 訊息交換協調器
│
├── configs/                     # 配置檔案
│   ├── agent_config.py          #   Agent 超參數
│   ├── env_config.py            #   環境設定
│   ├── exp_config.py            #   實驗設定（含 Agent 類別註冊）
│   └── region_config.py         #   Region 劃分方案（手動/預計算）
│
├── data/                        # 路網與車流資料
│   ├── Hangzhou/                #   杭州 4×4 路網
│   ├── Manhattan/               #   曼哈頓 16×3 路網
│   ├── Syn/                     #   合成路網 1
│   └── Syn2/                    #   合成路網 2
│
├── records/                     # 訓練結果輸出
│   └── {experiment_name}/       #   包含模型、曲線、日誌
│
└── SamplePhasePlan/             # 固定相位計畫（baseline 對比）
```

---

## 9. 使用方法

### 基本訓練（無通訊的 ABDQ）

```bash
python main.py --netname Hangzhou --netshape 4_4 --flow peak --agent ABDQ
```

### 帶跨區域通訊的 CommBDQ

```bash
python main.py --netname Hangzhou --netshape 4_4 --flow peak --agent CommBDQ
```

### 基礎 BDQ

```bash
python main.py --netname Hangzhou --netshape 4_4 --flow peak --agent BDQ
```

### 結果比較

```bash
python compare.py -f records/exp_1 records/exp_2 -n ABDQ CommBDQ --smooth 10
```

### 可用數據集

| 數據集 | netname | netshape | flow |
|--------|---------|----------|------|
| 杭州平峰 | Hangzhou | 4_4 | flat |
| 杭州尖峰 | Hangzhou | 4_4 | peak |
| 曼哈頓 | Manhattan | 16_3 | real |
| 合成 1 | Syn | 4_4 | gaussian |
| 合成 2 | Syn2 | 4_4 | gaussian |

---

## 10. 與相關工作的比較

| 方法 | 單路口/區域 | Agent 間通訊 | 通訊方式 | 可擴展性 |
|------|-----------|-------------|---------|---------|
| Independent DQN | 單路口 | ❌ | - | ✅ |
| CoLight (MARL) | 單路口 | ✅ | Graph Attention | ⚠️ 路口數多時通訊成本高 |
| RegionLight (ABDQ) | 區域 | ❌ | - | ✅ |
| **RegionLight (CommBDQ)** | **區域** | **✅** | **Gated Scaled Attention** | **✅ Region 粒度通訊** |

**CommBDQ 的優勢：**

1. **Region 粒度通訊：** 通訊發生在 Region 之間而非路口之間，通訊複雜度 $O(R^2)$ 遠小於 $O(N^2)$（$R$=Region 數, $N$=路口數）
2. **Learnable Gating：** 自適應控制通訊信任度，避免初期隨機 messages 干擾學習
3. **端到端訓練：** 不需要額外的通訊損失函數
4. **向後相容：** 選擇 `--agent ABDQ` 即回退到無通訊模式，Pipeline 完全相容
