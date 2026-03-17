# TSC_RL_hie2 架構整理

本文件整理目前 `TSC_RL_hie2` 版本的整體設計、資料流、關鍵模組與實驗輸出格式。

## 1. 設計目標

這個版本是以 `ABDQ` 為基底，加入 **Hierarchical Region Communication**，並強化：

- 跨 region 協作（Level 1 通訊）
- 可控訓練成本（curriculum、comm train interval、MeanField）
- 可觀測性（profile 與 gate statistics）
- 指標一致性（terminal AQL 與 episode-average AQL 同時紀錄）

## 2. 三層架構

### Level 0: 區域內決策（Local BDQ）

每個 agent 負責一個 region 內多個 intersection（branching action）。

- 模型：Branching Dueling Q Network
- 輸入：region state（queue/wave/phase）
- 輸出：每個 branch 的子動作 Q 值

關鍵檔案：

- `agentpool/AdaptiveBDQ_agent.py`

### Level 1: 區域間通訊（Inter-Region Communication）

`RegionCoordinator` 根據 region adjacency 圖做 message passing，輸出每個 region 的 context vector。

- `COMM_TYPE=GAT`: attention 聚合（表達力高）
- `COMM_TYPE=MEANFIELD`: 鄰居平均聚合（較快，實驗主力）
- 多輪通訊：`COMM_NUM_ROUNDS`
- Gate 更新：控制 local 與 neighbor 訊息融合比例

關鍵檔案：

- `agentpool/region_communication.py`

### Level 2: 決策融合（Decision Fusion）

agent 網路內部把 local representation 與 comm context 做 gated fusion，再算 Q。

- `comm_proj` 將 context 投影到共享特徵維度
- `comm_gate` 學習 local/context 權重

關鍵檔案：

- `agentpool/AdaptiveBDQ_agent.py`

## 3. 訓練資料流（每個 step）

1. `PipeLine.py` 從環境取得各 region observation。
2. 依 epsilon 先決定各 agent 是否探索（random path 可跳過 model forward）。
3. 若有需要 greedy 且通訊啟用：
   - `coordinator.get_context(all_obs)` 計算所有 region context。
4. 每個 agent 執行 `choose_action(obs, comm_context)`。
5. `env.step(joint_actions)` 回傳 next state / reward / done。
6. Agent replay 存 transition；週期性 `learn()`。
7. 通訊模組在 warmup 後按 interval 做 self-supervised train。

關鍵檔案：

- `PipeLine.py`

## 4. RegionCoordinator 訓練機制

`RegionCoordinator.train()` 採自監督雙目標：

- `L_self`: 用 context 預測 own reward
- `L_neighbor`: 用 context 預測 mean neighbor reward

總損失：

$$
L = L_{self} + L_{neighbor} + \lambda L_{gate\_reg}\; (optional)
$$

其中 gate regularization 可選（預設關），用來避免 gate 長期飽和。

相關設定：

- `GATE_REG_ENABLED`
- `GATE_REG_WEIGHT`
- `GATE_REG_TARGET_MEAN`
- `GATE_REG_SAT_THRESHOLD`

## 5. E2E 與非 E2E

`AdaptiveBDQ_agent.update_gradient()` 支援兩種模式：

- `e2e=True`: TD loss 梯度可回傳到 coordinator（較重）
- `e2e=False`（常用）: 使用 frozen context（較快）

目前實驗主力通常採：

- `E2E_ENABLED=False`
- 透過 self-supervised comm training 維持通訊品質

## 6. Curriculum 與穩定性控制

通訊 curriculum 由以下參數控制：

- `COMM_CURRICULUM_ENABLED`
- `COMM_CURRICULUM_START_STEP`

用途：延後通訊啟用，讓 local policy 先穩定，再引入跨區域訊號。

另外可用以下參數控速控穩：

- `COMM_TRAIN_INTERVAL`
- `COMM_WARMUP_STEPS`
- `COMM_LEARNING_RATE`
- `COMM_NUM_ROUNDS`

關鍵檔案：

- `configs/agent_config.py`
- `PipeLine.py`

## 7. 指標定義與輸出（重點）

### Reward

- 來自 waiting queue 的負值，逐 step 累積
- 對中途壅塞非常敏感

### Throughput

- 完成旅次車輛數（完成 trip 的車）

### ATT

- 平均旅行時間（完成旅次）

### AQL（本版同時輸出兩種）

1. `AQL_terminal`
   - 回合最後一刻的 queue snapshot
2. `AQL_episode_avg`
   - 該回合內每 step queue 的平均（更能反映過程）

輸出檔案：

- `episode_average_queue_length.npy`（相容舊流程，等同 terminal）
- `episode_average_queue_length_terminal.npy`
- `episode_average_queue_length_episode_avg.npy`
- `training_log.json` 內每回合欄位：
  - `avg_queue_length_terminal`
  - `avg_queue_length_episode`

關鍵檔案：

- `PipeLine.py`
- `main.py`
- `cityflow_env_wrapper.py`

## 8. 比較工具（compare.py）

`compare.py` 支援多 run 比較，並輸出：

- 全指標總圖
- 各指標單圖

在這個版本中，會比較：

- `average_travel_time`
- `intersection_reward`
- `throughput`
- `average_queue_length_terminal`
- `average_queue_length_episode_avg`

並對舊資料夾做 fallback（若缺 terminal 檔則用 `episode_average_queue_length.npy`）。

關鍵檔案：

- `compare.py`

## 9. 啟動與主要入口

- 實驗入口：`main.py`
- 主要流程：`PipeLine.py`
- Agent：`agentpool/AdaptiveBDQ_agent.py`
- 協調器：`agentpool/region_communication.py`
- 設定：`configs/agent_config.py`, `configs/exp_config.py`, `configs/env_config.py`

## 10. 目前版本特徵總結

- 以 MeanField 通訊為主，兼顧速度與效果
- 已有 curriculum、gate stats、profile、dual AQL
- 可做公平比較（terminal vs episode-average）避免 AQL 解讀偏差
- 已具備從「性能」、「品質」、「可解釋性」三方面調參的基礎
