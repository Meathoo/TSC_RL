AGENT_CONFIG={
    "BATCH_SIZE":32,
    "MEMORY_SIZE":200000,
    "MAX_EPSILON":1,
    "MIN_EPSILON":0.001,
    "DECAY_STEPS":20000,
    "NET_REPLACE_TYPE":"SOFT",
    "TAU":0.001,
    "REPLACE_INTERVAL":200,
    "GAMMA":0.99,#0.9

}

BDQ_AGENT_CONFIG={
    "TD_OPERATOR":"MEAN",
    "LEARNING_RATE":0.0001,
    # ---- Quick runtime profiling ----
    "PROFILE_CONFIG": {
        "ENABLED": True,              # print per-episode timing breakdown
        "PRINT_INTERVAL": 1,          # print every N episodes
    },
    # ---- Hierarchical Region Communication ----
    "COMM_CONFIG": {
        "ENABLED": True,               # set True to enable inter-region communication
        "COMM_TYPE": "MEANFIELD",       # "MEANFIELD" (fast, simple mean agg) or "GAT" (attention-based)
        "COMM_MESSAGE_DIM": 32,         # message compression dimension between regions
        "COMM_HIDDEN_DIM": 64,          # context vector dimension (State Encoder output dim)
        "COMM_NUM_HEADS": 4,            # GAT attention heads (only used when COMM_TYPE="GAT")
        "COMM_NUM_ROUNDS": 2,           # communication rounds (multi-hop propagation)
        "COMM_DROPOUT_RATE": 0.1,       # GAT attention dropout (only used when COMM_TYPE="GAT")
        "COMM_LEARNING_RATE": 0.0001,   # Adam learning rate for communication module
        "PROXIMITY_THRESHOLD": 2,       # Manhattan distance threshold for region adjacency
        # --- communication curriculum ---
        "COMM_CURRICULUM_ENABLED": True,   # if True, delay communication activation
        "COMM_CURRICULUM_START_STEP": 5000,    # enable communication when global_step >= this value
        "E2E_ENABLED": False,           # default off: use self-supervised comm training only (much faster)
        "E2E_FREQ": 5,                  # E2E backward through comm every N learn steps (1=always, 5=20% of steps)
        # --- self-supervised training ---
        "COMM_GRAD_CLIP_NORM": 5.0,     # gradient clipping (global norm)
        "COMM_TRAIN_INTERVAL": 50,      # train comm module every N global steps (higher = faster)
        "COMM_WARMUP_STEPS": 10000,     # start comm self-supervised training after replay warmup
        "COMM_BATCH_SIZE": 64,          # self-supervised training batch size
        "COMM_BUFFER_SIZE": 50000,      # replay buffer capacity
        # --- gate diagnostics & regularization ---
        "GATE_STATS_ENABLED": True,      # log gate distribution statistics periodically
        "GATE_STATS_LOG_INTERVAL": 100,  # coordinator train-step interval for gate stats refresh
        "GATE_REG_ENABLED": False,       # lightweight gate regularization switch (default off)
        "GATE_REG_WEIGHT": 0.001,        # regularization strength when enabled
        "GATE_REG_TARGET_MEAN": 0.5,     # target mean gate value
        "GATE_REG_SAT_THRESHOLD": 0.05,  # saturation threshold for low/high gate ratio
    },
}