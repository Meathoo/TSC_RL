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
    
    # Contrastive Pre-training Configuration
    "CONTRASTIVE_CONFIG": {
        "ENABLED": True,                # 是否启用对比预训练
        
        # Phase 1: Pre-training
        "COLLECT_EPISODES": 10,         # Random policy 收集 episode 数
        "PRETRAIN_EPOCHS": 150,         # 预训练 epoch 数
        "PRETRAIN_BATCH_SIZE": 256,     # Pre-training batch size
        "PRETRAIN_LR": 0.001,           # Pre-training 学习率（Adam）
        "PROJECTION_DIM": 128,          # Projection head 输出维度
        "TEMPERATURE": 0.1,             # NT-Xent loss temperature
        
        # Augmentation
        "NOISE_STD": 0.1,               # Gaussian noise 标准差
        "MASK_RATIO": 0.15,             # Intersection masking 比例
        
        # Phase 2: Auxiliary Loss
        "AUX_LOSS_WEIGHT": 0.01,        # 辅助损失权重 λ
    }
}
