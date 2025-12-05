# DQN Hyperparameters
DQNGAMMA = 0.99  # 折扣因子 γ
LEARNING_RATE = 0.001  # 学习率，可以调整
MEMORY_SIZE = 100000  # 记忆缓冲区大小
BATCH_SIZE = 20  # 最小一组数据个数
EXPLORATION_MAX = 1.0  # 最大探索率
EXPLORATION_MIN = 0.01  # 最小探索率
EXPLORATION_DECAY = 0.998  # 探索率下降，可以调整
DQNUNITS = 24 # 神经元数量，调少了速度快但需要更多训练时间，很推荐调整看看

# PPO Hyperparameters
GAMMA = 0.99 #伽马参数
LAM = 0.95 #lamda参数
CLIP_RATIO = 0.2 #上下裁剪区间，不用调整
LR = 0.001 #learning rate 学习率，可以尝试做调整
EPOCHS = 10 # 一组个数，不用调整
ENTROPY = 0.01 # 学习熵，如果要进一步做可以调整
UNITS = 64 # 神经元数量，调少了速度快但需要更多训练时间，很推荐调整看看

# Training
TOTAL_EPISODES = 100
TARGET_UPDATE_FREQ = 10

#Testing
BEST_SCORE = 400