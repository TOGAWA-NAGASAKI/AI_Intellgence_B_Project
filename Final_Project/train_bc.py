import numpy as np
import gymnasium as gym
import os
import matplotlib.pyplot as plt
from agents.behavioral_cloning import BehavioralCloningAgent
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

ENV_NAME = "CartPole-v1"
DATA_FILE = "data/expert_data_n5000.npz"

def train_and_evaluate():
    # === 1. 加载数据 ===
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found!")
        return

    print(f"Loading data from {DATA_FILE}...")
    data = np.load(DATA_FILE)
    states = data['obs']
    actions = data['actions']

    # 数据清洗与类型强制转换
    states = states.astype(np.float32)
    actions = actions.astype(np.int32)

    print("\n=== Data Debug Info ===")
    print(f"States Shape: {states.shape}, Type: {states.dtype}")
    print(f"Actions Shape: {actions.shape}, Type: {actions.dtype}")
    print("=======================\n")

    # === 2. 初始化 Agent ===
    env = gym.make(ENV_NAME)
    # 确保 agents/behavioral_cloning.py 里的类名对得上，且不需要手动传 LR
    agent = BehavioralCloningAgent(env.observation_space.shape[0], env.action_space.n)

    # === 3. 训练 ===
    print("\n=== Starting Offline Training ===")
    
    # 确保 models 文件夹存在
    if not os.path.exists("models"):
        os.makedirs("models")

    # 定义回调函数
    checkpoint_path = "models/bc_best_model.weights.h5"
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = agent.model.fit(
        states, 
        actions, 
        batch_size=32, 
        epochs=50, 
        validation_split=0.2,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    # 显式加载最好的权重
    try:
        agent.model.load_weights(checkpoint_path)
        print("Restored best weights from checkpoint.")
    except:
        pass

    # === 4. 评估 (生成可视化数据) ===
    print("\n=== Evaluating & Visualizing ===")
    
    eval_episodes = 10  # 测试次数
    scores = []
    
    for i in range(eval_episodes):
        state, _ = env.reset()
        # 兼容 reshape，防止 agent.act 报错
        if len(state.shape) == 1:
            state = np.reshape(state, [1, state.shape[0]])
            
        score = 0
        while True:
            action = agent.act(state)
            state, _, terminated, truncated, _ = env.step(action)
            state = np.reshape(state, [1, state.shape[0]])
            score += 1
            if terminated or truncated or score >= 500:
                break
        scores.append(score)
        print(f"Eval Ep {i+1}: {score}")

    avg_score = np.mean(scores)
    print(f"\nFinal Average Score: {avg_score:.2f}")

    # 确保 scores 文件夹存在
    if not os.path.exists("scores"):
        os.makedirs("scores")

    # === 绘图 1: 训练 Loss 曲线 ===
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'BC Training Loss (Final Avg Score: {avg_score:.0f})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('scores/bc_training_loss.png')
    print("Plot saved: scores/bc_training_loss.png")
    plt.close() # 关闭画布防止重叠

    # === 绘图 2: 评估结果 (分数 + 及格线) ===
    plt.figure(figsize=(10, 6))
    
    # 画折线图
    episodes_x = range(1, eval_episodes + 1)
    plt.plot(episodes_x, scores, marker='o', linestyle='-', color='#1f77b4', label='BC Agent Score')
    
    # 画及格线 (475分)
    plt.axhline(y=475, color='r', linestyle='--', linewidth=2, label='Threshold (475)')
    
    # 设置样式
    plt.title(f'Evaluation Performance (Average: {avg_score:.1f})', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 520) # Y轴稍微留点空隙
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    # 在图中间添加分数框
    plt.text(eval_episodes/2, 250, f'Avg Score: {avg_score:.1f}', 
             fontsize=14, ha='center', 
             bbox=dict(facecolor='white', alpha=0.9, boxstyle='round'))

    # 保存
    plt.savefig('scores/bc_evaluation_result.png', dpi=300)
    print("Plot saved: scores/bc_evaluation_result.png")
    plt.close()

    env.close()

if __name__ == "__main__":
    train_and_evaluate()