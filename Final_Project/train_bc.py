import numpy as np
import gymnasium as gym
import os
import time
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from agents.behavioral_cloning import BehavioralCloningAgent

# === 配置 ===
ENV_NAME = "CartPole-v1"
DATA_FILE = "data/expert_data_n5000.npz" # 确保文件名和你生成的一致
MODEL_PATH = "models/bc_agent_torch.pth"
BATCH_SIZE = 32
EPOCHS = 50

# 测试配置 (极速模式)
TEST_NUM_ENVS = 10      # 同时开10个环境
TEST_EPISODES = 100     # 总共测100局

def train_and_evaluate():
    # 1. 加载数据
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found!")
        return

    print(f"Loading data from {DATA_FILE}...")
    data = np.load(DATA_FILE)
    states = data['obs']
    actions = data['actions']

    # 2. 准备 PyTorch DataLoader
    # 将 Numpy 转换为 TensorDataset
    tensor_x = torch.FloatTensor(states)
    tensor_y = torch.LongTensor(actions)
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. 初始化 Agent
    env = gym.make(ENV_NAME)
    agent = BehavioralCloningAgent(env.observation_space.shape[0], env.action_space.n)

    # 4. 开始训练
    print(f"\n=== Starting Training on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} ===")
    if not os.path.exists("models"): os.makedirs("models")
    if not os.path.exists("scores"): os.makedirs("scores")

    loss_history = []
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        steps = 0
        
        for batch_states, batch_actions in dataloader:
            # 这里需要转回 numpy 传给 agent (或者你改一下 agent 直接收 tensor 也可以，但为了通用性这里转一下)
            loss = agent.train_step(batch_states.numpy(), batch_actions.numpy())
            epoch_loss += loss
            steps += 1
            
        avg_loss = epoch_loss / steps
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # 保存模型
    agent.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # 画 Loss 图
    plt.figure()
    plt.plot(loss_history)
    plt.title("BC Training Loss (PyTorch)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("scores/torch_bc_loss.png")
    plt.close()

    # ==========================================
    # 5. 极速并行测试 (Vectorized Evaluation)
    # ==========================================
    print(f"\n=== Starting High-Speed Evaluation ({TEST_EPISODES} episodes) ===")
    
    # 创建向量化环境 (10个进程同时跑)
    envs = gym.make_vec(ENV_NAME, num_envs=TEST_NUM_ENVS, vectorization_mode="async")
    
    start_time = time.time()
    
    states, _ = envs.reset()
    episode_counts = 0
    final_scores = []
    current_scores = np.zeros(TEST_NUM_ENVS)

    while episode_counts < TEST_EPISODES:
        # A. 批量预测
        actions = agent.batch_act(states)

        # B. 环境并行执行
        next_states, rewards, terminations, truncations, _ = envs.step(actions)
        
        # === [关键修改] ===
        # 原来的写法: current_scores += 1 (容易多算一步)
        # 现在的写法: current_scores += rewards (以环境反馈为准)
        current_scores += rewards 
        # =================
        
        dones = terminations | truncations

        for i in range(TEST_NUM_ENVS):
            if dones[i]:
                # 注意：Gymnasium 的 VectorEnv 在 done 时会自动 reset
                # 这里的 final_score 就是这一局的最终得分
                final_scores.append(current_scores[i])
                episode_counts += 1
                current_scores[i] = 0 # 重置分数
                
        if episode_counts >= TEST_EPISODES:
            break
            
        states = next_states

    duration = time.time() - start_time
    envs.close()

    # === 6. 结果可视化 ===
    avg_score = np.mean(final_scores)
    
    print("\n" + "="*30)
    print(f"PyTorch BC Evaluation Results")
    print("="*30)
    print(f"Total Episodes: {len(final_scores)}")
    print(f"Total Time:     {duration:.2f}s")
    print(f"Speed:          {len(final_scores)/duration:.1f} eps/sec")
    print(f"Average Score:  {avg_score:.2f}")
    print("="*30)

    # 画评估结果图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(final_scores)+1), final_scores, marker='o', linestyle='-', color='#1f77b4', label='BC Agent (Torch)')
    plt.axhline(y=475, color='r', linestyle='--', label='Threshold (475)')
    plt.title(f'Evaluation Performance (Avg: {avg_score:.1f})')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.ylim(0, 520)
    plt.legend()
    plt.text(len(final_scores)/2, 250, f'Avg: {avg_score:.1f}', 
             ha='center', bbox=dict(facecolor='white', alpha=0.9))
    plt.savefig('scores/torch_evaluation.png')
    print("Evaluation plot saved to scores/torch_evaluation.png")

if __name__ == "__main__":
    train_and_evaluate()