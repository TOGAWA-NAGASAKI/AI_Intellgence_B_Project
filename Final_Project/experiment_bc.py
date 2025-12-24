import numpy as np
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import os
import time

# 导入 PPO (作为专家)
from agents.simple_ppo_torch import SimplePPO
# 导入 BC (作为学生)
from agents.behavioral_cloning import BCAgent, BCConfig

ENV_NAME = "CartPole-v1"

MODEL_PATH = r"Final_Project\models\best_ppo_ppotorch_score_500.pth"

def generate_dataset(expert_agent, env, num_samples, noise_level=0.0):
    """
    生成数据集
    noise_level: 0.0=全专家, 1.0=全随机
    """
    states = []
    actions = []
    
    obs, _ = env.reset(seed=int(time.time()))
    while len(states) < num_samples:
        
        # 决定动作
        if np.random.rand() < noise_level:
            # 噪音：随机动作
            action = env.action_space.sample()
        else:
            # PPO 专家动作 (确定性)
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(expert_agent.actor.net[0].weight.device)
            with torch.no_grad():
                probs = expert_agent.actor(state_t).cpu().numpy()[0]
            action = np.argmax(probs)
            
        states.append(obs)
        actions.append(action)
        
        next_obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
            
    return np.array(states), np.array(actions)

def evaluate_student(student_agent, env, episodes=10):
    scores = []
    for _ in range(episodes):
        state, _ = env.reset()
        steps = 0
        done = False
        while not done:
            action = student_agent.act(state, evaluation_mode=True)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        scores.append(steps)
    return np.mean(scores)

def run_experiments():
    # 1. 检查模型
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件: {MODEL_PATH}")
        return

    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    print(f"🚀 加载 PPO 专家模型...")
    expert = SimplePPO(obs_dim, act_dim)
    try:
        expert.load_model(MODEL_PATH)
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

   
    print("\n=== Experiment 1: Dataset Size vs Performance ===")
    size_candidates = [100, 500, 1000, 2000, 3000, 4000] 
    scores_by_size = []

    for size in size_candidates:
        print(f"Training with Size={size} (Pure Expert)...")
        states, actions = generate_dataset(expert, env, num_samples=size, noise_level=0.0)
        
        # 每次必须重新初始化学生，保证从头学起
        student = BCAgent(obs_dim, act_dim, BCConfig(epochs=20, lr=0.001))
        student.train_offline(states, actions)
        
        score = evaluate_student(student, env)
        scores_by_size.append(score)
        print(f"   -> Size: {size}, Score: {score:.1f}")

    print("\n=== Experiment 2: Noise Level vs Performance ===")
    FIXED_SIZE = 1000 
    
    noise_candidates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] # 0%, 30%, 60%, 100% 噪音
    scores_by_noise = []

    for noise in noise_candidates:
        print(f"Training with Noise={noise} (Fixed Size={FIXED_SIZE})...")
        states, actions = generate_dataset(expert, env, num_samples=FIXED_SIZE, noise_level=noise)
        
        student = BCAgent(obs_dim, act_dim, BCConfig(epochs=20, lr=0.001))
        student.train_offline(states, actions)
        
        score = evaluate_student(student, env)
        scores_by_noise.append(score)
        print(f"   -> Noise: {noise}, Score: {score:.1f}")

    env.close()

   
    plt.figure(figsize=(14, 6))

    # 子图 1: 数据量的影响
    plt.subplot(1, 2, 1)
    plt.plot(size_candidates, scores_by_size, marker='o', linewidth=2, color='#1f77b4')
    plt.title("Impact of Dataset Size (Pure Expert)", fontsize=14)
    plt.xlabel("Number of Samples", fontsize=12)
    plt.ylabel("Avg Score (Max 500)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 550)
    # 标注点
    for i, txt in enumerate(scores_by_size):
        plt.annotate(f"{txt:.0f}", (size_candidates[i], scores_by_size[i]+10), ha='center')

    # 子图 2: 噪音的影响
    plt.subplot(1, 2, 2)
    plt.plot(noise_candidates, scores_by_noise, marker='s', linewidth=2, color='#ff7f0e')
    plt.title(f"Impact of Data Quality (Fixed Size={FIXED_SIZE})", fontsize=14)
    plt.xlabel("Noise Level (0.0=Expert, 1.0=Random)", fontsize=12)
    plt.ylabel("Avg Score (Max 500)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 550)
    
    # 标注点
    for i, txt in enumerate(scores_by_noise):
        plt.annotate(f"{txt:.0f}", (noise_candidates[i], scores_by_noise[i]+10), ha='center')

    plt.tight_layout()
    
    # 保存图片
    save_path = "scores/bc_comprehensive_analysis.png"
    if not os.path.exists("scores"): os.makedirs("scores")
    plt.savefig(save_path)
    print(f"\n✅ 实验分析图已保存至: {save_path}")
    print("现在你可以把这张图直接贴到报告的 Advanced 章节了！")

if __name__ == "__main__":
    run_experiments()