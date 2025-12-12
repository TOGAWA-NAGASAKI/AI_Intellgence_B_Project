import numpy as np
import gymnasium as gym
import os
from agents.simple_ppo import SimplePPO
import config

# 配置
ENV_NAME = "CartPole-v1"
OUTPUT_DIR = "data"
# 假设你已经训练好了一个PPO模型，在这里填入它的路径 (不要带后缀)
# 例如: checkpoints/checkpoint_episode_250_score_500
MODEL_PATH = "Final_Project/checkpoints/best_model_score_500" 
SAMPLES_TO_COLLECT = 5000  # 收集多少个 (state, action) 对

def generate_dataset():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    env = gym.make(ENV_NAME)
    # 初始化 PPO Agent 作为专家
    agent = SimplePPO(env.observation_space.shape[0], env.action_space.n)
    
    # 尝试加载模型，如果没有模型文件，生成的将是随机数据（即“菜鸟”数据）
    try:
        agent.load_model(MODEL_PATH)
        print(f"Loaded expert model from {MODEL_PATH}")
    except:
        print(f"Warning: Model not found at {MODEL_PATH}. Generating RANDOM data!")

    observations = []
    actions = []
    
    obs, _ = env.reset()
    collected_samples = 0
    episodes = 0
    total_score = 0
    
    print(f"Start collecting {SAMPLES_TO_COLLECT} samples...")

    while collected_samples < SAMPLES_TO_COLLECT:
        # PPO act 返回 action 和 probabilities，我们只需要 action
        # 注意：这里我们只记录 expert 的选择，不记录 reward，因为 BC 是监督学习
        _, probs = agent.act(obs)
        action = np.argmax(probs)
        
    
        
        observations.append(obs)
        actions.append(action)
        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_score += reward

        if done:
            episodes += 1
            obs, _ = env.reset()
        else:
            obs = next_obs
            
        collected_samples += 1
        
        if collected_samples % 1000 == 0:
            print(f"Collected {collected_samples}/{SAMPLES_TO_COLLECT} samples...")

    # 保存数据
    observations = np.array(observations)
    actions = np.array(actions)
    
    # 保存为 .npz 格式
    file_name = f"{OUTPUT_DIR}/expert_data_n{SAMPLES_TO_COLLECT}.npz"
    np.savez(file_name, obs=observations, actions=actions)
    
    avg_score = total_score / episodes if episodes > 0 else 0
    print(f"\nData collection finished!")
    print(f"Saved to: {file_name}")
    print(f"Expert Average Score during collection: {avg_score:.2f}")
    
    env.close()

if __name__ == "__main__":
    generate_dataset()