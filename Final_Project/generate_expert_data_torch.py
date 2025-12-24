import os
import numpy as np
import gymnasium as gym
import torch
from agents.simple_ppo_torch import SimplePPO
from agents.double_dqn_torch import DoubleDQNAgent 

# === ⚙️ 配置区域 (根据你的截图修改) ===
ENV_NAME = "CartPole-v1"
OUTPUT_DIR = "data"
OUTPUT_FILENAME = "expert_data_n5000.npz"
NUM_SAMPLES = 5000  # 生成多少条数据


ALGO = "ppo"
MODEL_PATH = "Final_Project/models/best_ppo_ppotorch_score_500.pth"



def generate_data():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件: {MODEL_PATH}")
        print("请检查文件名是否与截图一致！")
        return

    # 1初始化环境
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # 加载 Agent
    print(f" 正在加载 {ALGO.upper()} 模型: {MODEL_PATH} ...")
    
    if ALGO == "ppo":
        agent = SimplePPO(obs_dim, act_dim)
        # PPO 加载逻辑
        agent.load_model(MODEL_PATH) 
        
    elif ALGO == "doubledqn":
        agent = DoubleDQNAgent(obs_dim, act_dim)
        # DoubleDQN 加载逻辑
        agent.load(MODEL_PATH)
        
    else:
        raise ValueError("Unknown Algorithm")

    # 开始采集数据
    collected_states = []
    collected_actions = []
    
    obs, _ = env.reset(seed=2024)
    current_samples = 0
    
    print("开始采集专家演示数据...")
    
    while current_samples < NUM_SAMPLES:
        # === 核心：获取确定的专家动作 (去除随机性) ===
        if ALGO == "doubledqn":
            # DoubleDQN 提供了 evaluation_mode
            state_in = np.reshape(obs, (1, obs_dim))
            action = agent.act(state_in, evaluation_mode=True)
            
        elif ALGO == "ppo":
            # PPO 需要手动取最大概率 (Argmax)，确保是专家行为
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(agent.actor.net[0].weight.device)
            with torch.no_grad():
                probs = agent.actor(state_t).cpu().numpy()[0]
            action = np.argmax(probs) # 取概率最大的动作

        # 记录数据
        collected_states.append(obs)
        collected_actions.append(action)
        
        # 环境推进一步
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs
            
        current_samples += 1
        if current_samples % 1000 == 0:
            print(f"   已收集: {current_samples}/{NUM_SAMPLES}")

    # 保存为 .npz 格式
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    # 截取到刚好 NUM_SAMPLES 个
    final_obs = np.array(collected_states)[:NUM_SAMPLES]
    final_acts = np.array(collected_actions)[:NUM_SAMPLES]
    
    np.savez(save_path, obs=final_obs, actions=final_acts)
    
    print("\n" + "="*40)
    print(f" 成功！专家数据已保存至: {save_path}")
    print(f" 数据形状: States={final_obs.shape}, Actions={final_acts.shape}")
    print("="*40)
    print("👉 下一步: 运行 'python behavioral_cloning.py' (或你的BC训练脚本) 来训练模仿者。")

if __name__ == "__main__":
    generate_data()