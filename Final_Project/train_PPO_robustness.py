from __future__ import annotations
import numpy as np
import gymnasium as gym
import torch
import time
import os
import config 

from agents.simple_ppo_torch import SimplePPO
from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"


# 1. 内部故障故障模拟 
ACTION_NOISE_PROB = 0.0     # 动作噪声 (0.1, 0.15, 0.2)

# 2. 传感器故障模拟 
STATE_NOISE_STD = 0.05       # 状态噪声 (0.01, 0.05, 0.1)

# 3. 奖励信号干扰 
REWARD_NOISE_STD = 0.00      # 奖励噪声  (0.1, 0.5, 1.0)
SPARSE_REWARD = False       # 稀疏奖励 (True)

# 4. 外部物理攻击 
ADVERSARIAL_ATTACK = False   # True开启 (对抗性推力)
ATTACK_INTERVAL = 150       # 每隔多少步攻击一次
ATTACK_FORCE = 1.0          # 攻击强度

def train_ppo_robust():
    #自动生成日志后缀
    suffix = ""
    if ACTION_NOISE_PROB > 0: suffix += f"_ActNoise_{ACTION_NOISE_PROB}"
    if STATE_NOISE_STD > 0: suffix += f"_StateNoise_{STATE_NOISE_STD}"
    if REWARD_NOISE_STD > 0: suffix += f"_RewNoise_{REWARD_NOISE_STD}"
    if SPARSE_REWARD: suffix += "_Sparse"
    if ADVERSARIAL_ATTACK: suffix += f"_AdvAttack_F{ATTACK_FORCE}_I{ATTACK_INTERVAL}"
    
    if not suffix: suffix = "_Baseline"
        
    logger_name = f"{ENV_NAME}_PPO{suffix}"
    score_logger = ScoreLogger(logger_name)
    
    print(f"--- 启动 PPO 训练: {logger_name} ---")
    print(f"干扰设置: 动作噪声={ACTION_NOISE_PROB}, 状态噪声={STATE_NOISE_STD}, 奖励噪声={REWARD_NOISE_STD}, 稀疏奖励={SPARSE_REWARD}, 对抗攻击={ADVERSARIAL_ATTACK}")

    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    agent = SimplePPO(observation_space, action_space)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_score = 0

    for episode in range(1, config.TOTAL_EPISODES + 1):
        state, _ = env.reset()
        
        # 初始状态噪声
        if STATE_NOISE_STD > 0:
            state += np.random.normal(0, STATE_NOISE_STD, state.shape)
            
        episode_states, episode_actions, episode_rewards, episode_dones = [], [], [], []
        step = 0
        
        while step < 500:
            # Agent基于(可能带有噪声的)状态做出决策
            intended_action, prob = agent.act(state)
            
            # 动作干扰
            real_action = intended_action
            if np.random.rand() < ACTION_NOISE_PROB:
                real_action = env.action_space.sample()
            
            # 环境执行动作
            next_state, reward, terminated, truncated, _ = env.step(real_action)
            
            # 对抗性推力攻击 
            if ADVERSARIAL_ATTACK and step > 0 and step % ATTACK_INTERVAL == 0:
                current_env_state = list(env.unwrapped.state)
                direction = 1 if np.random.rand() > 0.5 else -1
                current_env_state[3] += direction * ATTACK_FORCE
                
                env.unwrapped.state = tuple(current_env_state)
                # 更新真实的物理状态
                next_state = np.array(current_env_state, dtype=np.float32)

            # 状态噪声 
            if STATE_NOISE_STD > 0:
                noise = np.random.normal(0, STATE_NOISE_STD, next_state.shape)
                next_state += noise
            
            # 奖励干扰
            done = terminated or truncated

            if REWARD_NOISE_STD > 0:
                # 叠加噪声
                noise = np.random.normal(0, REWARD_NOISE_STD)
                reward += noise
                reward = max(0.1, reward) 
            
            if SPARSE_REWARD:
                if not done:
                    reward = 0
                else:
                    reward = 100 if step >= 490 else -1        

            # 存储经验
            episode_states.append(state)
            episode_actions.append(real_action)
            episode_rewards.append(reward)
            episode_dones.append(done)

            state = next_state
            step += 1
            if done: break

        # --- 训练逻辑 ---
        if len(episode_states) > 0:
            # 多练几轮
            is_hard_mode = (STATE_NOISE_STD > 0 or ADVERSARIAL_ATTACK or REWARD_NOISE_STD > 0 or SPARSE_REWARD)
            train_loops = 3 if is_hard_mode else 1
            
            for _ in range(train_loops):
                agent.train_episode(episode_states, episode_actions, episode_rewards, episode_dones)

        score = step 
        score_logger.add_score(score, episode)
        
        if score >= best_score:
            best_score = score
            agent.save_model(f"{checkpoint_dir}/best_robust_model")

        if episode % 10 == 0:
            print(f"Ep: {episode}, Score: {score}, Best: {best_score}")

    env.close()
    return agent

if __name__ == "__main__":
    train_ppo_robust()
