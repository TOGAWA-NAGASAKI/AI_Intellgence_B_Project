"""
Double DQN Robustness & Adversarial Attack Analysis (PyTorch)
-------------------------------------------------------------
Tests DDQN agent's performance under:
1. Action Perturbation (Slippery controls)
2. Reward Corruption (Noisy sensors)
3. Sparse Reward (Delayed gratification)
4. Adversarial Push (External force attack)
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
import torch
import time
import os
import config 

# 导入 Double DQN Agent
from agents.double_dqn_torch import DoubleDQNAgent
from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models"

# 鲁棒性与攻击设置 (每次只开一个进行测试)


# 内部故障模拟 
ACTION_NOISE_PROB = 0.15     # 动作噪声 (0.1, 0.15, 0.2)
REWARD_NOISE_STD = 0.0      # 奖励噪声 (1.0, 2.0)
SPARSE_REWARD = False       # 稀疏奖励 (True)

# 外部攻击模拟
ADVERSARIAL_ATTACK = False  # True开启
ATTACK_INTERVAL = 150       # 每隔多少步攻击一次
ATTACK_FORCE = 1.0          # 攻击强度 (0.5, 1.0, 2.0)


def train_ddqn_robust():
    # --- 自动生成日志后缀 ---
    suffix = ""
    if ACTION_NOISE_PROB > 0: suffix += f"_ActionNoise_{ACTION_NOISE_PROB}"
    if REWARD_NOISE_STD > 0: suffix += f"_RewardNoise_{REWARD_NOISE_STD}"
    if SPARSE_REWARD: suffix += "_Sparse"
    if ADVERSARIAL_ATTACK: suffix += f"_AdvAttack_F{ATTACK_FORCE}_I{ATTACK_INTERVAL}"
    
    if not suffix: suffix = "_Baseline"
        
    logger_name = f"{ENV_NAME}_DDQN{suffix}"
    score_logger = ScoreLogger(logger_name)
    
    print(f"--- 启动 Double DQN 鲁棒性训练: {logger_name} ---")
    print(f"干扰设置: 动作噪声={ACTION_NOISE_PROB}, 奖励噪声={REWARD_NOISE_STD}, 稀疏奖励={SPARSE_REWARD}, 对抗攻击={ADVERSARIAL_ATTACK}")

    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    # 初始化 DDQN Agent
    agent = DoubleDQNAgent(obs_dim, act_dim)
    
    print(f"[Info] Using device: {agent.device}")

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_score = 0

    # 使用 config 中的总局数
    for episode in range(1, config.TOTAL_EPISODES + 1):
        state, _ = env.reset(seed=episode)
        state = np.reshape(state, (1, obs_dim))
        
        steps = 0
        done = False
        
        while not done:
            steps += 1
            
            # 1. Agent 选择动作
            intended_action = agent.act(state)
            
            # --- 攻击 1: 动作干扰 (模拟电机/控制器故障) ---
            real_action = intended_action
            if np.random.rand() < ACTION_NOISE_PROB:
                real_action = env.action_space.sample() # 强制随机
            
            # 2. 环境执行动作
            next_state_raw, reward, terminated, truncated, _ = env.step(real_action)
            
            # 攻击 4: 对抗性推力攻击 (模拟外部冲击)
            if ADVERSARIAL_ATTACK and steps > 0 and steps % ATTACK_INTERVAL == 0:
                print(f"   ⚠️  Ep:{episode}, Step:{steps} -> DDQN 遭受攻击！")
                current_env_state = list(env.unwrapped.state)
                
                # 随机拍杆子
                direction = 1 if np.random.rand() > 0.5 else -1
                # 修改角速度
                current_env_state[3] += direction * ATTACK_FORCE
                
                env.unwrapped.state = tuple(current_env_state)
                # 更新 next_state，因为环境被改变了
                next_state_raw = np.array(current_env_state, dtype=np.float32)

            # --- 状态处理 ---
            done = terminated or truncated
            next_state = np.reshape(next_state_raw, (1, obs_dim))

            # 奖励逻辑整理
            
            if SPARSE_REWARD:
                # --- 攻击 3: 稀疏奖励模式 ---
                if not done:
                    reward = 0
                else:
                    # 只有真正通关才给大奖，失败给惩罚
                    reward = 100 if steps >= 490 else -1
            else:
                # --- 正常模式 (含动作噪声/对抗攻击/奖励噪声) ---
                
                # 1. 基础惩罚 (必加！否则DDQN学不会怕死)
                if done and steps < 500:
                    reward = -1.0
                
                # 2. 攻击 2: 奖励噪声 (只在正常模式下叠加)
                if REWARD_NOISE_STD > 0:
                    reward += np.random.normal(0, REWARD_NOISE_STD)

            # 3. Agent 学习 (注意：传入 real_action，即实际发生的动作)
            agent.step(state, real_action, reward, next_state, done)
            
            state = next_state

        # --- 记录与保存 ---
        epsilon = getattr(agent, 'exploration_rate', 0.0)
        score_logger.add_score(steps, episode)
        
        if steps > best_score: 
            best_score = steps
            agent.save(f"{checkpoint_dir}/best_ddqn_robust_model.torch")

        if episode % 10 == 0:
            print(f"Ep: {episode}, Epsilon: {epsilon:.3f}, Score: {steps}, Best: {best_score}")

    env.close()
    return agent

if __name__ == "__main__":
    train_ddqn_robust()