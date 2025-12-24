from __future__ import annotations
import numpy as np
import gymnasium as gym
import torch
import time
import os
import config 
from collections import deque

from agents.simple_grpo_torch import SimpleGRPO
from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"
MODEL_DIR = "models

# 动作故障
ACTION_NOISE_PROB = 0.2     # 动作噪声 (0.1, 0.2)

# 环境噪声
STATE_NOISE_STD = 0.0       # 状态噪声 (0.01, 0.05)

# 奖励信号干扰
REWARD_NOISE_STD = 0.0      # 奖励噪声 (1.0)
SPARSE_REWARD = False       # 稀疏奖励 (True)

# 外部攻击 
ADVERSARIAL_ATTACK = False  # 对抗性推力开关
ATTACK_INTERVAL = 150       # 攻击频率
ATTACK_FORCE = 1.0          # 攻击强度

class RobustGRPOBatchTrainer:
    def __init__(self, env_name=ENV_NAME):
        self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        
        self.agent = SimpleGRPO(self.observation_space, self.action_space)
        
        self.batch_episodes = 16  # 每次采集多少局作为一个 Group
        self.max_steps_per_episode = 3200
        
    def collect_episode(self, episode_idx):
        state, _ = self.env.reset()
        
        #  初始状态噪声
        if STATE_NOISE_STD > 0:
            state += np.random.normal(0, STATE_NOISE_STD, state.shape)
            
        states, actions, rewards = [], [], []
        
        for step in range(self.max_steps_per_episode):
            # Agent决策 
            intended_action, _ = self.agent.act(state)
            
            # 动作噪声
            real_action = intended_action
            if np.random.rand() < ACTION_NOISE_PROB:
                real_action = self.env.action_space.sample()
            
            # 环境执行
            next_state, reward, terminated, truncated, _ = self.env.step(real_action)
            
            # 对抗性推力攻击 
            if ADVERSARIAL_ATTACK and step > 0 and step % ATTACK_INTERVAL == 0:
                if episode_idx % 10 == 0:
                   print(f"    Step:{step} -> GRPO 遭受攻击！")
                
                current_env_state = list(self.env.unwrapped.state)
                direction = 1 if np.random.rand() > 0.5 else -1
                current_env_state[3] += direction * ATTACK_FORCE
                self.env.unwrapped.state = tuple(current_env_state)
                
                # 更新物理状态
                next_state = np.array(current_env_state, dtype=np.float32)

            #  环境噪声
            if STATE_NOISE_STD > 0:
                noise = np.random.normal(0, STATE_NOISE_STD, next_state.shape)
                next_state += noise
            
            #  奖励噪声 
            if REWARD_NOISE_STD > 0:
                noise = np.random.normal(0, REWARD_NOISE_STD)
                reward += noise
                reward = max(0.1, reward) # 防止负分
            
            #  稀疏奖励
            done = terminated or truncated
            if SPARSE_REWARD:
                if not done:
                    reward = 0
                else:
                    reward = 100 if step >= 490 else -1
            
            # 记录数据
            states.append(state)
            actions.append(real_action)
            rewards.append(reward)
            
            state = next_state
            
            if done:
                break
        
        return states, actions, rewards
    
    def train_batch(self, num_episodes=config.TOTAL_EPISODES):
        suffix = ""
        if ACTION_NOISE_PROB > 0: suffix += f"_ActNoise_{ACTION_NOISE_PROB}"
        if STATE_NOISE_STD > 0: suffix += f"_StateNoise_{STATE_NOISE_STD}"
        if REWARD_NOISE_STD > 0: suffix += f"_RewNoise_{REWARD_NOISE_STD}"
        if SPARSE_REWARD: suffix += "_Sparse"
        if ADVERSARIAL_ATTACK: suffix += f"_AdvAttack_F{ATTACK_FORCE}_I{ATTACK_INTERVAL}"
        if not suffix: suffix = "_Baseline"
        
        logger_name = f"{ENV_NAME}_GRPO{suffix}"
        score_logger = ScoreLogger(logger_name)
        
        print(f" 启动 GRPO 鲁棒性训练: {logger_name} ")
        print(f"干扰: 动作={ACTION_NOISE_PROB}, 状态={STATE_NOISE_STD}, 奖励={REWARD_NOISE_STD}, 稀疏={SPARSE_REWARD}, 对抗={ADVERSARIAL_ATTACK}")
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        best_score = 0
        scores_window = deque(maxlen=100)
        
        total_batches = num_episodes // self.batch_episodes
        if total_batches < 1: total_batches = 1

        for batch_idx in range(1, total_batches + 1):
            batch_episodes_data = []
            batch_scores = []
            
            # 收集一个Group的数据
            for _ in range(self.batch_episodes):
                states, actions, rewards = self.collect_episode(batch_idx)
                batch_episodes_data.append((states, actions, rewards))
                
                score = len(rewards)
                batch_scores.append(score)
                scores_window.append(score)
            
            batch_avg_score = np.mean(batch_scores)
            
            actor_loss, info = self.agent.train_episode_batch(batch_episodes_data)
            
            score_logger.add_score(int(batch_avg_score), batch_idx * self.batch_episodes)
            
            if batch_avg_score >= best_score:
                best_score = batch_avg_score
                self.agent.save_model(
                    os.path.join(MODEL_DIR, "best_grpo_robust"), 
                    score=int(best_score)
                )
            
            if batch_idx % 2 == 0: # 每两个batch打印一次
                print(f"Batch {batch_idx}/{total_batches} | "
                      f"Score: {batch_avg_score:5.1f} | "
                      f"Best: {best_score:5.1f} | "
                      f"Loss: {actor_loss:.4f}")

        self.env.close()
        print(f"\nFinished: Best Score {best_score:.1f}")
        return self.agent

if __name__ == "__main__":
    trainer = RobustGRPOBatchTrainer()

    trainer.train_batch(num_episodes=config.TOTAL_EPISODES)
