"""
真正的GRPO训练器 - 批量收集多个episode进行训练
"""
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
MODEL_DIR = "models"

class GRPOBatchTrainer:
    def __init__(self, env_name=ENV_NAME):
        self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        
        self.agent = SimpleGRPO(self.observation_space, self.action_space)
        
        self.batch_episodes = 8
        self.max_steps_per_episode = 500
        
        self.episode_buffer = deque(maxlen=20)
        
    def collect_episode(self):
        state, _ = self.env.reset()
        states, actions, rewards = [], [], []
        
        for _ in range(self.max_steps_per_episode):
            action, _ = self.agent.act(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
            
            if done:
                break
        
        return states, actions, rewards
    
    def train_batch(self, num_episodes=5000):
        os.makedirs(MODEL_DIR, exist_ok=True)
        score_logger = ScoreLogger(f"{ENV_NAME}_RealGRPO")
        
        best_score = 0
        scores_window = deque(maxlen=100)
        done = False

        while not done:
            for batch_num in range(0, num_episodes, self.batch_episodes):
                batch_episodes_data = []
                batch_scores = []
                
                for _ in range(self.batch_episodes):
                    states, actions, rewards = self.collect_episode()
                    batch_episodes_data.append((states, actions, rewards))
                    score = len(rewards)
                    batch_scores.append(score)
                    scores_window.append(score)
                
                batch_avg_score = np.mean(batch_scores)
                overall_avg = np.mean(scores_window) if scores_window else batch_avg_score
                
                actor_loss, info = self.agent.train_episode_batch(batch_episodes_data)
                
                if batch_avg_score >= best_score and best_score >= 485:
                    best_score = batch_avg_score
                    self.agent.save_model(
                        os.path.join(MODEL_DIR, "best_real_grpo"), 
                        score=int(best_score)
                    )
                
                score_logger.add_score(int(batch_avg_score), batch_num // self.batch_episodes)
                
                if (batch_num // self.batch_episodes) % 5 == 0:
                    print(f"Batch {batch_num//self.batch_episodes:3d} | "
                        f"Score: {batch_avg_score:5.1f} | "
                        f"Avg100: {overall_avg:5.1f} | "
                        f"Actor Loss: {actor_loss:.4f}")
            self.env.close()
        self.agent.save_model(os.path.join(MODEL_DIR, "real_grpo_final"))
        print(f"\nFinished: {best_score:.1f}")
        
        return self.agent

if __name__ == "__main__":
    trainer = GRPOBatchTrainer()
    agent = trainer.train_batch(num_episodes=500)
    model_path = os.path.join(MODEL_DIR, "real_grpo_final.pth")