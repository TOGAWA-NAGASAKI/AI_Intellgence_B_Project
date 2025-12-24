import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import copy
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, units):
        super(ActorNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, units),
            nn.ReLU(),
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, act_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

class SimpleGRPO:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.best_score = 0
        self.gamma = config.GAMMA
        self.clip_ratio = config.CLIP_RATIO
        self.learning_rate = config.LR
        self.epochs = config.EPOCHS
        self.entropy_coef = config.ENTROPY
        self.units = config.UNITS
        
        self.group_size = config.GRPO_GROUP_SIZE
        self.temperature = config.GRPO_TEMPERATURE
        self.use_monte_carlo = True
        
        self.actor = ActorNetwork(observation_space, action_space, self.units).to(device)
        
        self.old_actor = copy.deepcopy(self.actor)
        self.old_actor.eval()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)

    def act(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            probabilities = self.actor(state_t)
        
        probs_np = probabilities.cpu().numpy()[0]
        action = np.random.choice(self.action_space, p=probs_np)
        return action, probs_np

    def compute_monte_carlo_returns(self, rewards, gamma):
        returns = []
        R = 0
        
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        return np.array(returns)

    def compute_group_relative_advantages(self, rewards, returns):
        if len(returns) < self.group_size:
            advantages = returns - np.mean(returns)
            return advantages / (np.std(advantages) + 1e-8)
        
        n_groups = len(returns) // self.group_size
        grouped_returns = []
        group_indices = []
        
        for i in range(n_groups):
            start_idx = i * self.group_size
            end_idx = start_idx + self.group_size
            group_returns = returns[start_idx:end_idx]
            grouped_returns.append(group_returns)
            group_indices.append((start_idx, end_idx))
        
        remainder_start = n_groups * self.group_size
        if remainder_start < len(returns):
            grouped_returns.append(returns[remainder_start:])
            group_indices.append((remainder_start, len(returns)))
        
        all_advantages = np.zeros_like(returns)
        
        for i, (group_returns, (start, end)) in enumerate(zip(grouped_returns, group_indices)):
            group_mean = np.mean(group_returns)
            group_std = np.std(group_returns) + 1e-8
            
            normalized = (group_returns - group_mean) / group_std
            
            scaled = normalized / self.temperature
            
            exp_scaled = np.exp(scaled - np.max(scaled))
            relative_probs = exp_scaled / (np.sum(exp_scaled) + 1e-8)
            
            group_advantages = relative_probs * group_std + group_mean
            
            group_advantages = group_advantages - np.mean(group_advantages)
            
            all_advantages[start:end] = group_advantages
        
        return all_advantages

    def train_episode_batch(self, episodes_data):
        if len(episodes_data) < 2:
            return None,
        
        all_states = []
        all_actions = []
        all_log_probs = []
        all_returns = []
        all_rewards = []
        
        for states, actions, rewards in episodes_data:
            returns = self.compute_monte_carlo_returns(rewards, self.gamma)
            
            states_t = torch.FloatTensor(np.array(states)).to(device)
            actions_t = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
            
            with torch.no_grad():
                probs = self.old_actor(states_t)
                log_probs = torch.log(probs.gather(1, actions_t) + 1e-8)
            
            all_states.extend(states)
            all_actions.extend(actions)
            all_log_probs.extend(log_probs.cpu().numpy().flatten())
            all_returns.extend(returns)
            all_rewards.extend(rewards)
        
        all_states_np = np.array(all_states)
        all_actions_np = np.array(all_actions)
        all_log_probs_np = np.array(all_log_probs)
        all_returns_np = np.array(all_returns)
        
        advantages_np = self.compute_group_relative_advantages(all_rewards, all_returns_np)
        
        advantages_np = (advantages_np - advantages_np.mean()) / (advantages_np.std() + 1e-8)
        
        states_t = torch.FloatTensor(all_states_np).to(device)
        actions_t = torch.LongTensor(all_actions_np).unsqueeze(1).to(device)
        old_log_probs_t = torch.FloatTensor(all_log_probs_np).unsqueeze(1).to(device)
        advantages_t = torch.FloatTensor(advantages_np).unsqueeze(1).to(device)
        
        self.old_actor.load_state_dict(self.actor.state_dict())
        
        actor_losses = []
        
        for _ in range(self.epochs):
            new_probs = self.actor(states_t)
            new_log_probs = torch.log(new_probs.gather(1, actions_t) + 1e-8)
            
            ratio = torch.exp(new_log_probs - old_log_probs_t)
            
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()
            
            entropy = -(new_probs * torch.log(new_probs + 1e-8)).sum(dim=1).mean()
            
            actor_loss = policy_loss - self.entropy_coef * entropy
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            actor_losses.append(actor_loss.item())
        
        return np.mean(actor_losses), f"训练了 {len(episodes_data)} 个episode，总步数: {len(all_states)}"

    def train_episode(self, states, actions, rewards, dones):
        returns = self.compute_monte_carlo_returns(rewards, self.gamma)
        
        advantages = self.compute_group_relative_advantages(rewards, returns)
        
        states_t = torch.FloatTensor(np.array(states)).to(device)
        actions_t = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
        
        with torch.no_grad():
            old_probs = self.old_actor(states_t)
            old_log_probs = torch.log(old_probs.gather(1, actions_t) + 1e-8)
        
        self.old_actor.load_state_dict(self.actor.state_dict())
        
        advantages_t = torch.FloatTensor(advantages).unsqueeze(1).to(device)
        
        actor_losses = []
        
        for _ in range(self.epochs):
            new_probs = self.actor(states_t)
            new_log_probs = torch.log(new_probs.gather(1, actions_t) + 1e-8)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()
            
            entropy = -(new_probs * torch.log(new_probs + 1e-8)).sum(dim=1).mean()
            
            actor_loss = policy_loss - self.entropy_coef * entropy
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            actor_losses.append(actor_loss.item())
        
        return np.mean(actor_losses), None
    
    def save_model(self, filepath, score=None):
        if score is not None:
            base_path = f"{filepath}_realgrpo_score_{int(score)}"
        else:
            base_path = filepath
        os.makedirs(os.path.dirname(base_path) if os.path.dirname(base_path) else '.', exist_ok=True)
        
        path = f"{base_path}.pth"
        
        torch.save({
            'actor': self.actor.state_dict(),
            'old_actor': self.old_actor.state_dict()
        }, path)
        
        print(f"Saved Real GRPO model to: {path}")
    
    def load_model(self, filepath):
        if not filepath.endswith('.pth'):
            path = f"{filepath}.pth"
        else:
            path = filepath
            
        checkpoint = torch.load(path, map_location=device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.old_actor.load_state_dict(checkpoint['old_actor'])
        
        print(f"Loaded Real GRPO model from: {path}")