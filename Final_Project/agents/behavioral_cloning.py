import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass

# 自动选择设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1.添加 Config 类
@dataclass
class BCConfig:
    lr: float = 0.001
    epochs: int = 50
    batch_size: int = 64
    hidden_dim: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class BCNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(BCNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 2. 类名建议改为 BCAgent 以保持统一，或者你在 import 时要注意别名
class BCAgent:
    def __init__(self, obs_dim, act_dim, cfg=None):
        self.cfg = cfg if cfg else BCConfig()
        self.device = torch.device(self.cfg.device)
        
        # 初始化模型
        self.model = BCNetwork(obs_dim, act_dim, self.cfg.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.criterion = nn.CrossEntropyLoss()
        
    def act(self, state, evaluation_mode=True):
        """单次推理"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).to(self.device)
            else:
                state_tensor = state.to(self.device)
                
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)

            logits = self.model(state_tensor)
            action = torch.argmax(logits, dim=1).item()
        return action

    def step(self, state, action, reward, next_state, done):
        # 这里的 step 是为了兼容 train.py 接口
        # 但 BC 是离线学习，在线交互时不需要学习，所以留空即可
        pass

    def train_offline(self, states, actions):
        """
        专门用于实验脚本的离线训练函数
        """
        self.model.train()
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        # 转换为 Tensor
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)

        for epoch in range(self.cfg.epochs):
            np.random.shuffle(indices)
            epoch_loss = 0
            num_batches = 0
            
            for start_idx in range(0, dataset_size, self.cfg.batch_size):
                end_idx = min(start_idx + self.cfg.batch_size, dataset_size)
                batch_idx = indices[start_idx:end_idx]
                
                batch_states = states_tensor[batch_idx]
                batch_actions = actions_tensor[batch_idx]
                
                self.optimizer.zero_grad()
                logits = self.model(batch_states)
                loss = self.criterion(logits, batch_actions)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # 每 10 个 epoch 打印一次进度
            # if (epoch + 1) % 10 == 0:
            #     print(f"Epoch {epoch+1}/{self.cfg.epochs}, Loss: {epoch_loss/num_batches:.4f}")

    def save(self, path):
        import os
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))