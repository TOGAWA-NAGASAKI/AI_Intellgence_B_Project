import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 自动选择设备：如果有显卡就用显卡，没有就用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BCNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(BCNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x) # 输出 Raw Logits

class BehavioralCloningAgent:
    def __init__(self, state_dim, action_dim, lr=0.001):
        self.model = BCNetwork(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_step(self, states, actions):
        """
        这就是 PyTorch 版本特有的训练函数，
        你之前的 Keras 版本没有这个函数，所以报错了。
        """
        self.model.train()
        
        # 转换数据到 Tensor 并移到设备上
        states_tensor = torch.FloatTensor(states).to(DEVICE)
        actions_tensor = torch.LongTensor(actions).to(DEVICE) # 分类标签必须是 Long
        
        # 1. 清空梯度
        self.optimizer.zero_grad()
        
        # 2. 前向传播
        logits = self.model(states_tensor)
        
        # 3. 计算 Loss
        loss = self.criterion(logits, actions_tensor)
        
        # 4. 反向传播与更新
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def act(self, state):
        """单次推理 (用于普通评估)"""
        self.model.eval()
        with torch.no_grad():
            # 兼容：如果输入是numpy数组，转tensor
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).to(DEVICE)
            else:
                state_tensor = state.to(DEVICE)
                
            # 如果维度是 (4,) 增加 batch 维度 -> (1, 4)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)

            logits = self.model(state_tensor)
            action = torch.argmax(logits, dim=1).item()
        return action

    def batch_act(self, states):
        """批量推理 (用于多线程极速评估)"""
        self.model.eval()
        with torch.no_grad():
            # states 已经是 (Batch, 4) 的 numpy 数组
            states_tensor = torch.FloatTensor(states).to(DEVICE)
            logits = self.model(states_tensor)
            actions = torch.argmax(logits, dim=1)
        return actions.cpu().numpy() # 转回 Numpy

    def save(self, path):
        # 确保保存文件夹存在
        import os
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))