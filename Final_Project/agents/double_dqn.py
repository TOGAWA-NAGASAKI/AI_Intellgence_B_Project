import random
from collections import deque
from typing import Deque, Tuple, List

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
GAMMA = 0.95  # Discount factor γ
LEARNING_RATE = 0.001  # Learning rate for Adam optimizer
MEMORY_SIZE = 100000  # Experience replay buffer capacity
BATCH_SIZE = 20  # Mini-batch size for learning
EXPLORATION_MAX = 1.0  # Initial ε (full exploration)
EXPLORATION_MIN = 0.01  # Minimum ε (greedy phase)
EXPLORATION_DECAY = 0.998  # Decay rate for ε per training step

class DoubleDQNSolver:
    def __init__(self, observation_space: int, action_space: int):
        # -----------------------------
        # 改进1: 添加目标网络相关参数
        # -----------------------------
        self.learn_step_counter: int = 0
        self.target_update_counter: int = 0
        self.target_update_freq: int = 25

        # Exploration schedule (保持不变)
        self.exploration_rate: float = EXPLORATION_MAX

        # Spaces (保持不变)
        self.action_space: int = action_space

        # Replay buffer (保持不变)
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=MEMORY_SIZE)

        # -----------------------------
        # 改进2: 创建主网络和目标网络
        # -----------------------------
        self.model = self._build_model(observation_space, action_space)
        self.target_model = self._build_model(observation_space, action_space)
        self._update_target_network()  # 初始化时同步目标网络

    def _build_model(self, observation_space: int, action_space: int) -> Sequential:
        """构建神经网络模型（保持不变）"""
        model = Sequential()
        model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.action_space, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    def _update_target_network(self) -> None:
        """更新目标网络权重"""
        self.target_model.set_weights(self.model.get_weights())

    # -----------------------------
    # Memory handling (保持不变)
    # -----------------------------
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool) -> None:
        """Store a single experience in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    # -----------------------------
    # Action selection (保持不变)
    # -----------------------------
    def act(self, state: np.ndarray) -> int:
        """
        Select an action using an ε-greedy policy.
        With probability ε: choose a random action.
        Otherwise: choose the action with the highest Q-value.
        """
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state, verbose=0)
        return int(np.argmax(q_values[0]))

    # -----------------------------
    # Learning - Double DQN核心改进
    # -----------------------------
    def experience_replay(self) -> None:
        """
        Double DQN核心改进：分离动作选择和Q值评估
        原始DQN问题：使用同一个网络选择动作和评估Q值，容易导致过高估计
        Double DQN解决方案：
          - 使用主网络选择最佳动作
          - 使用目标网络评估该动作的Q值
        """
        # Skip training until enough samples are collected
        if len(self.memory) < BATCH_SIZE or self.learn_step_counter % 4 != 0:
            self.learn_step_counter += 1
            return

        # Randomly sample a batch of transitions
        batch: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = random.sample(self.memory, BATCH_SIZE)

        # Stack batch components
        states = np.vstack([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.vstack([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])

        # -----------------------------
        # Double DQN核心改进部分
        # -----------------------------
        # 预测当前Q值（用于更新）
        q_values = self.model.predict(states, verbose=0)

        # -----------------------------
        # 改进3: Double DQN的动作选择和价值评估分离
        # -----------------------------
        # 步骤1: 使用主网络选择下一个状态的最佳动作
        q_next_main = self.model.predict(next_states, verbose=0)
        best_actions = np.argmax(q_next_main, axis=1)

        # 步骤2: 使用目标网络评估这些动作的Q值
        q_next_target = self.target_model.predict(next_states, verbose=0)

        # 计算目标值
        for i in range(BATCH_SIZE):
            target = rewards[i]
            if not dones[i]:
                # Double DQN核心公式:
                # target = r + γ * Q_target(s', argmax_a Q_main(s', a))
                target += GAMMA * q_next_target[i][best_actions[i]]

            # 只更新实际执行的动作的Q值
            q_values[i][actions[i]] = target

        # Train on the full batch once
        self.model.fit(states, q_values, verbose=0, epochs=1, batch_size=BATCH_SIZE)

        # Decay exploration rate
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

        # -----------------------------
        # 改进4: 定期更新目标网络
        # -----------------------------
        self.target_update_counter += 1
        if self.target_update_counter % self.target_update_freq == 0:
            self._update_target_network()