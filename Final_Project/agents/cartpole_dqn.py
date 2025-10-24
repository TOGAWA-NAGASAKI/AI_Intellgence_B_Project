"""
CartPole DQN Agent (Optimized Teaching Version)
-----------------------------------------------
This module defines a simple Deep Q-Network (DQN) agent for the CartPole-v1 environment.

Compared with the baseline:
  - Uses vectorized batch updates (1 fit() per batch instead of many).
  - Includes clear pedagogical comments and TODO placeholders for exploration and learning logic.
  - Designed for educational use in STA303 / Reinforcement Learning coursework.

Students are expected to:
  1) Understand how the replay buffer and epsilon-greedy exploration work.
  2) Implement or experiment with different architectures or algorithms (Double DQN, PPO, etc.).
"""

import random
from collections import deque
from typing import Deque, Tuple, List

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# -----------------------------
# Hyperparameters (can be tuned)
# -----------------------------
GAMMA = 0.95                # Discount factor γ
LEARNING_RATE = 0.001       # Learning rate for Adam optimizer
MEMORY_SIZE = 100000       # Experience replay buffer capacity
BATCH_SIZE = 20             # Mini-batch size for learning
EXPLORATION_MAX = 1.0       # Initial ε (full exploration)
EXPLORATION_MIN = 0.01      # Minimum ε (greedy phase)
EXPLORATION_DECAY = 0.995   # Decay rate for ε per training step


class DQNSolver:
    """
    Deep Q-Network agent that learns to balance the CartPole.
    """

    def __init__(self, observation_space: int, action_space: int):
        # Exploration schedule
        self.exploration_rate: float = EXPLORATION_MAX

        # Spaces
        self.action_space: int = action_space

        # Replay buffer: stores (state, action, reward, next_state, done)
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=MEMORY_SIZE)

        # -----------------------------
        # Q-network definition
        # -----------------------------
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))

    # -----------------------------
    # Memory handling
    # -----------------------------
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool) -> None:
        """Store a single experience in the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    # -----------------------------
    # Action selection
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
    # Learning
    # -----------------------------
    def experience_replay(self) -> None:
        """
        Sample a mini-batch from memory and perform one optimization step.
        Uses vectorized batch updates for efficiency.
        """
        # Skip training until enough samples are collected
        if len(self.memory) < BATCH_SIZE:
            return

        # Randomly sample a batch of transitions
        batch: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = random.sample(self.memory, BATCH_SIZE)

        # Stack batch components
        states = np.vstack([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.vstack([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])

        # Predict current and next Q-values
        q_values = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)

        # Compute target values
        for i in range(BATCH_SIZE):
            target = rewards[i]
            if not dones[i]:
                target += GAMMA * np.max(q_next[i])
            q_values[i][actions[i]] = target

        # Train on the full batch once
        self.model.fit(states, q_values, verbose=0, epochs=1, batch_size=BATCH_SIZE)

        # Decay exploration rate
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)