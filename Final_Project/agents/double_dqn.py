import random
import config
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
GAMMA = config.DQNGAMMA
LEARNING_RATE = config.LEARNING_RATE
MEMORY_SIZE = config.MEMORY_SIZE
BATCH_SIZE = config.BATCH_SIZE
EXPLORATION_MAX = config.EXPLORATION_MAX
EXPLORATION_MIN = config.EXPLORATION_MIN
EXPLORATION_DECAY =config.EXPLORATION_DECAY
UNIT = config.DQNUNITS

class DoubleDQNSolver:
    def __init__(self, observation_space: int, action_space: int):
        self.learn_step_counter: int = 0
        self.target_update_counter: int = 0
        self.target_update_freq: int = 25
        self.exploration_rate: float = EXPLORATION_MAX
        self.action_space: int = action_space
        self.memory: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=MEMORY_SIZE)
        self.model = self._build_model(observation_space, action_space)
        self.target_model = self._build_model(observation_space, action_space)
        self._update_target_network()

    def _build_model(self, observation_space: int, action_space: int) -> Sequential:
        model = Sequential()
        model.add(Dense(UNIT, input_shape=(observation_space,), activation="relu"))
        model.add(Dense(UNIT, activation="relu"))
        model.add(Dense(self.action_space, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    def _update_target_network(self) -> None:
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

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

    def experience_replay(self) -> None:
        # Skip training until enough samples are collected
        if len(self.memory) < BATCH_SIZE or self.learn_step_counter % 4 != 0:
            self.learn_step_counter += 1
            return
        
        batch: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = random.sample(self.memory, BATCH_SIZE)
        states = np.vstack([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.vstack([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])

        q_values = self.model.predict(states, verbose=0)
        q_next_main = self.model.predict(next_states, verbose=0)
        best_actions = np.argmax(q_next_main, axis=1)
        q_next_target = self.target_model.predict(next_states, verbose=0)

        for i in range(BATCH_SIZE):
            target = rewards[i]
            if not dones[i]:
                # target = r + γ * Q_target(s', argmax_a Q_main(s', a))
                target += GAMMA * q_next_target[i][best_actions[i]]
            q_values[i][actions[i]] = target

        self.model.fit(states, q_values, verbose=0, epochs=1, batch_size=BATCH_SIZE)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

        self.target_update_counter += 1
        if self.target_update_counter % self.target_update_freq == 0:
            self._update_target_network()

    def save_weights(self, filepath):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        self.model.save_weights(f"{filepath}_ddqn_model.weights.h5")
        self.target_model.save_weights(f"{filepath}_ddqn_target.weights.h5")
        print(f"Saved: {filepath}")
    
    def load_weights(self, filepath):
        self.model.load_weights(f"{filepath}_ddqn_model.weights.h5")
        self.target_model.load_weights(f"{filepath}_ddqn_target.weights.h5")
        print(f"Loaded: {filepath}")
    
    def save_checkpoint(self, checkpoint_dir, episode, score=None):
        if score is not None:
            filename = f"{checkpoint_dir}/checkpoint_ep{episode:04d}_score{score}"
        else:
            filename = f"{checkpoint_dir}/checkpoint_ep{episode:04d}"
        
        self.save_weights(filename)
    
    def save_best_model(self, score, checkpoint_dir="models"):
        filename = f"{checkpoint_dir}/best_double_dqn_score{score}"
        self.save_weights(filename)