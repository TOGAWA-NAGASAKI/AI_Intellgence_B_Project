import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import os

import json
import pickle
import pandas as pd
from datetime import datetime
import config

class SimplePPO:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.best_score = 0

        # Basic parameters are witten in the file config.py, I've trie for several times and it seemly the best.
        self.gamma = config.GAMMA
        self.lam = config.LAM
        self.clip_ratio = config.CLIP_RATIO
        self.learning_rate = config.LR
        self.epochs = config.EPOCHS
        self.entropy_coef = config.ENTROPY

        self.actor, self.critic = self._build_networks()
        self.old_actor = self._build_networks()[0]

    def _build_networks(self):
        # it's a simplified networks with RELU , actually you can try tanh.
        actor_input = Input(shape=(self.observation_space,))
        actor_dense1 = Dense(config.UNITS, activation='relu')(actor_input)
        actor_dense2 = Dense(config.UNITS, activation='relu')(actor_dense1)
        actor_output = Dense(self.action_space, activation='softmax')(actor_dense2)

        actor = Model(inputs=actor_input, outputs=actor_output)
        actor.compile(optimizer=Adam(learning_rate=self.learning_rate))

        critic_input = Input(shape=(self.observation_space,))
        critic_dense1 = Dense(config.UNITS, activation='relu')(critic_input)
        critic_dense2 = Dense(config.UNITS, activation='relu')(critic_dense1)
        critic_output = Dense(1, activation='linear')(critic_dense2)

        critic = Model(inputs=critic_input, outputs=critic_output)
        critic.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')

        return actor, critic


    def act(self, state):
        state = state.reshape(1, -1)
        probabilities = self.actor.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_space, p=probabilities)
        return action, probabilities

    def compute_returns_advantages(self, rewards, values, dones, next_value=0):
        # I use GAE here and it performs well.
        returns = []
        advantages = []
        gae = 0
        values = values + [next_value]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])

        return np.array(returns), np.array(advantages)

    def train_episode(self, states, actions, rewards, dones):
        if len(states) < 4:
            return None, None

        # Here are math functions we needn't care about
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        values = self.critic.predict(states, verbose=0).flatten()
        next_value = 0

        returns, advantages = self.compute_returns_advantages(
            rewards.tolist(), values.tolist(), dones.tolist(), next_value
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.old_actor.set_weights(self.actor.get_weights())

        old_probs = self.old_actor.predict(states, verbose=0)
        old_action_probs = old_probs[np.arange(len(actions)), actions]

        actor_losses = []
        critic_losses = []

        # Here the main idea of PPO, which contains Clip() , entropy , updating actors and critics......
        for epoch in range(self.epochs):
            # 更新Actor
            with tf.GradientTape() as tape:
                new_probs = self.actor(states, training=True)
                new_action_probs = tf.gather(new_probs, actions, batch_dims=1)

                ratio = new_action_probs / (old_action_probs + 1e-8)
                surr1 = ratio * advantages
                surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

                entropy = -tf.reduce_mean(tf.reduce_sum(new_probs * tf.math.log(new_probs + 1e-8), axis=1))
                actor_loss = policy_loss - self.entropy_coef * entropy

            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            critic_loss = self.critic.train_on_batch(states, returns.reshape(-1, 1))

            actor_losses.append(actor_loss.numpy())
            critic_losses.append(critic_loss)

        return np.mean(actor_losses), np.mean(critic_losses)
    
    def save_model(self, filepath, score=None):
        if score is not None:
            base_path = f"{filepath}_score_{int(score)}"
        else:
            base_path = filepath
        os.makedirs(os.path.dirname(base_path) if os.path.dirname(base_path) else '.', exist_ok=True)
        
        actor_path = f"{base_path}_actor.weights.h5"
        critic_path = f"{base_path}_critic.weights.h5"
        old_actor_path = f"{base_path}_old_actor.weights.h5"
        
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        self.old_actor.save_weights(old_actor_path)
        
        print(f"Saved")
        print(f"   Actor: {actor_path}")
        print(f"   Critic: {critic_path}")
    
    def load_model(self, filepath):
        actor_path = f"{filepath}_actor.weights.h5"
        critic_path = f"{filepath}_critic.weights.h5"
        
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
        self.old_actor.load_weights(actor_path)
        
        print(f"Loaded: {filepath}")