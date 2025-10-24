import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

GAMMA = 0.99
LAM = 0.95
CLIP_RATIO = 0.2
LR = 0.001
EPOCHS = 10
ENTROPY = 0.01


class SimplePPO:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        # Basic parameters are witten in the file config.py, I've trie for several times and it seemly the best.
        self.gamma = GAMMA
        self.lam = LAM
        self.clip_ratio = CLIP_RATIO
        self.learning_rate = LR
        self.epochs = EPOCHS
        self.entropy_coef = ENTROPY

        self.actor, self.critic = self._build_networks()
        self.old_actor = self._build_networks()[0]

    def _build_networks(self):
        # it's a simplified networks with RELU , actually you can try tanh.
        actor_input = Input(shape=(self.observation_space,))
        actor_dense1 = Dense(64, activation='relu')(actor_input)
        actor_dense2 = Dense(64, activation='relu')(actor_dense1)
        actor_output = Dense(self.action_space, activation='softmax')(actor_dense2)

        actor = Model(inputs=actor_input, outputs=actor_output)
        actor.compile(optimizer=Adam(learning_rate=self.learning_rate))

        critic_input = Input(shape=(self.observation_space,))
        critic_dense1 = Dense(64, activation='relu')(critic_input)
        critic_dense2 = Dense(64, activation='relu')(critic_dense1)
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