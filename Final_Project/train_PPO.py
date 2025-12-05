import numpy as np
import gymnasium as gym
from agents.simple_ppo import SimplePPO
from scores.score_logger import ScoreLogger
import time

# some tricks to speed up

import tensorflow as tf
import os
import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

ENV_NAME = "CartPole-v1"


def train_ppo():
    # Set up
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    score_logger = ScoreLogger(ENV_NAME)
    agent = SimplePPO(observation_space, action_space)

    print("Starting PPO training...")

    start_time = time.time()
    best_score = config.BEST_SCORE
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    training_history = {
        'episode': [],
        'score': [],
        'actor_loss': [],
        'critic_loss': [],
        'time': []
    }
    # Actually, 300 episodes are enough for passing the task.
    for episode in range(1, 301):
        state, _ = env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_dones = []

        step = 0
        while step < 500:
            action, prob = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_dones.append(done)

            state = next_state
            step += 1

            if done:
                break

        if len(episode_states) > 0:
            actor_loss, critic_loss = agent.train_episode(
                episode_states, episode_actions, episode_rewards, episode_dones
            )

        # Mark down like a notebook, in order to speed up and simplify, I choose to sum up the Actor Loss once 10 eps.

        score = len(episode_rewards)
        if score >= best_score:
            best_score = score
            agent.save_model(f"{checkpoint_dir}/best_model", score=score)

        if episode % 10 == 0:
            print(f"Ep: {episode:4d}, Score: {score:3d}")
            if actor_loss is not None:
                print(f"Actor: {actor_loss:7.4f}, Critic: {critic_loss:7.2f}")
        if episode % 25 == 0:
            agent.save_model(f"{checkpoint_dir}/checkpoint_episode_{episode}")

        score_logger.add_score(score, episode)
        training_history['episode'].append(episode)
        training_history['score'].append(score)
        training_history['actor_loss'].append(actor_loss if actor_loss is not None else 0)
        training_history['critic_loss'].append(critic_loss if critic_loss is not None else 0)
        training_history['time'].append(time.time() - start_time)

    env.close()

    total_time = time.time() - start_time
    print(f"\n=== Training Complete ===")
    print(f"Total time: {total_time / 60:.1f} minutes")
    agent.save_model(f"{checkpoint_dir}/final_model")
    return agent


if __name__ == "__main__":
    trained_agent = train_ppo()