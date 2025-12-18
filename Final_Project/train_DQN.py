"""
CartPole Training Script (Gymnasium Version)
--------------------------------------------
Main training loop that runs the CartPole-v1 environment using a DQN agent.
The training progress is logged and plotted using ScoreLogger.
"""

import numpy as np
import gymnasium as gym
from agents.double_dqn import DoubleDQNSolver
from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"


def train():
    # Initialize environment (Gymnasium)
    env = gym.make(ENV_NAME)
    score_logger = ScoreLogger(ENV_NAME)

    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    dqn_solver = DoubleDQNSolver(observation_space, action_space)

    # Train for a fixed number of episodes
    for run in range(1, 101):  # 100 times
        state, info = env.reset(seed=run) # reset the environment
        state = np.reshape(state, [1, observation_space])
        step = 0

        while True:
            step += 1

            # Render environment (visualize)

            # Select action via epsilon-greedy policy
            action = dqn_solver.act(state) # ε-greedy
            next_state, reward, terminated, truncated, info = env.step(action) # update situation
            done = terminated or truncated

            # Modify reward for terminal step
            reward = reward if not done else -reward # punish if not done

            # Store experience and update agent
            next_state = np.reshape(next_state, [1, observation_space])
            dqn_solver.remember(state, action, reward, next_state, done) # put the experience to buffer
            state = next_state

            # If episode ends
            if done:
                print(f"Run: {run}, Exploration: {dqn_solver.exploration_rate:.4f}, Score: {step}")
                score_logger.add_score(step, run)
                break

            # Learn from past experience
            dqn_solver.experience_replay()

    env.close()
    return dqn_solver

def evaluate(dqn_solver):
    env = gym.make(ENV_NAME, render_mode="human")
    state, _ = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    done = False
    while not done:
        action = np.argmax(dqn_solver.model.predict(state))
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = np.reshape(next_state, [1, env.observation_space.shape[0]])
    env.close()

if __name__ == "__main__":
    Trained_Agent = train()
    evaluate(Trained_Agent)