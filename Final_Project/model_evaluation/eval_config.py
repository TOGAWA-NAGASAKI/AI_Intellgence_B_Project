import json
from pathlib import Path

class EvalConfig:
    ENV_NAME = "CartPole-v1"
    NUM_TEST_EPISODES = 20
    MAX_STEPS = 500
    MODEL_PATHS = {
        "DQN": "../agants/dqn_cartpole.pt",
        "DDQN": "../models/ddqn_cartpole.pt",
        "PPO": "../models/ppo_cartpole.pt",
        "PPO_torch": "../models/ppo_torch_cartpole.pt",
        "BC": "../models/bc_cartpole.pt"
    }
    
    EVAL_MODES = ["normal", "noisy", "perturbed"]
    NOISE_LEVELS = [0.01, 0.05, 0.1, 0.2]
    
    PERTURBATIONS = {
        "gravity": [9.8, 12.0, 15.0],
        "masscart": [1.0, 1.5, 2.0],
        "masspole": [0.1, 0.2, 0.3]
    }
    
    @classmethod
    def save(cls, path="config.json"):
        with open(path, 'w') as f:
            json.dump(cls.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path="config.json"):
        with open(path, 'r') as f:
            data = json.load(f)
            for key, value in data.items():
                setattr(cls, key, value)