import numpy as np
from collections import defaultdict
import time
import psutil
import torch

class MetricsCalculator:
    @staticmethod
    def performance_metrics(episode_rewards, episode_lengths):
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "success_rate": np.mean([1 if r >= 475 else 0 for r in episode_rewards])
        }
    
    @staticmethod
    def speed_metrics(model, env, num_steps=1000):
        state = env.reset()
        start_time = time.time()
        for _ in range(num_steps):
            if isinstance(state, tuple):
                state = state[0]
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = model(state_tensor).argmax().item() if hasattr(model, 'argmax') else model(state_tensor)
            state, _, done, _ = env.step(action)
            if done:
                state = env.reset()
        inference_time = time.time() - start_time
        steps_per_second = num_steps / inference_time
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        return {
            "inference_speed": steps_per_second,
            "memory_mb": memory_usage
        }
    
    @staticmethod
    def robustness_metrics(noise_results, perturb_results):
        robustness = {}
        
        if noise_results:
            noise_scores = []
            for level, metrics in noise_results.items():
                noise_scores.append(metrics["mean_reward"])
            robustness["noise_degradation"] = np.mean(noise_scores)
        
        if perturb_results:
            perturb_scores = []
            for param, values in perturb_results.items():
                for val, metrics in values.items():
                    perturb_scores.append(metrics["mean_reward"])
            robustness["perturbation_degradation"] = np.mean(perturb_scores)
        
        return robustness