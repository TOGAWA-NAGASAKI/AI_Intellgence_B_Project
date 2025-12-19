import gymnasium
import numpy as np
import torch
from tqdm import tqdm
import json
import metrics
import os

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.results = {}
    
    def load_model(self, model_name, model_path):
        if "dqn" in model_name.lower():
            from Final_Project.agents import DQNAgent
            model = DQNAgent()
        elif "ppo" in model_name.lower():
            from Final_Project.agents import PPOAgent
            model = PPOAgent()
        elif "bc" in model_name.lower():
            from Final_Project.agents import BCAgent
            model = BCAgent()
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        model.load(model_path)
        return model
    
    def evaluate_normal(self, model, env_name="CartPole-v1"):
        env = gymnasium.make(env_name)
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(self.config.NUM_TEST_EPISODES):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            total_reward = 0
            steps = 0
            
            for step in range(self.config.MAX_STEPS):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action = model.act(state_tensor)
                
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        env.close()
        return metrics.MetricsCalculator.performance_metrics(episode_rewards, episode_lengths)
    
    def evaluate_noisy(self, model, noise_level=0.1):
        env = gymnasium.make(self.config.ENV_NAME)
        episode_rewards = []
        
        for _ in range(self.config.NUM_TEST_EPISODES):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            total_reward = 0
            
            for step in range(self.config.MAX_STEPS):
                noisy_state = state + np.random.normal(0, noise_level, state.shape)
                state_tensor = torch.FloatTensor(noisy_state).unsqueeze(0)
                
                with torch.no_grad():
                    action = model.act(state_tensor)
                
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
        
        env.close()
        return {"mean_reward": np.mean(episode_rewards)}
    
    def run_all_evaluations(self):
        all_results = {}
        
        for model_name, model_path in tqdm(self.config.MODEL_PATHS.items(), desc="Evaluating models"):
            print(f"\nEvaluating {model_name}...")
            
            try:
                model = self.load_model(model_name, model_path)
                results = {}
                results["normal"] = self.evaluate_normal(model)
                env = gymnasium.make(self.config.ENV_NAME)
                results["speed"] = metrics.MetricsCalculator.speed_metrics(model, env)
                env.close()
                noise_results = {}
                for noise_level in self.config.NOISE_LEVELS:
                    noise_results[f"noise_{noise_level}"] = self.evaluate_noisy(model, noise_level)
                results["noise_robustness"] = noise_results

                results["robustness_score"] = metrics.MetricsCalculator.robustness_metrics(
                    noise_results, {}
                )
                
                all_results[model_name] = results
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                continue
        
        self.results = all_results
        return all_results
    
    def save_results(self, filename="model_evaluation/results/evaluation_results.json"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    
    def print_summary(self):
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        
        for model_name, results in self.results.items():
            print(f"\n{model_name}:")
            if "normal" in results:
                perf = results["normal"]
                print(f"  Performance: {perf['mean_reward']:.1f} ± {perf['std_reward']:.1f}")
                print(f"  Success Rate: {perf['success_rate']*100:.1f}%")
            
            if "speed" in results:
                speed = results["speed"]
                print(f"  Speed: {speed['inference_speed']:.0f} steps/sec")
            
            if "robustness_score" in results:
                rob = results["robustness_score"]
                if "noise_degradation" in rob:
                    print(f"  Noise Robustness: {rob['noise_degradation']:.1f}")