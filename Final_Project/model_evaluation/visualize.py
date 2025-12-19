import matplotlib.pyplot as plt
import numpy as np
import json
import os

class ResultVisualizer:
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def plot_performance_comparison(self, results):
        models = []
        mean_rewards = []
        success_rates = []
        
        for model_name, data in results.items():
            if "normal" in data:
                models.append(model_name)
                mean_rewards.append(data["normal"]["mean_reward"])
                success_rates.append(data["normal"]["success_rate"] * 100)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        bars1 = ax1.bar(models, mean_rewards)
        ax1.set_ylabel('Mean Reward')
        ax1.set_title('Mean Reward Comparison')
        ax1.set_ylim(0, 500)
        
        bars2 = ax2.bar(models, success_rates)
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rate Comparison')
        ax2.set_ylim(0, 100)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax = bar.axes
                ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                       f'{height:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/performance_comparison.png', dpi=150)
        plt.show()
    
    def plot_robustness(self, results):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        noise_levels = [0.01, 0.05, 0.1, 0.2]
        for model_name, data in results.items():
            if "noise_robustness" in data:
                noise_scores = []
                for level in noise_levels:
                    key = f"noise_{level}"
                    if key in data["noise_robustness"]:
                        noise_scores.append(data["noise_robustness"][key]["mean_reward"])
                
                axes[0].plot(noise_levels, noise_scores, marker='o', label=model_name)
        
        axes[0].set_xlabel('Noise Level')
        axes[0].set_ylabel('Mean Reward')
        axes[0].set_title('Noise Robustness')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        models = []
        speeds = []
        
        for model_name, data in results.items():
            if "speed" in data:
                models.append(model_name)
                speeds.append(data["speed"]["inference_speed"])
        
        bars = axes[1].bar(models, speeds)
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('Steps per Second')
        axes[1].set_title('Inference Speed')
        axes[1].tick_params(axis='x', rotation=45)
        
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 5,
                       f'{height:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/robustness_analysis.png', dpi=150)
        plt.show()