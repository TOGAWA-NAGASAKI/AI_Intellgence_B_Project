import sys
import os
sys.path.append('..')

from eval_config import EvalConfig
from evaluator import ModelEvaluator
from visualize import ResultVisualizer

def main():
    print("Starting model evaluation...")
    evaluator = ModelEvaluator(EvalConfig)
    results = evaluator.run_all_evaluations()
    
    evaluator.save_results("results/evaluation_results.json")
    evaluator.print_summary()
    visualizer = ResultVisualizer("results")
    visualizer.plot_performance_comparison(results)
    visualizer.plot_robustness(results)
    
    print(f"\nEvaluation complete! Results saved to 'results/' directory")

if __name__ == "__main__":
    main()