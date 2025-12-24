import numpy as np
import gymnasium as gym
import os
import time
import matplotlib.pyplot as plt
import torch

# 导入 PPO (作为专家，用于生成数据)
from agents.simple_ppo_torch import SimplePPO
# 导入 BC (作为学生)
from agents.behavioral_cloning import BCAgent, BCConfig

ENV_NAME = "CartPole-v1"

EXPERT_MODEL_PATH = r"Final_Project\models\best_ppo_ppotorch_score_500.pth"


DATA_NUM_SAMPLES = 5000  
BC_EPOCHS = 50          
EVAL_EPISODES = 5       
GOAL_SCORE = 475        

# 图片保存路径
SCORE_PLOT_PATH = "scores/bc_training_progress_graph.png" 
LOSS_PLOT_PATH = "scores/bc_training_loss_graph.png"     
MODEL_SAVE_PATH = "models/bc_trained_model.pth"



class TrainingVisualizer:
    def __init__(self, score_path, loss_path):
        self.epochs = []
        self.scores = []
        self.losses = []  
        self.score_path = score_path
        self.loss_path = loss_path
        self.goal = GOAL_SCORE 

    def update(self, epoch, score, loss):
        self.epochs.append(epoch)
        self.scores.append(score)
        self.losses.append(loss) 

    def save_plots(self):
       
        if not self.scores: return

       
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.scores, label='Score per Epoch', color='#1f77b4', alpha=0.9)
        
        # 计算滑动平均 (Last 10)
        moving_avgs = []
        target_window = 10  
        window_size = min(target_window, len(self.scores))
        for i in range(len(self.scores)):
            start_idx = max(0, i - window_size + 1)
            subset = self.scores[start_idx : i + 1]
            moving_avgs.append(np.mean(subset))
            
        plt.plot(self.epochs, moving_avgs, label=f'Average of last {target_window}', 
                 color='#ff7f0e', linestyle='--', linewidth=2)
        plt.axhline(y=self.goal, color='green', linestyle=':', label=f'Goal ({self.goal} Avg)', alpha=0.8)

        if len(self.epochs) > 1:
            z = np.polyfit(self.epochs, self.scores, 1)
            p = np.poly1d(z)
            plt.plot(self.epochs, p(self.epochs), "r-.", label='Trend', linewidth=1.5)

        plt.title(f"{ENV_NAME} - BC Training Progress (Scores)")
        plt.xlabel("Training Epochs")
        plt.ylabel("Evaluation Score")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 520) 
        
        if not os.path.exists(os.path.dirname(self.score_path)):
            os.makedirs(os.path.dirname(self.score_path))
        plt.savefig(self.score_path)
        plt.close()

    
        plt.figure(figsize=(10, 6))
        
        # 绘制 Loss 曲线
        plt.plot(self.epochs, self.losses, label='Training Loss', color='#1f77b4', linewidth=2)
        
        plt.title(f"BC Training Loss (PyTorch)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        # plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存 Loss 图
        if not os.path.exists(os.path.dirname(self.loss_path)):
            os.makedirs(os.path.dirname(self.loss_path))
        plt.savefig(self.loss_path)
        plt.close()


# --- 数据生成函数 ---
def generate_expert_data(expert_agent, env, num_samples):
    states = []
    actions = []
    
    obs, _ = env.reset(seed=int(time.time()))
    while len(states) < num_samples:
        state_t = torch.FloatTensor(obs).unsqueeze(0).to(expert_agent.actor.net[0].weight.device)
        with torch.no_grad():
            probs = expert_agent.actor(state_t).cpu().numpy()[0]
        action = np.argmax(probs)
        
        states.append(obs)
        actions.append(action)
        
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
            
    return np.array(states), np.array(actions)

# --- 评估函数 ---
def evaluate_bc_agent(student_agent, env, episodes=EVAL_EPISODES):
    scores = []
    for _ in range(episodes):
        state, _ = env.reset()
        steps = 0
        done = False
        while not done:
            action = student_agent.act(state, evaluation_mode=True)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        scores.append(steps)
    return np.mean(scores)


def train_bc_with_visualization():
    # 检查专家模型
    if not os.path.exists(EXPERT_MODEL_PATH):
        print(f"错误: 找不到专家模型文件: {EXPERT_MODEL_PATH}")
        return

    # 初始化环境和专家
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    
    expert_agent = SimplePPO(obs_dim, act_dim)
    try:
        expert_agent.load_model(EXPERT_MODEL_PATH)
        print(f"PPO专家模型加载成功：{EXPERT_MODEL_PATH}")
    except Exception as e:
        print(f"加载PPO专家模型失败: {e}")
        return

    # 生成高质量的专家数据集
    print(f"\n正在生成高质量专家数据集 (N={DATA_NUM_SAMPLES})...")
    states, actions = generate_expert_data(expert_agent, env, DATA_NUM_SAMPLES)
    
    # 初始化 BC 学生 Agent
    print(f"\n初始化 BC 学生 Agent...")
    bc_cfg = BCConfig(epochs=BC_EPOCHS, batch_size=64, lr=0.001)
    student_agent = BCAgent(obs_dim, act_dim, cfg=bc_cfg)
    
    # 初始化可视化器 (传入两个路径)
    visualizer = TrainingVisualizer(score_path=SCORE_PLOT_PATH, loss_path=LOSS_PLOT_PATH)

    #  BC 训练循环
    print(f"\n--- 开始 BC 训练 (共 {BC_EPOCHS} Epochs) ---")
    
    states_tensor = torch.FloatTensor(states).to(student_agent.device)
    actions_tensor = torch.LongTensor(actions).to(student_agent.device)
    dataset_size = len(states)
    indices = np.arange(dataset_size)
    
    student_agent.model.train() 

    for epoch in range(1, BC_EPOCHS + 1):
        # --- 训练一个 Epoch ---
        np.random.shuffle(indices)
        epoch_loss = 0
        batches = 0
        
        for start_idx in range(0, dataset_size, bc_cfg.batch_size):
            end_idx = min(start_idx + bc_cfg.batch_size, dataset_size)
            batch_idx = indices[start_idx:end_idx]
            
            batch_s = states_tensor[batch_idx]
            batch_a = actions_tensor[batch_idx]
            
            student_agent.optimizer.zero_grad()
            logits = student_agent.model(batch_s)
            loss = student_agent.criterion(logits, batch_a)
            loss.backward()
            student_agent.optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
        
        avg_epoch_loss = epoch_loss / batches

        # --- 评估 ---
        student_agent.model.eval() # 切换到评估模式
        current_eval_score = evaluate_bc_agent(student_agent, env, episodes=EVAL_EPISODES)
        student_agent.model.train() # 切换回训练模式
        
        print(f"Epoch {epoch}/{BC_EPOCHS} | Loss: {avg_epoch_loss:.4f} | Eval Score: {current_eval_score:.1f}")
        
        
        visualizer.update(epoch, current_eval_score, avg_epoch_loss)
        visualizer.save_plots() 

    
    student_agent.save(MODEL_SAVE_PATH)
    env.close()
    print(f"\nBC 训练完成！模型已保存至: {MODEL_SAVE_PATH}")
    print(f"进度图已保存至: {SCORE_PLOT_PATH}")
    print(f"Loss图已保存至: {LOSS_PLOT_PATH}")

if __name__ == "__main__":
    os.makedirs("scores", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    train_bc_with_visualization()