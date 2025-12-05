#测试完成后checkpoint里保存了结果，之后跑这个程序来验证分数
import numpy as np
import gymnasium as gym
from agents.simple_ppo import SimplePPO

env = gym.make("CartPole-v1")
agent = SimplePPO(4, 2)
agent.load_model("checkpoints/checkpoint_episode_250")#在这里改文件名选择文件，到数字为止就行
scores = []
for i in range(10):
    state, _ = env.reset()
    score = 0
    
    while True:
        action, _ = agent.act(state)
        state, _, terminated, truncated, _ = env.step(action)
        score += 1
        
        if terminated or truncated or score >= 500:
            break
    
    scores.append(score)
    print(f"Done {i + 1}/100 Times")

env.close()
print(f"\nFinished")
print(f"Average score: {np.mean(scores):.2f}")