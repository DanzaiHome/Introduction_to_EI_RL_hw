import os
import sys
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from envs.flexible_env import FlexibleAgentEnv
from data_define import data_used as d
from CustomNetwork.CustomNetwork import CustomCombinedExtractor

def visualize_map(grid: np.ndarray, agent_pos: tuple):
    H, W = np.array(grid).shape
    visual = ""
    for i in range(H):
        for j in range(W):
            if [i, j] == agent_pos:
                visual += " A "
            else:
                visual += " . "
        visual += "\n"
    print(visual)
    

def visualize_score_map(grid: np.ndarray):
    H, W = grid.shape
    visual = "\nScore Map:\n"
    
    visual += "   " + " ".join([f"{j:>3}" for j in range(W)]) + "\n"
    visual += "   " + "----" * W + "\n"

    for i in range(H):
        row_str = f"{i:>2}|"
        for j in range(W):
            val = grid[i][j]
            row_str += f"{val:>4}"
        visual += row_str + "\n"

    print(visual)
    time.sleep(5)
    
    
def test_model(model_path: str, test_episodes: int = 5, render: bool = False):
    env = FlexibleAgentEnv()
    env = DummyVecEnv([lambda: env])

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(grid_size=d.flexible_grid_size),
    )
    
    model = PPO.load(model_path, env=env, custom_objects={"policy_kwargs": policy_kwargs})
    print(f"Model '{model_path}' loaded successfully.")


    for ep in range(test_episodes):
        obs = env.reset()
        visualize_score_map(env.envs[0].score_map)
        done = False
        total_reward = 0
        step = 0

        visualize_map(env.envs[0].score_map, env.envs[0].agent_pos)
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if not done:
                visualize_map(env.envs[0].score_map, env.envs[0].agent_pos)
            else:
                temp_map = env.envs[0].score_map
                H, W = np.array(temp_map).shape
                visualize_map(temp_map, [H-1,W-1])
            total_reward += reward[0]
            step += 1

            if render:
                env.render()
                time.sleep(0.2)

        print(f"Episode {ep+1}: Total Reward = {total_reward}, Steps = {step}")
        print("Press Enter for the next episode...")
        input()

    env.close()


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    model_path = f"{project_root}/{d.save_dir_name}/ppo_model.zip"
    test_model(model_path, test_episodes=5, render=False)
