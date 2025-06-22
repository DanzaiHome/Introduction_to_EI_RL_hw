import os
import sys
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from CustomNetwork.CustomNetworkPro import CustomCombinedExtractorPro
from envs.flexible_env_pro import FlexibleProAgentEnv
from data_define import data_used as d
from test import max_score_path

def test_model(model_path: str, 
               test_episodes: int = 5, 
               render: bool = False,):
    success_count = 0
    
    env = FlexibleProAgentEnv()
    env = DummyVecEnv([lambda: env])
    H,W = d.flexible_grid_size
    
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractorPro,
        features_extractor_kwargs=dict(grid_size=[2*H-1,2*W-1]),
    )
    
    model = PPO.load(model_path, env=env, policy_kwargs=policy_kwargs, verbose=1)
    print(f"Model '{model_path}' loaded successfully.")


    for ep in tqdm(range(test_episodes)):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]

            if render:
                env.render()
                time.sleep(0.2)

        standard_reward = max_score_path(env.envs[0].score_map)
        if(standard_reward >= total_reward):
            success_count += 1
            
    env.close()
    
    return success_count / test_episodes


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    model_path = f"{project_root}/{d.save_dir_name}/flexible_pro_model.zip"
    print(f"Success Rate: {test_model(model_path, test_episodes=500, render=False)}")