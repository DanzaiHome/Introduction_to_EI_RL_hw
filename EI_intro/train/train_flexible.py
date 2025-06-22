import os
import sys
import copy
import gym
import torch
import torch.nn as nn
import numpy as np
from typing import Dict

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from envs.flexible_env import FlexibleAgentEnv
from data_define import data_used as d
from CustomNetwork.CustomNetwork import CustomCombinedExtractor

def main():

    env = FlexibleAgentEnv()
    
    env = DummyVecEnv([lambda: env])
    
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(grid_size=d.flexible_grid_size),
    )
    
    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    
    total_timesteps = d.FLEXIBLE_EPISODES
    model.learn(total_timesteps=total_timesteps)
    
    project_root = os.path.dirname(os.path.dirname(__file__))
    save_dir = os.path.join(project_root, d.save_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ppo_model.zip")

    model.save(save_path)
    print(f"Training finished. Model saved as '{d.save_dir_name}/ppo_flexible_agent.zip'.")

if __name__ == "__main__":
    main()
