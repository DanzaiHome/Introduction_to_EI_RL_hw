import os
import sys
import gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from CustomNetwork.CustomNetworkPro import CustomCombinedExtractorPro
from envs.flexible_env_pro import FlexibleProAgentEnv
from data_define import data_used as d

def main():
    env = FlexibleProAgentEnv()

    env = DummyVecEnv([lambda: env])
    
    H,W = d.flexible_grid_size
    
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractorPro,
        features_extractor_kwargs=dict(grid_size=[2*H-1,2*W-1]),
    )
    
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=d.LR,
        n_steps=2048,
        batch_size=d.BATCH_SIZE,
        n_epochs=10,
        gamma=d.GAMMA,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
    )
    
    '''
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=d.LR,
        n_steps=2048,
        batch_size=d.BATCH_SIZE,
        n_epochs=10,
        gamma=d.GAMMA,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
    )
    '''
    
    total_timesteps = d.FLEXIBLE_TIMESTEPS
    model.learn(total_timesteps=total_timesteps)

    project_root = os.path.dirname(os.path.dirname(__file__))
    save_dir = os.path.join(project_root, d.save_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "flexible_pro_model.zip")

    model.save(save_path)
    print(f"Training finished. Model saved as '{d.save_dir_name}/flexible_pro_agent.zip'.")

if __name__ == "__main__":
    main()
