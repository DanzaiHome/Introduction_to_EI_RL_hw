import torch
import torch.nn as nn
from typing import Dict
import os
import sys
import gym
from CustomNetwork.CustomNetwork import CustomCombinedExtractor

class FlexibleQNetwork(nn.Module):
    def __init__(self, observation_space: gym.spaces.Dict, n_actions: int, grid_size):
        super(FlexibleQNetwork, self).__init__()
        self.feature_extractor = CustomCombinedExtractor(observation_space, grid_size=grid_size)
        self.q_head = nn.Linear(512, n_actions)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:

        features = self.feature_extractor(obs)  # [B, 512]
        q_values = self.q_head(features)        # [B, n_actions]
        return q_values
    
class StableQNetwork(nn.Module):
    def __init__(self, n_states: int, n_actions: int, embed_dim=64):
        super(StableQNetwork, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=n_states, embedding_dim=embed_dim)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, n_actions)
        
    def forward(self, obs: torch.LongTensor) -> torch.Tensor:
        x = self.embedding(obs)         # (batch_size, embed_dim)
        x = torch.relu(self.fc1(x))     # (batch_size, 128)
        q_values = self.fc2(x)          # (batch_size, n_actions)
        return q_values
