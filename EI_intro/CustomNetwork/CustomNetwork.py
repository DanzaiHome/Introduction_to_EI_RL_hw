import torch
import torch.nn as nn
import gym
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict

class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
      - "score_map": shape=(1, H, W) or [B, 1, H, W]
      - "position": shape = (1,) or [B,1]
    """

    def __init__(self, observation_space: gym.spaces.Dict, grid_size):
        super().__init__(observation_space, features_dim=512)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        dummy_input = torch.zeros(1, 1, *grid_size)  # shape=[1,1,H,W]
        with torch.no_grad():
            dummy_out = self.cnn(dummy_input)
        self.cnn_output_size = dummy_out.shape[1] 

        self.position_fc = nn.Sequential(
            nn.Linear(grid_size[0] * grid_size[1], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(self.cnn_output_size + 64, 512),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        map_data = observations["score_map"]
        position_data = observations["position"]

        if map_data.ndim == 2:
            map_data = map_data.unsqueeze(0).unsqueeze(0)
        elif map_data.ndim == 3:
            map_data = map_data.unsqueeze(1)
        elif map_data.ndim == 4:
            map_data = map_data
        else:
            raise ValueError(f"Unsupported map_data shape: {map_data.shape}")

        map_data = map_data.float()

        position_data = position_data.float()
                
        # DEBUG OPTIONS
        # print(f"[DEBUG] map_data.shape = {map_data.shape}")
        # print(f"[DEBUG] position_data.shape = {position_data.shape}")

        assert map_data.shape[0] == position_data.shape[0], f"Batch size mismatch between map_data ({map_data.shape}) and position_data ({position_data.shape})"

        map_features = self.cnn(map_data)  # shape: [B, self.cnn_output_size]
        position_features = self.position_fc(position_data)  # shape: [B, 64]
        
        if position_features.ndim == 3 and position_data.shape[1] == 1:
            position_features = position_features.view(position_features.shape[0], -1)
          
        # DEBUG OPTIONS  
        # print(f"[DEBUG] map_data.shape = {map_data.shape}")
        # print(f"[DEBUG] position_data.shaep = {position_data.shape}")

        combined_features = torch.cat([map_features, position_features], dim=1)  # shape: [B, (cnn_out + 64)]

        return self.combined_fc(combined_features)
