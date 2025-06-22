import torch
import torch.nn as nn
import gym
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict

class CustomCombinedExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, grid_size):
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


        self.combined_fc = nn.Sequential(
            nn.Linear(self.cnn_output_size, 512),
            nn.ReLU(),
        )

    def forward(self, observations) -> torch.Tensor:
        map_data = observations

        if map_data.ndim == 2:
            map_data = map_data.unsqueeze(0).unsqueeze(0)
        elif map_data.ndim == 3:
            map_data = map_data.unsqueeze(1)
        elif map_data.ndim == 4:
            map_data = map_data
        else:
            raise ValueError(f"Unsupported map_data shape: {map_data.shape}")

        map_data = map_data.float()

                
        # DEBUG OPTIONS
        # print(f"[DEBUG] map_data.shape = {map_data.shape}")
        # print(f"[DEBUG] position_data.shape = {position_data.shape}")


        map_features = self.cnn(map_data)  # shape: [B, self.cnn_output_size]

        # combined_features = torch.cat([map_features, position_features], dim=1)  # shape: [B, (cnn_out + 64)]

        return self.combined_fc(map_features)

class CustomCombinedExtractorPro(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, grid_size):
        super().__init__(observation_space, features_dim=512)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        dummy_input = torch.zeros(1, 1, *grid_size)
        with torch.no_grad():
            dummy_out = self.cnn(dummy_input)
        self.cnn_output_size = dummy_out.shape[1]
        # DEBUG OPTIONS
        # print(f"[DEBUG] CNN output shape: {dummy_out.shape}")

        self.combined_fc = nn.Sequential(
            nn.Linear(self.cnn_output_size, 512),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        map_data = observations.float()  # [B, 1, H, W]
        map_features = self.cnn(map_data)
        return self.combined_fc(map_features)
