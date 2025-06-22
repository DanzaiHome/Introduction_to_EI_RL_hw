import torch
import torch.nn as nn
import gym
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict

class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    假设 observation_space 里包含:
      - "map": shape=(1, H, W) 或 [B, 1, H, W]  (单通道地图)
      - "position": shape=(2,) 或 [B, 2]        (x,y位置)
    """

    def __init__(self, observation_space: gym.spaces.Dict, grid_size):
        # 在构造函数里指定最终输出features_dim=512
        super().__init__(observation_space, features_dim=512)

        # 1) 卷积层: 输入通道=1
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 2) 利用 dummy input 自动推断cnn_output_size，避免手动计算
        #    假设 grid_size=(H, W)，手动/随便指定 batch_size=1
        dummy_input = torch.zeros(1, 1, *grid_size)  # shape=[1,1,H,W]
        with torch.no_grad():
            dummy_out = self.cnn(dummy_input)
        self.cnn_output_size = dummy_out.shape[1]  # 通道 x 高 x 宽

        # 3) 位置特征处理: (2, ) -> 64
        self.arm_state_fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # 4) 将cnn输出 + 位置特征拼接后映射到512
        self.combined_fc = nn.Sequential(
            nn.Linear(self.cnn_output_size + 64, 512),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        map_data = observations["map"]
        position_data = observations["position"]

        # convert data [H,W] or [1,H,W] to [1,1,H,W]
        if map_data.ndim == 2:
            map_data = map_data.unsqueeze(0).unsqueeze(0)
        elif map_data.ndim == 3:
            map_data = map_data.unsqueeze(1)
        elif map_data.ndim == 4:
            map_data = map_data
        else:
            raise ValueError(f"Unsupported map_data shape: {map_data.shape}")

        map_data = map_data.float()

        # convet [2] to [1,2] if necessary
        if position_data.ndim == 1:
            position_data = position_data.unsqueeze(0)
        position_data = position_data.float()

        # DEBUG OPTIONS
        # print(f"[DEBUG] map_data.shape = {map_data.shape}")
        # print(f"[DEBUG] position_data.shape = {position_data.shape}")

        assert map_data.shape[0] == position_data.shape[0], f"Batch size mismatch between map_data ({map_data.shape}) and position_data ({position_data.shape})"

        map_features = self.cnn(map_data)  # shape: [B, self.cnn_output_size]
        position_features = self.arm_state_fc(position_data)  # shape: [B, 64]

        combined_features = torch.cat([map_features, position_features], dim=1)  # shape: [B, (cnn_out + 64)]

        return self.combined_fc(combined_features)
