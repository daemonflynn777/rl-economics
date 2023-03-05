import torch
from torch import nn
from typing import Tuple, List

from rl_economics.models.base_policy import BasePolicy


class FirmPolicy(BasePolicy):
    def __init__(self, num_input_features: int,
                 num_prices: int, num_wages: int,
                 mlp_layer_width: int = 128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(num_input_features, mlp_layer_width),
            nn.ReLU(),
            nn.Linear(mlp_layer_width, mlp_layer_width),
            nn.ReLU(),
            nn.Linear(mlp_layer_width, mlp_layer_width),
        )

        self.price_head = nn.Linear(mlp_layer_width, num_prices)

        self.wage_head = nn.Linear(mlp_layer_width, num_wages)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[float]]:
        x = nn.ReLU(self.mlp(x))
        return -1
