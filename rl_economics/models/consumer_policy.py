import torch
from torch import nn
from typing import Tuple, List

from rl_economics.models.base_policy import BasePolicy


class ConsumerPolicy(BasePolicy):
    def __init__(self, num_input_features: int,
                 num_items: int, num_working_hours: int,
                 mlp_layer_width: int = 128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(num_input_features, mlp_layer_width),
            nn.ReLU(),
            nn.Linear(mlp_layer_width, mlp_layer_width),
            nn.ReLU(),
            nn.Linear(mlp_layer_width, mlp_layer_width),
        )

        self.item_head_1 = nn.Linear(mlp_layer_width, num_items)
        self.item_head_2 = nn.Linear(mlp_layer_width, num_items)
        self.item_head_3 = nn.Linear(mlp_layer_width, num_items)
        self.item_head_4 = nn.Linear(mlp_layer_width, num_items)
        self.item_head_5 = nn.Linear(mlp_layer_width, num_items)
        self.item_head_6 = nn.Linear(mlp_layer_width, num_items)
        self.item_head_7 = nn.Linear(mlp_layer_width, num_items)
        self.item_head_8 = nn.Linear(mlp_layer_width, num_items)

        self.working_hours_head = nn.Linear(mlp_layer_width, num_working_hours)
    
    def forward(self, x: torch.Tensor) -> Tuple[List[float]]:
        x = nn.ReLU(self.mlp(x))
        return -1
