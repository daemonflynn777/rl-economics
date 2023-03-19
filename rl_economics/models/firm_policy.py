import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np

from rl_economics.models.base_policy import BasePolicy


class FirmPolicy(BasePolicy):
    def __init__(self, num_input_features: int,
                 num_prices: int, num_wages: int,
                 mlp_layer_width: int = 128, device: str = "cpu"):
        super().__init__()

        self.device = device

        self.mlp = nn.Sequential(
            nn.Linear(num_input_features, mlp_layer_width),
            nn.ReLU(),
            nn.Linear(mlp_layer_width, mlp_layer_width),
            nn.ReLU(),
            nn.Linear(mlp_layer_width, mlp_layer_width),
            nn.ReLU()
        )

        self.price_head = nn.Linear(mlp_layer_width, num_prices)

        self.wage_head = nn.Linear(mlp_layer_width, num_wages)

        self.head_indices = [
            (0, num_prices),
            (num_prices, num_prices+num_wages)
        ]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)

        x_price = self.price_head(x)
        x_wage = self.price_head(x)
        return torch.cat((F.softmax(x_price, dim=1), F.softmax(x_wage, dim=1),), dim=1)
    
    def act(self, state: np.ndarray):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state).cpu()
        m_price = Categorical(probs[0, self.head_indices[0][0]:self.head_indices[0][1]])
        m_wage = Categorical(probs[0, self.head_indices[1][0]:self.head_indices[1][1]])

        action_price = m_price.sample()
        action_wage = m_wage.sample()
        return [
            action_price.item(), m_price.log_prob(action_price),
            action_wage.item(), m_wage.log_prob(action_wage),
        ]
