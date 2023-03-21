import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np

from rl_economics.models.base_policy import BasePolicy


class GovernmentPolicy(BasePolicy):
    def __init__(self, num_input_features: int, num_taxes: int,
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

        self.tax_head = nn.Linear(mlp_layer_width, num_taxes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)

        x_tax = self.tax_head(x)
        return F.softmax(x_tax, dim=1)
    
    def act(self, state: np.ndarray):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state).cpu()
        m_tax = Categorical(probs)

        action_tax = m_tax.sample()
        return action_tax.item(), m_tax.log_prob(action_tax)
