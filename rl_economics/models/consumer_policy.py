import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np

from rl_economics.models.base_policy import BasePolicy


class ConsumerPolicy(BasePolicy):
    def __init__(self, num_input_features: int,
                 num_items: int, num_working_hours: int,
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

        self.item_head_0 = nn.Linear(mlp_layer_width, num_items)
        self.item_head_1 = nn.Linear(mlp_layer_width, num_items)
        self.item_head_2 = nn.Linear(mlp_layer_width, num_items)
        self.item_head_3 = nn.Linear(mlp_layer_width, num_items)
        self.item_head_4 = nn.Linear(mlp_layer_width, num_items)
        self.item_head_5 = nn.Linear(mlp_layer_width, num_items)
        self.item_head_6 = nn.Linear(mlp_layer_width, num_items)
        self.item_head_7 = nn.Linear(mlp_layer_width, num_items)

        self.working_hours_head = nn.Linear(mlp_layer_width, num_working_hours)

        self.head_indices = []
        for h in range(8):
            self.head_indices.append((h*num_items, h*num_items + num_items))
        self.head_indices.append((8*num_items, 8*num_items + num_working_hours))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)

        x_0 = self.item_head_0(x)
        x_1 = self.item_head_1(x)
        x_2 = self.item_head_2(x)
        x_3 = self.item_head_3(x)
        x_4 = self.item_head_4(x)
        x_5 = self.item_head_5(x)
        x_6 = self.item_head_6(x)
        x_7 = self.item_head_7(x)

        x_hours = self.working_hours_head(x)
        return torch.cat(
            (F.softmax(x_0, dim=1), F.softmax(x_1, dim=1), F.softmax(x_2, dim=1),
             F.softmax(x_3, dim=1), F.softmax(x_4, dim=1), F.softmax(x_5, dim=1),
             F.softmax(x_6, dim=1), F.softmax(x_7, dim=1), F.softmax(x_hours, dim=1)),
            dim=1
        )
    
    def act(self, state: np.ndarray):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state).cpu()
        m_0 = Categorical(probs[0, self.head_indices[0][0]:self.head_indices[0][1]])
        m_1 = Categorical(probs[0, self.head_indices[1][0]:self.head_indices[1][1]])
        m_2 = Categorical(probs[0, self.head_indices[2][0]:self.head_indices[2][1]])
        m_3 = Categorical(probs[0, self.head_indices[3][0]:self.head_indices[3][1]])
        m_4 = Categorical(probs[0, self.head_indices[4][0]:self.head_indices[4][1]])
        m_5 = Categorical(probs[0, self.head_indices[5][0]:self.head_indices[5][1]])
        m_6 = Categorical(probs[0, self.head_indices[6][0]:self.head_indices[6][1]])
        m_7 = Categorical(probs[0, self.head_indices[7][0]:self.head_indices[7][1]])
        m_hours = Categorical(probs[0, self.head_indices[8][0]:self.head_indices[8][1]])

        action_0 = m_0.sample()
        action_1 = m_1.sample()
        action_2 = m_2.sample()
        action_3 = m_3.sample()
        action_4 = m_4.sample()
        action_5 = m_5.sample()
        action_6 = m_6.sample()
        action_7 = m_7.sample()
        action_hours = m_hours.sample()
        composite_prob = (
            torch.exp(m_0.log_prob(action_0)) * torch.exp(m_1.log_prob(action_1)) * torch.exp(m_2.log_prob(action_2)) *
            torch.exp(m_3.log_prob(action_3)) * torch.exp(m_4.log_prob(action_4)) * torch.exp(m_5.log_prob(action_5)) *
            torch.exp(m_6.log_prob(action_6)) * torch.exp(m_7.log_prob(action_7)) * torch.exp(m_hours.log_prob(action_hours))
        )
        log_composite_prob = torch.log(composite_prob)
        return [
            action_0.item(), action_1.item(), action_2.item(), action_3.item(), action_4.item(),
            action_5.item(), action_6.item(), action_7.item(), action_hours.item(), log_composite_prob
        ]
        # return [
        #     action_0.item(), m_0.log_prob(action_0), action_1.item(), m_1.log_prob(action_1),
        #     action_2.item(), m_2.log_prob(action_1), action_3.item(), m_3.log_prob(action_3),
        #     action_4.item(), m_4.log_prob(action_4), action_5.item(), m_5.log_prob(action_5),
        #     action_6.item(), m_6.log_prob(action_6), action_7.item(), m_7.log_prob(action_7),
        #     action_hours.item(), m_hours.log_prob(action_hours),
        # ]
