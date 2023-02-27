from torch import nn

from rl_economics.models.base_policy import BasePolicy


class ConsumerPolicy(BasePolicy):
    def __init__(self, num_input_features: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(num_input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
    
    def forward(self):
        return -1


consumer_policy = ConsumerPolicy(num_input_features=228)