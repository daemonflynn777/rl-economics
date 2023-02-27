from abc import ABC, abstractmethod
from torch import nn


class BasePolicy(ABC, nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError