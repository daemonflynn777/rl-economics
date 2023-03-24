from typing import Tuple, Dict, List
import numpy as np
from collections import defaultdict
import torch

from rl_economics import config as cfg


def consumersToFirmsDistribution(num_consumers: int,
                                 num_firms: int) -> Tuple[dict, dict]:
    dist =  np.random.randint(0, num_firms, num_consumers)
    consumer_to_firm: Dict[int, int] = {}
    firm_to_consumer = defaultdict(list)

    for i, el in enumerate(dist):
        consumer_to_firm[i] = el
        firm_to_consumer[el].append(i)

    return consumer_to_firm, firm_to_consumer


def availableGoods(available_quantity: int,
                   desired_quantity: np.ndarray) -> np.array:
    scaling_coeff = min(available_quantity / (desired_quantity.sum() + cfg.EPS), 1.0)

    return desired_quantity * scaling_coeff


def productionFunction(labour: float, capital: float, alpha: float) -> float:
    return 100*(capital**(1-alpha))*(labour**alpha)


def calcDiscountedReturns(rewards: List[float], gamma: float) -> List[float]:
    returns: List[float] = []
    for t in range(len(rewards)-1, -1, -1):
        disc_return_t = (returns[0] if len(returns)>0 else 0)
        returns.insert(0, gamma*disc_return_t + rewards[t]) # append values from the left
    returns = np.array(returns)
    #returns = (returns - returns.mean()) / (returns.std() + cfg.EPS)
    return returns


def calcLoss(discounted_returns: List[float], log_probs: List[torch.Tensor]) -> torch.Tensor:
    loss = discounted_returns[0]*log_probs[0]*(-1)
    for i in range(1, len(discounted_returns)):
        loss += discounted_returns[i]*log_probs[i]*(-1)
    return loss/len(discounted_returns)

def calcAgentsMeanLoss(rewards: List[float], log_probs: List[torch.Tensor],
                       gamma: float) -> Tuple[torch.Tensor, float]:
    assert len(rewards) == len(log_probs), "rewards and log_probs lists are not equal length"
    losses = []
    returns = []

    for i in range(len(rewards)):
        agent_returns = calcDiscountedReturns(rewards[i], gamma)
        agent_loss = calcLoss(agent_returns, log_probs[i])
        losses.append(agent_loss)
        returns.append(agent_returns[-1])

    mean_loss = losses[0]
    for i in range(1, len(losses)):
        mean_loss += losses[i]
    return mean_loss/len(losses), np.mean(returns)
