from typing import List
import numpy as np
from collections import defaultdict


def consumersToFirmsDistribution(num_consumers: int,
                                 num_firms: int) -> List[int]:
    return np.random.randint(0, num_firms, num_consumers)

# arr = consumersToFirmsDistribution(100, 8)
# d = defaultdict(int)
# for v in arr:
#     d[v] += 1
# print(arr)
# print(d)

def availableGoods(available_quantity: int,
                   desired_quantity: np.ndarray) -> np.array:
    scaling_coeff = available_quantity / desired_quantity.sum()
    return desired_quantity * scaling_coeff