from typing import Tuple, Dict
import numpy as np
from collections import defaultdict


def consumersToFirmsDistribution(num_consumers: int,
                                 num_firms: int) -> Tuple[dict, dict]:
    dist =  np.random.randint(0, num_firms, num_consumers)
    consumer_to_firm: Dict[int, int] = {}
    firm_to_consumer = defaultdict(list)

    for i, el in enumerate(dist):
        consumer_to_firm[i] = el
        firm_to_consumer[el].append(i)

    return consumer_to_firm, firm_to_consumer

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
