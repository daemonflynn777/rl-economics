from typing import List
import numpy as np
from collections import defaultdict


def consumersToFirmsDistribution(num_consumers: int,
                                 num_firms: int) -> List[int]:
    return np.random.randint(0, num_firms, num_consumers)

arr = consumersToFirmsDistribution(100, 8)
d = defaultdict(int)
for v in arr:
    d[v] += 1
print(arr)
print(d)