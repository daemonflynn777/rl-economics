from typing import List
import numpy as np


def consumerUtility(items_consumption: List[float],
                    items_prices: List[int],
                    firms_working_hours: List[int],
                    labour_disutility: float,
                    crra_uf_param: float,
                    budget: float) -> float:
    assert_msg = "items_consumption and firms_working_hours should be equal length!"
    res_utility: float = 0.0

    assert len(items_consumption) == len(firms_working_hours), assert_msg

    if np.dot(items_consumption, items_prices) > budget:
        return -1.0
    
    for i, h in zip(items_consumption, firms_working_hours):
        res_utility += ((i + 1)**(1 - crra_uf_param) - 1) / (1 - crra_uf_param) - h*labour_disutility/2
    
    return res_utility


def firmUtility(price: int, quantity: int, salary: int,
                employees_total_working_hours: int,
                ivestments: float) -> float:
    res_utility: float = 0.0

    res_utility = price*quantity - salary*employees_total_working_hours - ivestments
    
    return res_utility


def governmentUtility(consumer_utilities: List[float],
                      firms_utilities: List[float]) -> float:
    res_utility: float = 0.0

    res_utility = sum(consumer_utilities) + sum(firms_utilities)
    
    return res_utility

