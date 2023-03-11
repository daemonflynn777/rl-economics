from dataclasses import dataclass
from typing import List
import numpy as np

# (n.__dict__) to get all class attributes (to calc number of input feaures for NN)
# print(ConsumerState.__dict__["__match_args__"])

@dataclass
class ConsumerState:
    curr_tax: List[float]
    item_prices: List[List[int]]
    item_quantities: List[List[int]]
    wage: List[int]
    working_hours: List[int]
    labour_disutility: List[float]
    crra_uf_param: List[float]
    budget: List[float]

    @classmethod
    def initialState(cls,
                     num_firms: int,
                     num_consumers: int,
                     curr_tax: float = 0.0,
                     item_prices: int = 0,
                     item_quantities: int = 0,
                     wage: int = 0,
                     working_hours: int = 0,
                     labour_disutility: float = 0.01,
                     crra_uf_param: float = 0.1,
                     budget: float = 0.0):
        return cls(
            [curr_tax]*num_consumers,
            [[item_prices]*num_firms]*num_consumers,
            [[item_quantities]*num_firms]*num_consumers,
            [wage]*num_consumers,
            [working_hours]*num_consumers,
            [labour_disutility]*num_consumers,
            [crra_uf_param]*num_consumers,
            [budget]*num_consumers
        )

    def updateState(self,
                    curr_tax: List[float],
                    item_prices: List[List[int]],
                    item_quantities: List[List[int]],
                    wage: List[int],
                    working_hours: List[int],
                    labour_disutility: List[float],
                    crra_uf_param: List[float],
                    budget: List[float]) -> None:
        self.curr_tax = curr_tax
        self.item_prices = item_prices
        self.item_quantities = item_quantities
        self.wage = wage
        self.working_hours = working_hours
        self.curr_tax = labour_disutility
        self.curr_tax = crra_uf_param
        self.curr_tax = budget

        return None
    
    def getConsumerState(self, consumer_id: int) -> np.ndarray:
        state = []
        consumer_number = [0]*len(self.curr_tax)
        consumer_number[consumer_id] = 1

        state.append(self.curr_tax[consumer_id])
        state.append(self.wage[consumer_id])
        state.append(self.working_hours[consumer_id])
        state.append(self.labour_disutility[consumer_id])
        state.append(self.crra_uf_param[consumer_id])
        state.append(self.budget[consumer_id])
        state.extend(self.item_prices[consumer_id])
        state.extend(self.item_quantities[consumer_id])
        state.extend(consumer_number)

        return np.array(state)
        


@dataclass
class FirmState:
    total_labour: List[int]
    capital: List[float]
    budget: List[float]
    curr_tax: List[float]
    firm_number: List[List[int]]
    investments: List[float]

    def updateState(self,
                    total_labour: List[int],
                    capital: List[float],
                    budget: List[float],
                    curr_tax: List[float],
                    firm_number: List[List[int]],
                    investments: List[float]) -> None:
        self.total_labour = total_labour
        self.capital = capital
        self.budget = budget
        self.curr_tax = curr_tax
        self.firm_number = firm_number
        self.investments = investments

        return None


@dataclass
class GovernmentState:
    number_of_consumers: List[int]
    number_of_firms: List[int]
    total_hours_worked: List[int]
    item_prices: List[List[int]]
    item_quantities: List[List[int]]
    total_wage_payed: List[int]

    def updateState(self,
                    number_of_consumers: List[int],
                    number_of_firms: List[int],
                    total_hours_worked: List[int],
                    item_prices: List[List[int]],
                    item_quantities: List[List[int]],
                    total_wage_payed: List[int]) -> None:
        self.number_of_consumers = number_of_consumers
        self.number_of_firms = number_of_firms
        self.total_hours_worked = total_hours_worked
        self.item_prices = item_prices
        self.item_quantities = item_quantities
        self.total_wage_payed = total_wage_payed

        return None
    
print(ConsumerState.__dict__["__match_args__"])