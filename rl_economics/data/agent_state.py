from dataclasses import dataclass
from typing import List
import numpy as np

# (n.__dict__) to get all class attributes (to calc number of input feaures for NN)
# print(ConsumerState.__dict__["__match_args__"])

@dataclass
class ConsumerState:
    curr_tax: float
    item_prices: List[int]
    item_quantities: List[int]
    item_overdemand: List[int]
    wage: List[int]
    labour_disutility: float
    crra_uf_param: float
    budget: List[float]

    @classmethod
    def initialState(cls,
                     num_firms: int,
                     num_consumers: int,
                     curr_tax: float = 0.0,
                     item_prices: int = 0,
                     item_quantities: int = 0,
                     wage: int = 0,
                     labour_disutility: float = 0.01,
                     crra_uf_param: float = 0.1,
                     budget: float = 0.0):
        return cls(
            curr_tax,
            [item_prices]*num_firms,
            [item_quantities]*num_firms,
            [wage]*num_consumers,
            labour_disutility,
            crra_uf_param,
            [budget]*num_consumers
        )

    def updateState(self,
                    curr_tax: float,
                    item_prices: List[int],
                    item_quantities: List[int],
                    wage: List[int],
                    labour_disutility: float,
                    crra_uf_param: float,
                    budget: List[float]) -> None:
        self.curr_tax = curr_tax
        self.item_prices = item_prices
        self.item_quantities = item_quantities
        self.wage = wage
        self.curr_tax = labour_disutility
        self.curr_tax = crra_uf_param
        self.curr_tax = budget

        return None
    
    def getConsumerState(self, consumer_id: int) -> np.ndarray:
        state = []
        consumer_number = [0]*len(self.wage)
        consumer_number[consumer_id] = 1

        state.append(self.curr_tax)
        state.append(self.wage[consumer_id]/1e1)
        state.append(self.labour_disutility)
        state.append(self.crra_uf_param)
        state.append(self.budget[consumer_id]/1e5)
        state.extend((np.array(self.item_prices)/2.5e3).tolist())
        state.extend((np.array(self.item_quantities)/5e7).tolist())
        state.extend(self.item_overdemand)
        state.extend(consumer_number)
        # state.append(self.curr_tax)
        # state.append(self.wage[consumer_id])
        # state.append(self.labour_disutility)
        # state.append(self.crra_uf_param)
        # state.append(self.budget[consumer_id])
        # state.extend(self.item_prices)
        # state.extend(self.item_quantities)
        # state.extend(self.item_overdemand)
        # state.extend(consumer_number)

        return np.array(state)
        

@dataclass
class FirmState:
    total_labour: List[int]
    capital: List[float]
    pf_alphas: List[float]
    budget: List[float]
    curr_tax: float
    investments: List[float]
    item_prices: List[int]
    item_quantities: List[int]
    item_overdemand: List[int]

    def updateState(self,
                    total_labour: List[int],
                    capital: List[float],
                    pf_alphas: List[float],
                    budget: List[float],
                    curr_tax: float,
                    investments: List[float]) -> None:
        self.total_labour = total_labour
        self.capital = capital
        self.pf_alphas = pf_alphas
        self.budget = budget
        self.curr_tax = curr_tax
        self.investments = investments

        return None
    
    def getFirmState(self, firm_id: int) -> np.ndarray:
        state = []
        firm_number = [0]*len(self.total_labour)
        firm_number[firm_id] = 1

        state.append(self.total_labour[firm_id]/7e3)
        state.append(self.capital[firm_id]/1e7)
        state.append(self.pf_alphas[firm_id])
        state.append(self.budget[firm_id]/2e7)
        state.append(self.curr_tax)
        state.append(self.investments[firm_id]/1e6)
        state.extend((np.array(self.item_prices)/2.5e3).tolist())
        state.extend((np.array(self.item_quantities)/5e7).tolist())
        state.extend(self.item_overdemand)
        state.extend(firm_number)
        # state.append(self.total_labour[firm_id])
        # state.append(self.capital[firm_id])
        # state.append(self.pf_alphas[firm_id])
        # state.append(self.budget[firm_id])
        # state.append(self.curr_tax)
        # state.append(self.investments[firm_id])
        # state.extend(self.item_prices)
        # state.extend(self.item_quantities)
        # state.extend(self.item_overdemand)
        # state.extend(firm_number)

        return np.array(state)


@dataclass
class GovernmentState:
    number_of_consumers: int
    number_of_firms: int
    total_hours_worked: int
    total_wage_payed: int
    total_tax_payed: float
    item_prices: List[int]
    item_quantities: List[int]

    def updateState(self,
                    number_of_consumers: int,
                    number_of_firms: int,
                    total_hours_worked: int,
                    total_wage_payed: int,
                    total_tax_payed: float,
                    item_prices: List[int],
                    item_quantities: List[int],) -> None:
        self.number_of_consumers = number_of_consumers
        self.number_of_firms = number_of_firms
        self.total_hours_worked = total_hours_worked
        self.total_wage_payed = total_wage_payed
        self.total_tax_payed = total_tax_payed
        self.item_prices = item_prices
        self.item_quantities = item_quantities

        return None

    def getGovernmentState(self) -> np.ndarray:
        state = []

        state.append(self.number_of_consumers/5e1)
        state.append(self.number_of_firms/8)
        state.append(self.total_hours_worked/3e4)
        state.append(self.total_wage_payed/6e5)
        state.append(self.total_tax_payed/1e5)
        state.extend((np.array(self.item_prices)/2.5e3).tolist())
        state.extend((np.array(self.item_quantities)/5e7).tolist())
        # state.append(self.number_of_consumers)
        # state.append(self.number_of_firms)
        # state.append(self.total_hours_worked)
        # state.append(self.total_wage_payed)
        # state.append(self.total_tax_payed)
        # state.extend(self.item_prices)
        # state.extend(self.item_quantities)

        return np.array(state)
    
print(ConsumerState.__dict__["__match_args__"])