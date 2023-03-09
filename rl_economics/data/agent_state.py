from dataclasses import dataclass
from typing import List

# (n.__dict__) to get all class attributes (to calc number of input feaures for NN)
# print(ConsumerState.__dict__["__match_args__"])

@dataclass
class ConsumerState:
    curr_tax: List[float]
    item_prices: List[List[int]]
    item_quantities: List[List[int]]
    consumer_number: List[List[int]]
    prev_wage: List[int]
    prev_working_hours: List[int]
    labour_disutility: List[float]
    crra_uf_param: List[float]
    budget: List[float]

    def updateState(self,
                    curr_tax: List[float],
                    item_prices: List[List[int]],
                    item_quantities: List[List[int]],
                    consumer_number: List[List[int]],
                    prev_wage: List[int],
                    prev_working_hours: List[int],
                    labour_disutility: List[float],
                    crra_uf_param: List[float],
                    budget: List[float]) -> None:
        self.curr_tax = curr_tax
        self.item_prices = item_prices
        self.item_quantities = item_quantities
        self.consumer_number = consumer_number
        self.prev_wage = prev_wage
        self.prev_working_hours = prev_working_hours
        self.curr_tax = labour_disutility
        self.curr_tax = crra_uf_param
        self.curr_tax = budget

        return None


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