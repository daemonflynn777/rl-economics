from dataclasses import dataclass
from typing import List

# (n.__dict__) to get all class attributes (to calc number of input feaures for NN)

@dataclass
class ConsumerState:
    curr_tax: float
    item_prices: List[int]
    item_quantities: List[int]
    consumer_number: List[int]
    prev_wage: int
    prev_working_hours: int
    labour_disutility: float
    crra_uf_param: float
    budget: float

    def updateState(self,
                    curr_tax: float,
                    item_prices: List[int],
                    item_quantities: List[int],
                    consumer_number: List[int],
                    prev_wage: int,
                    prev_working_hours: int,
                    labour_disutility: float,
                    crra_uf_param: float,
                    budget: float) -> None:
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
    total_labour: int
    capital: float
    budget: float
    curr_tax: float
    firm_number: List[int]
    investments: float

    def updateState(self,
                    total_labour: int,
                    capital: float,
                    budget: float,
                    curr_tax: float,
                    firm_number: List[int],
                    investments: float) -> None:
        self.total_labour = total_labour
        self.capital = capital
        self.budget = budget
        self.curr_tax = curr_tax
        self.firm_number = firm_number
        self.investments = investments

        return None


@dataclass
class GovernmentState:
    number_of_consumers: int
    number_of_firms: int
    total_hours_worked: int
    item_prices: List[int]
    item_quantities: List[int]
    total_wage_payed: int

    

    def updateState(self,
                    number_of_consumers: int,
                    number_of_firms: int,
                    total_hours_worked: int,
                    item_prices: List[int],
                    item_quantities: List[int],
                    total_wage_payed: int) -> None:
        self.number_of_consumers = number_of_consumers
        self.number_of_firms = number_of_firms
        self.total_hours_worked = total_hours_worked
        self.item_prices = item_prices
        self.item_quantities = item_quantities
        self.total_wage_payed = total_wage_payed

        return None