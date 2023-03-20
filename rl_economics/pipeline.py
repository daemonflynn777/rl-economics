import os
import fire
from torch.optim import Adagrad
from typing import List
import numpy as np

from rl_economics.utils.misc import loadYamlConfig, initSeeds
from rl_economics.functions.general import (
    consumersToFirmsDistribution,
    availableGoods
)
from rl_economics.functions.utility import (
    consumerUtility
)
from rl_economics.models import (
    ConsumerPolicy,
    FirmPolicy,
    GovernmentPolicy
)
from rl_economics.data.agent_state import (
    ConsumerState,
    FirmState,
    GovernmentState
)
from rl_economics import config as cfg


class Pipeline:
    def __init__(self, config_file_path: str):
        full_config_file_path = os.path.join(os.getcwd(), config_file_path)
        
        self.simulation_config: dict = loadYamlConfig(full_config_file_path)

        self.tech_params = self.simulation_config["tech"]
        self.environment_params = self.simulation_config["environment"]
        self.consumer_params = self.simulation_config["consumer"]
        self.firm_params = self.simulation_config["firm"]
        self.government_params = self.simulation_config["government"]

        print(self.simulation_config)

    def initMappingDicts(self) -> None:
        self.possible_working_hours = {
            k: v for k, v in enumerate(list(np.arange(
                self.consumer_params["model"]["working_hours"]["min"],
                self.consumer_params["model"]["working_hours"]["max"] + 1,
                self.consumer_params["model"]["working_hours"]["step"]
            )))
        }
        self.possible_salaries = {
            k: v for k, v in enumerate(list(np.arange(
                self.firm_params["model"]["wages"]["min"],
                self.firm_params["model"]["wages"]["max"] + 1,
                self.firm_params["model"]["wages"]["step"]
            )))
        }
        self.possible_prices = {
            k: v for k, v in enumerate(list(np.arange(
                self.firm_params["model"]["prices"]["min"],
                self.firm_params["model"]["prices"]["max"] + 1,
                self.firm_params["model"]["prices"]["step"]
            )))
        }
        self.possible_taxes = {
            k: v for k, v in enumerate(list(np.arange(
                self.government_params["model"]["taxes"]["min"],
                self.government_params["model"]["taxes"]["max"] + 0.1,
                self.government_params["model"]["taxes"]["step"]
            )))
        }
        self.consumer_to_firm, self.firm_to_consumer = consumersToFirmsDistribution(
            num_consumers=self.environment_params["num_consumers"],
            num_firms=self.environment_params["num_firms"]
        )
        return None
    
    def initPolicies(self) -> None:
        print("Init consumer policy")
        num_working_hours = ((self.consumer_params["model"]["working_hours"]["max"] -
                              self.consumer_params["model"]["working_hours"]["min"]) //
                             self.consumer_params["model"]["working_hours"]["step"] + 1)
        consumer_num_unput_features = (len(ConsumerState.__dict__["__match_args__"]) - 2 +
                                       2*self.environment_params["num_firms"] +
                                       self.environment_params["num_consumers"])
        self.consumer_policy = ConsumerPolicy(
            num_input_features=consumer_num_unput_features,
            num_items=self.consumer_params["model"].get("num_items", 10),
            num_working_hours=num_working_hours,
            mlp_layer_width=self.consumer_params["model"].get("mlp_layer_width", 128),
            device=self.tech_params["device"]
        )
        self.consumer_policy.to(self.tech_params["device"])
        
        print("Init firm policy")
        num_prices = ((self.firm_params["model"]["prices"]["max"] -
                       self.firm_params["model"]["prices"]["min"]) //
                      self.firm_params["model"]["prices"]["step"] + 1)
        num_wages = ((self.firm_params["model"]["wages"]["max"] -
                      self.firm_params["model"]["wages"]["min"]) //
                     self.firm_params["model"]["wages"]["step"] + 1)
        firm_num_unput_features = (len(FirmState.__dict__["__match_args__"]) +
                                       self.environment_params["num_firms"])
        self.firm_policy = FirmPolicy(
            num_input_features=firm_num_unput_features,
            num_prices=num_prices,
            num_wages=num_wages,
            mlp_layer_width=self.firm_params["model"].get("mlp_layer_width", 128),
            device=self.tech_params["device"]
        )
        self.firm_policy.to(self.tech_params["device"])

        print("Init government policy")
        num_taxes = int((self.government_params["model"]["taxes"]["max"] -
                         self.government_params["model"]["taxes"]["min"]) //
                        self.government_params["model"]["taxes"]["step"])
        self.government_policy = GovernmentPolicy(
            num_input_features=1,
            num_taxes=num_taxes,
            mlp_layer_width=self.government_params["model"].get("mlp_layer_width", 128)
        )
    
    def initOptimizers(self) -> None:
        self.consumer_optimizer = Adagrad(params=self.consumer_policy.parameters(),
                                          lr=self.consumer_params["optimizer"]["lr"])
        
        self.firm_optimizer = Adagrad(params=self.firm_policy.parameters(),
                                      lr=self.firm_params["optimizer"]["lr"])
        
        self.government_optimizer = Adagrad(params=self.government_policy.parameters(),
                                            lr=self.government_params["optimizer"]["lr"])

    def initLosses(self) -> None:
        pass

    def reinforce(self) -> None:
        pass

    def simulateConsumers(self, state: ConsumerState, item_prices: List[int],
                         item_quantities: List[int], consumer_to_firm: dict):
        choices: list = [] # shape is (num_consumer, num_firms + 1)
        log_probs: list = [] # shape is (num_consumer, num_firms + 1)
        rewards: list = []
        item_scaled_consumption: list = []

        # get actions for each consumer
        for i in range(self.environment_params["num_consumers"]):
            policy_input = state.getConsumerState(i)
            policy_output = self.consumer_policy.act(policy_input)
            choices.append(policy_output[::2])
            log_probs.append(policy_output[1::2])
        
        choices = np.array(choices) # shape is (num_consumer, num_firms + 1)
        # get item consumption for each consumer
        for i in range(self.environment_params["num_firms"]):
            item_demand = choices[:, i].reshape(-1)
            item_available = item_quantities[i]
            item_scaled_consumption.append(availableGoods(item_available, item_demand))
        item_scaled_consumption = np.array(item_scaled_consumption).T # shape is (num_consumer, num_firms)

        # calculate budget change for each consumer: + salary - taxes
        # TODO: maybe create new state as a copy of current state and update budget in new state
        payed_taxes: list = []
        budgets_before_purchase: list = []
        for i in range(self.environment_params["num_consumers"]):
            consumer_payed_taxes = choices[i][-1] * state.wage[i] * state.curr_tax[i]
            consumer_received_salary = choices[i][-1] * state.wage[i] * (1 - state.curr_tax[i])
            payed_taxes.append(consumer_payed_taxes)
            budgets_before_purchase.append(state.budget[i] + consumer_received_salary)
            
        payed_taxes = np.array(payed_taxes) # shape is (num_consumers,)
        budgets_before_purchase = np.array(budgets_before_purchase) # shape is (num_consumers,)
        working_hours_choices = choices[:, -1].reshape(-1) # shape is (num_consumers,)

        # calculate rewards and consumers' budgets after purchase
        budgets_after_purchase: list = []
        for i in range(self.environment_params["num_consumers"]):
            consumer_items_consumption = item_scaled_consumption[i, :].reshape(-1).tolist()
            working_hours = [0]*self.environment_params["num_firms"]
            working_hours[consumer_to_firm[i]] = choices[i][-1]
            reward, budget_after_purchase = consumerUtility(consumer_items_consumption, item_prices,
                                                            working_hours, self.environment_params["labour_disutility"],
                                                            self.environment_params["crra_uf_param"],
                                                            budgets_before_purchase[i])
            rewards.append(reward)
            budgets_after_purchase.append(budget_after_purchase)
        
        rewards = np.array(rewards) # shape is (num_consumers,)
        budgets_after_purchase = np.array(budgets_after_purchase) # shape is (num_consumers,)

        return (item_scaled_consumption, working_hours_choices, budgets_after_purchase,
                payed_taxes, rewards, log_probs)
    
    def simulateFirms(self, state: FirmState):
        choices: list = [] # shape is (num_firms, 2)
        log_probs: list = [] # shape is (num_firms, 2)

        # get actions for each firm
        for i in range(self.environment_params["num_firms"]):
            policy_input = state.getFirmState(i)
            policy_output = self.firm_policy.act(policy_input)
            choices.append(policy_output[::2])
            log_probs.append(policy_output[1::2])
        
        choices = np.array(choices)

        price_choices = choices[:, 0]
        wage_choices = choices[:, 1]

        return price_choices, wage_choices, log_probs
        
        # update item quantities
        # update prices
        # update wages

    def simulateGovernment(self):
        pass

    def run(self) -> None:
        print("Init seeds for all random things")
        initSeeds(self.tech_params.get("seed", 666))

        print("Initialize mapping dicts for possible salaries, prices etc.")
        self.initMappingDicts()

        print("Start initializing policies")
        self.initPolicies()

        print("Create inital states")

        print("Set initial states for consumers, firms and government")
        # code goes here

        print(f"Start training neural networks, number of epochs: {self.environment_params['epochs']}")
        for i in range(self.environment_params["epochs"]):
            # code goes here
            print("Init variables for new epoch")
            consumers_rewards: list = []
            consumers_log_probs: list = []
            firms_rewards = []
            firms_log_probs = []

            for t in range(self.environment_params["timesteps"]):
                if t == 0:
                    # initial simulation:
                    # create consumers state
                    # simulate consumers
                    # create firms state
                    # simulate firm policy
                    # create government policy
                    # simulate government policy
                    c_state = ConsumerState.initialState(
                        self.environment_params["num_firms"],
                        self.environment_params["num_consumers"]
                    )
                    (item_scaled_consumption, working_hours_choices,
                     budgets_after_purchase, consumers_payed_taxes,
                     consumers_rewards, consumers_log_probs) = self.simulateConsumers(
                        c_state,
                        [0]*self.environment_params["num_firms"],
                        [0]*self.environment_params["num_firms"],
                        self.consumer_to_firm
                    )

                    # maybe make it a separate method or move into simulateFirms
                    total_goods_sales = []
                    firms_payed_taxes = []
                    for sales, price in zip(np.sum(item_scaled_consumption, axis=0), [0]*self.environment_params["num_firms"]):
                        total_goods_sales.append(sales*price*(1.0-0.0)) # these are actually firms' rewards
                        firms_payed_taxes.append(sales*price*0.0) 
                    total_goods_sales = np.array(total_goods_sales)
                    total_labour_per_firm = [0] * self.environment_params["num_firms"] # add produced items
                    for i in range(self.environment_params["num_consumers"]):
                        firm_id = self.consumer_to_firm[i]
                        num_hours = self.possible_working_hours[working_hours_choices[i]]
                        total_labour_per_firm[firm_id] += num_hours
                    total_salaries_payed = []
                    for i in range(self.environment_params["num_firms"]):
                        total_salaries_payed.append(total_labour_per_firm[i]*0)
                    initial_budgets = []
                    initial_investments = []
                    for i in range(self.environment_params["num_firms"]):
                        initial_budgets.append(
                            (2200000 + total_goods_sales[i])*(1-self.environment_params["investments_percent"]) - total_salaries_payed[i]
                        )
                        initial_investments.append((2200000 + total_goods_sales[i])*self.environment_params["investments_percent"])
                    firms_capitals = [
                        cfg.FIRMS_INITIAL_STATES[i]["capital"] + (2200000 + total_goods_sales[i])*self.environment_params["investments_percent"]
                        for i in range(self.environment_params["num_firms"])
                    ]
                    firms_pf_alphas = [
                        cfg.FIRMS_INITIAL_STATES[i]["pf_alpha"] for i in range(self.environment_params["num_firms"])
                    ]
                    f_state = FirmState(
                        total_labour_per_firm,
                        firms_capitals,
                        firms_pf_alphas,
                        initial_budgets,
                        [0.0]*self.environment_params["num_firms"],
                        initial_investments
                    )
                    prices_choices, wages_choices, firms_log_probs = self.simulateFirms(f_state)
                break

            

            # c_state = ConsumerState(
            #     curr_tax=[0.0]*self.environment_params["num_consumers"],
            #     item_prices=[[1000]*self.environment_params["num_firms"]]*self.environment_params["num_consumers"],
            #     item_quantities=[[0]*self.environment_params["num_firms"]]*self.environment_params["num_consumers"],
            #     wage=[0]*self.environment_params["num_consumers"],
            #     working_hours=[0]*self.environment_params["num_consumers"],
            #     labour_disutility=[self.environment_params["labour_disutility"]]*self.environment_params["num_consumers"],
            #     crra_uf_param=[self.environment_params["crra_uf_param"]]*self.environment_params["num_consumers"],
            #     budget=[0.0]*self.environment_params["num_consumers"]
            # )
            # # print(c_state.getConsumerState(j).shape)
            # for j in range(self.environment_params["num_consumers"]):
            #     action = self.consumer_policy.act(c_state.getConsumerState(j))
            #     action_items = action[::2]
            #     action_probs = action[1::2]
            #     print(action_items)
            #     print(action_probs)
            #     break


if __name__ == "__main__":
    fire.Fire(Pipeline)
