import os
import fire
import torch
from torch.optim import Adam
from typing import List
import numpy as np

from rl_economics.utils.misc import loadYamlConfig, initSeeds, transposeList2d
from rl_economics.functions.general import (
    consumersToFirmsDistribution,
    availableGoods,
    productionFunction,
    calcDiscountedReturns,
    calcLoss,
    calcAgentsMeanLoss
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

        # Consumers' variables
        self.consumers_budgets = [0.0]*self.environment_params["num_consumers"]
        self.consumers_working_hours = [0]*self.environment_params["num_consumers"]
        self.consumers_wages = [0]*self.environment_params["num_consumers"]

        # Firms' variables
        self.items_quantities = [0.0]*self.environment_params["num_firms"]
        self.items_prices = [0]*self.environment_params["num_firms"]
        self.firms_total_labour = [0]*self.environment_params["num_firms"]
        self.firms_wages = [0]*self.environment_params["num_firms"]
        self.firms_budgets = [22000000]*self.environment_params["num_firms"]
        self.firms_capitals = [cfg.FIRMS_INITIAL_STATES[i]["capital"] for i in range(self.environment_params["num_firms"])]
        self.firms_pf_alphas = [cfg.FIRMS_INITIAL_STATES[i]["pf_alpha"] for i in range(self.environment_params["num_firms"])]

        # Government's variales
        self.tax_rate = 0.0
        self.distributed_tax = 0.0
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
        government_num_unput_features = (len(GovernmentState.__dict__["__match_args__"]) - 2 +
                                         2*self.environment_params["num_firms"])
        self.government_policy = GovernmentPolicy(
            num_input_features=government_num_unput_features,
            num_taxes=num_taxes,
            mlp_layer_width=self.government_params["model"].get("mlp_layer_width", 128),
            device=self.tech_params["device"]
        )
        self.government_policy.to(self.tech_params["device"])
    
    def initOptimizers(self) -> None:
        self.consumer_optimizer = Adam(params=self.consumer_policy.parameters(),
                                       lr=self.consumer_params["optimizer"]["lr"])
        
        self.firm_optimizer = Adam(params=self.firm_policy.parameters(),
                                   lr=self.firm_params["optimizer"]["lr"])
        
        self.government_optimizer = Adam(params=self.government_policy.parameters(),
                                         lr=self.government_params["optimizer"]["lr"])

    def initLosses(self) -> None:
        pass

    def simulateConsumers(self, consumer_to_firm: dict):
        choices: list = [] # shape is (num_consumers, num_firms + 1)
        log_probs: list = [] # shape is (num_consumers, num_firms + 1)
        rewards: list = [] # shape is (num_consumers,)

        c_state = ConsumerState(
            self.tax_rate,
            self.items_prices,
            self.items_quantities,
            self.consumers_wages,
            self.environment_params["labour_disutility"],
            self.environment_params["crra_uf_param"],
            self.consumers_budgets
        )

        # Get actions for each consumer
        for i in range(self.environment_params["num_consumers"]):
            policy_input = c_state.getConsumerState(i)
            policy_output = self.consumer_policy.act(policy_input)
            choices.append(policy_output[:-1])
            log_probs.append(policy_output[-1])
            # choices.append(policy_output[::2])
            # log_probs.append(policy_output[1::2])
        choices = np.array(choices) # shape is (num_consumer, num_firms + 1)
        #print(choices)

        # Get item consumption for each consumer
        item_scaled_consumption: list = []
        for i in range(self.environment_params["num_firms"]):
            item_demand = choices[:, i].reshape(-1) # In this case choice represents real number (unlike working_hours, etc.)
            item_available = self.items_quantities[i]
            item_scaled_consumption.append(availableGoods(item_available, item_demand))
        item_scaled_consumption = np.array(item_scaled_consumption).T # shape is (num_consumer, num_firms)

        # Sequnce of actions: salary, taxes, consumption
        working_hours_choices: list = []
        received_salaries: list = []
        payed_taxes: list = []
        updated_budgets: list = []
        real_consumed_items: list = []
        for i in range(self.environment_params["num_consumers"]):
            # Map wh choice to actual number of wh
            chosen_working_hours = self.possible_working_hours[choices[i][-1]]
            # Calculate received salary
            received_salary = chosen_working_hours*self.consumers_wages[i]*(1-self.tax_rate)
            # Calculate payed taxes
            payed_tax = chosen_working_hours*self.consumers_wages[i]*self.tax_rate
            # Calculate new available budget (curr budget + salary)
            available_budget = c_state.budget[i] + received_salary #+ self.distributed_tax
            # Select items consumption for i-th user
            consumer_items_consumption = item_scaled_consumption[i, :].reshape(-1).tolist()
            # Map user's wh to firm
            working_hours = [0]*self.environment_params["num_firms"]
            working_hours[consumer_to_firm[i]] = chosen_working_hours
            reward, budget_after_purchase, real_items_consumption = consumerUtility(
                consumer_items_consumption, self.items_prices,
                working_hours, self.environment_params["labour_disutility"],
                self.environment_params["crra_uf_param"],
                available_budget
            )
            # print(reward)
            # Append all values needed for further timestamps and agents
            rewards.append(reward) #/self.consumer_params["reward_scaling_factor"])
            working_hours_choices.append(chosen_working_hours)
            received_salaries.append(received_salary)
            payed_taxes.append(payed_tax)
            updated_budgets.append(budget_after_purchase)
            real_consumed_items.append(real_items_consumption)
        
        # Cast some lists to np.ndarrays
        rewards = np.array(rewards) # shape is (num_consumers,)
        real_consumed_items = np.array(real_consumed_items)
        
        # Update env vars such as cosumers' budgets, items iventory etc.
        self.consumers_budgets = updated_budgets
        self.consumers_working_hours = working_hours_choices
        self.items_quantities = (np.array(self.items_quantities) -
                                 np.sum(real_consumed_items, axis=0)).tolist()

        # Aggregate some variables among consumers
        total_payed_taxes = np.sum(payed_taxes)
        total_received_salaries = np.sum(received_salaries)
        # firm_item_scaled_consumption = np.sum(item_scaled_consumption, axis=0)
        firm_item_scaled_consumption = np.sum(real_consumed_items, axis=0)

        return rewards, log_probs, firm_item_scaled_consumption, total_received_salaries, total_payed_taxes
    
    def simulateFirms(self, items_total_consumption: np.ndarray):
        choices: list = [] # shape is (num_firms, 2)
        log_probs: list = [] # shape is (num_firms, 2)
        rewards: list = []

        # Calculate labour available to each firm
        total_labour_per_firm = [0] * self.environment_params["num_firms"] # add produced items
        for i in range(self.environment_params["num_consumers"]):
            firm_id = self.consumer_to_firm[i]
            num_hours = self.consumers_working_hours[i]
            total_labour_per_firm[firm_id] += num_hours

        # Calculate reward for each firm, update their budgets and capital
        payed_salaries: list = []
        payed_taxes: list = []
        updated_budgets: list = []
        updated_capitals: list = []
        produced_items: list = []
        firms_investments: list = []
        for i in range(self.environment_params["num_firms"]):
            # Caclute reward (profit after taxes and payed taxes)
            payed_salary = total_labour_per_firm[i]*self.firms_wages[i]
            reward = items_total_consumption[i]*self.items_prices[i]*(1-self.tax_rate)-payed_salary
            # Calculate payed taxes
            payed_tax = items_total_consumption[i]*self.items_prices[i]*self.tax_rate
            # Calculate new available budget
            updated_budget = (self.firms_budgets[i]+reward)*(1-self.environment_params["investments_percent"])
            # Calculate investments
            investments = (self.firms_budgets[i]+reward)*self.environment_params["investments_percent"]
            # Calculate new available capital
            updated_capital = self.firms_capitals[i]+investments
            # Calculate produced items
            firm_produced_items = productionFunction(total_labour_per_firm[i], updated_capital, self.firms_pf_alphas[i])
            # Append all values needed for further timestamps and agents
            rewards.append(reward) #/self.firm_params["reward_scaling_factor"])
            payed_salaries.append(payed_salary)
            payed_taxes.append(payed_tax)
            updated_budgets.append(updated_budget)
            updated_capitals.append(updated_capital)
            produced_items.append(firm_produced_items)
            firms_investments.append(investments)

        f_state = FirmState(
            total_labour_per_firm,
            updated_capitals,
            self.firms_pf_alphas,
            updated_budgets,
            self.tax_rate,
            firms_investments
        )

        # get actions for each firm
        items_new_prices: list = []
        new_wages: list = []
        for i in range(self.environment_params["num_firms"]):
            policy_input = f_state.getFirmState(i)
            policy_output = self.firm_policy.act(policy_input)
            choices.append(policy_output[:-1])
            log_probs.append(policy_output[-1])
            items_new_prices.append(self.possible_prices[choices[i][0]])
            new_wages.append(self.possible_salaries[choices[i][1]])
        choices = np.array(choices)

        # Update env vars such as wages (new items prices are set after government acts)
        self.firms_budgets = updated_budgets
        self.firms_capitals = updated_capitals
        self.firms_wages = new_wages
        self.items_quantities = (np.array(self.items_quantities) +
                                 np.sum(produced_items, axis=0)).tolist()
        for i in range(self.environment_params["num_consumers"]):
            self.consumers_wages[i] = self.firms_wages[self.consumer_to_firm[i]]
        
        # Cast some lists to np.ndarrays
        rewards = np.array(rewards)

        # Aggregate some variables among firms
        total_payed_taxes = np.sum(payed_taxes)
        total_payed_salaries = np.sum(payed_salaries)


        return rewards, log_probs, items_new_prices, total_payed_taxes, total_payed_salaries

    def simulateGovernment(self, consumers_payed_taxes: float, firms_payed_taxes: float,
                           firms_payed_wages: int, consumers_rewards: List[float],
                           firms_rewards: List[float]):
        choices: list = [] # shape is (1,)
        log_probs: list = [] # shape is (1,)
        rewards: list = []

        self.distributed_tax = ((consumers_payed_taxes+firms_payed_taxes)/
                                (self.environment_params["num_consumers"]))
        # print(self.distributed_tax)
        g_state = GovernmentState(
            self.environment_params["num_consumers"],
            self.environment_params["num_firms"],
            np.sum(self.consumers_working_hours),
            firms_payed_wages,
            consumers_payed_taxes+firms_payed_taxes,
            self.items_quantities,
            self.items_prices
        )

        policy_input = g_state.getGovernmentState()
        policy_output = self.government_policy.act(policy_input)
        choices.append(policy_output[0])
        log_probs.append(policy_output[1])

        choices = np.array(choices)

        rewards.append((np.sum(consumers_rewards) + np.sum(firms_rewards))) #/self.government_params["reward_scaling_factor"])

        # Update some env vars
        self.tax_rate = self.possible_taxes[choices[0]]
        
        # Cast some lists to np.ndarrays
        rewards = np.array(rewards)

        return rewards, log_probs
    
    def reinforce(self,
                  consumers_rewards: List[List[float]],
                  consumers_log_probs: List[List[torch.Tensor]],
                  firms_rewards: List[List[float]],
                  firms_log_probs: List[List[torch.Tensor]],
                  government_rewards: List[List[float]],
                  government_log_probs: List[List[torch.Tensor]],) -> None:
        consumers_mean_loss, cosumers_mean_return = calcAgentsMeanLoss(
            consumers_rewards, consumers_log_probs, self.environment_params["discount_factor"]
        )
        firms_mean_loss, firms_mean_return = calcAgentsMeanLoss(
            firms_rewards, firms_log_probs, self.environment_params["discount_factor"]
        )
        government_mean_loss, government_mean_return = calcAgentsMeanLoss(
            government_rewards, government_log_probs, self.environment_params["discount_factor"]
        )
        print(f"Consumers' loss: {round(consumers_mean_loss.item(), 5)}, "
              f"firms' loss: {round(firms_mean_loss.item(), 5)}, "
              f"government loss: {round(government_mean_loss.item(), 5)}")
        print(f"Consumers' return: {round(cosumers_mean_return, 5)}, "
              f"firms' return: {round(firms_mean_return, 5)}, "
              f"government return: {round(government_mean_return, 5)}")
        print()

        self.consumer_optimizer.zero_grad()
        consumers_mean_loss.backward()
        self.consumer_optimizer.step()

        self.firm_optimizer.zero_grad()
        firms_mean_loss.backward()
        self.firm_optimizer.step()

        self.government_optimizer.zero_grad()
        government_mean_loss.backward()
        self.government_optimizer.step()

    def run(self) -> None:
        print("Init seeds for all random things")
        initSeeds(self.tech_params.get("seed", 666))

        print("Start initializing policies")
        self.initPolicies()

        print("Init optimizers")
        self.initOptimizers()

        print(f"Start training neural networks, number of epochs: {self.environment_params['epochs']}")
        for i in range(self.environment_params["epochs"]):
            #print(f"Training epoch: {i+1}")
            # print("Init variables for new epoch")
            consumers_rewards_list: list = []
            consumers_log_probs_list: list = []
            firms_rewards_list: list = []
            firms_log_probs_list: list = []
            government_rewards_list: list = []
            government_log_probs_list: list = []

            # print("Initialize mapping dicts for possible salaries, prices etc. for new epoch")
            self.initMappingDicts()

            for t in range(self.environment_params["timesteps"]):
                # Simulate consumers
                (consumers_rewards, consumers_log_probs,
                 item_scaled_consumption,
                 consumers_received_salaries,
                 consumers_payed_taxes) = self.simulateConsumers(self.consumer_to_firm)
                consumers_rewards_list.append(consumers_rewards)
                consumers_log_probs_list.append(consumers_log_probs)

                # Simulate firms
                (firms_rewards, firms_log_probs,
                 new_items_prices, firms_payed_taxes,
                 firms_payed_salaries) = self.simulateFirms(item_scaled_consumption)
                firms_rewards_list.append(firms_rewards)
                firms_log_probs_list.append(firms_log_probs)

                # Simulate government
                government_reward, government_log_prob = self.simulateGovernment(
                    consumers_payed_taxes, firms_payed_taxes,
                    firms_payed_salaries, consumers_rewards,
                    firms_rewards
                )
                government_rewards_list.append(government_reward)
                government_log_probs_list.append(government_log_prob)

                # Update env var
                self.items_prices = new_items_prices

                print(self.items_prices)
                print(self.consumers_working_hours)
                # print(self.consumers_wages)
                # print(self.consumers_budgets)
                # print(self.items_quantities)
                # print(self.tax_rate)
                # # print(self.possible_salaries)
                # print()

            print(
                np.mean(
                    consumers_rewards_list[-1],
                    axis=0
                )
            )
            print(
                np.mean(
                    firms_rewards_list[-1],
                    axis=0
                )
            )
            # print(firms_rewards_list)
            # print(self.consumer_policy.working_hours_head.weight)
            # Trim firms' and government's lists, because their actions imply on t+1
            firms_rewards_list = firms_rewards_list[1:]
            firms_log_probs_list = firms_log_probs_list[:-1]
            government_rewards_list = government_rewards_list[1:]
            government_log_probs_list = government_log_probs_list[:-1]

            
            # Transpose lists to shape (num_agents, num_timesteps)
            consumers_rewards_list = transposeList2d(consumers_rewards_list)
            consumers_log_probs_list = transposeList2d(consumers_log_probs_list)
            firms_rewards_list = transposeList2d(firms_rewards_list)
            firms_log_probs_list = transposeList2d(firms_log_probs_list)
            government_rewards_list = transposeList2d(government_rewards_list)
            government_log_probs_list = transposeList2d(government_log_probs_list)

            # REINFORCE algorithm goes here
            self.reinforce(
                consumers_rewards_list, consumers_log_probs_list,
                firms_rewards_list, firms_log_probs_list,
                government_rewards_list, government_log_probs_list
            )

            # print(consumers_log_probs_list[0][0][0]*consumers_log_probs_list[0][0][1])
            # print(consumers_log_probs_list)
            # print(firms_rewards_list)
            # print(government_rewards_list)


if __name__ == "__main__":
    fire.Fire(Pipeline)
