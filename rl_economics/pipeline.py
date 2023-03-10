import os
import fire
from torch.optim import Adagrad
from typing import List

from rl_economics.utils.misc import loadYamlConfig, initSeeds
from rl_economics.functions.general import consumersToFirmsDistribution
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
    
    def initPolicies(self) -> None:
        print("Init consumer policy")
        num_working_hours = ((self.consumer_params["model"]["working_hours"]["max"] -
                              self.consumer_params["model"]["working_hours"]["min"]) //
                             self.consumer_params["model"]["working_hours"]["step"])
        consumer_num_unput_features = (len(ConsumerState.__dict__["__match_args__"]) - 2 +
                                       2*self.environment_params["num_firms"] +
                                       self.environment_params["num_consumers"])
        print(consumer_num_unput_features)
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
                      self.firm_params["model"]["prices"]["step"])
        num_wages = ((self.firm_params["model"]["wages"]["max"] -
                      self.firm_params["model"]["wages"]["min"]) //
                     self.firm_params["model"]["wages"]["step"])
        self.firm_policy = FirmPolicy(
            num_input_features=1,
            num_prices=num_prices,
            num_wages=num_wages,
            mlp_layer_width=self.firm_params["model"].get("mlp_layer_width", 128)
        )

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

    def run(self) -> None:
        print("Init seeds for all random things")
        initSeeds(self.tech_params.get("seed", 666))

        print("Start initializing policies")
        self.initPolicies()

        print("Create inital states")

        print("Distribute consumers among firms")
        consumersToFirmsDistribution(
            num_consumers=self.environment_params["num_consumers"],
            num_firms=self.environment_params["num_firms"]
        )

        print("Set initial states for consumers, firms and government")
        # code goes here

        print(f"Start training neural networks, number of epochs: {self.environment_params['epochs']}")
        for i in range(self.environment_params["epochs"]):
            # code goes here
            consumer_states: List[ConsumerState] = []

            c_state = ConsumerState(
                curr_tax=[0.0]*self.environment_params["num_consumers"],
                item_prices=[[1000]*self.environment_params["num_firms"]]*self.environment_params["num_consumers"],
                item_quantities=[[0]*self.environment_params["num_firms"]]*self.environment_params["num_consumers"],
                wage=[0]*self.environment_params["num_consumers"],
                working_hours=[0]*self.environment_params["num_consumers"],
                labour_disutility=[self.environment_params["labour_disutility"]]*self.environment_params["num_consumers"],
                crra_uf_param=[self.environment_params["crra_uf_param"]]*self.environment_params["num_consumers"],
                budget=[0.0]*self.environment_params["num_consumers"]
            )
            # print(c_state.getConsumerState(j).shape)
            for j in range(self.environment_params["num_consumers"]):
                action = self.consumer_policy.act(c_state.getConsumerState(j))
                action_items = action[::2]
                action_probs = action[1::2]
                print(action_items)
                print(action_probs)
                break


if __name__ == "__main__":
    fire.Fire(Pipeline)
