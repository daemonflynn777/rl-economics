import yaml
from importlib_resources import files, as_file
import random
import numpy as np
import torch

import rl_economics.configs


def loadYamlConfig(yaml_file_name: str) -> dict:
    # full_yaml_file_path = files(rl_economics.configs).joinpath(yaml_file_name)
    # yaml_config = {}

    # with as_file(full_yaml_file_path) as yaml_file:
    #     with open(yaml_file, "r") as stream:
    #         try:
    #             yaml_config = yaml.safe_load(stream)
    #         except yaml.YAMLError as exc:
    #             print(exc)
    
    # return yaml_config

    yaml_config: dict

    with open(yaml_file_name, "r") as stream:
        try:
            yaml_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    return yaml_config


# c = loadYamlConfig("/home/nick/Documents/Repos/rl-economics/rl_economics/configs/simulation_config.yml")
# print(c)


def initSeeds(seed: int = 666) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    return None
