import yaml
from importlib_resources import files, as_file
import random
import numpy as np
import torch

import rl_economics.configs


def loadYamlConfig(yaml_file_name: str) -> dict:
    yaml_config: dict

    with open(yaml_file_name, "r") as stream:
        try:
            yaml_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return yaml_config


def initSeeds(seed: int = 666) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


def transposeList2d(input_list: list) -> list:
    res = []
    for i in range(len(input_list[0])):
        column = []
        for j in range(len(input_list)):
            column.append(input_list[j][i])
        res.append(column)
    return res
