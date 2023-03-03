import os
import fire

from rl_economics.utils.misc import loadYamlConfig


class Pipeline:
    def __init__(self, config_file_path: str):
        full_config_file_path = os.path.join(os.getcwd(), config_file_path)
        
        self.simulation_config = loadYamlConfig(full_config_file_path)

        print(self.simulation_config)

    def run(self):
        print("jopa")


if __name__ == "__main__":
    fire.Fire(Pipeline)
