import os
import time

from sumo_rl import SumoEnvironment

# Base class import
from envs.multiagentenv import MultiAgentEnv

BASE_PATH = "./src/envs/traffic/nets/"


class TrafficEnv(SumoEnvironment, MultiAgentEnv):
    def __init__(self, seed, use_gui=True):
        self.results_output = f"./results/traffic/{time.time()}/"
        os.mkdir(self.results_output)
        self.results_csv = f"{self.results_output}/output.csv"

        self.use_gui = use_gui

        super().__init__(
            f'{BASE_PATH}4x4-Lucas/4x4.net.xml',
            f'{BASE_PATH}4x4-Lucas/4x4c1c2c1c2.rou.xml',
            out_csv_name=self.results_csv,
            use_gui=use_gui,
            num_seconds=80000,
            max_depart_delay=0
        )

        # Set PYMARL stuff
        # TODO: FIXXX
        self.episode_limit = 2000

    def get_state_size(self):
        print(self.ts_ids)
        print(self.observation_space)
        print(self.action_space)
        print(self.observation_space)
        exit()

