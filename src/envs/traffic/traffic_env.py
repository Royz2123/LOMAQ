import numpy as np
import os
import time

from sumo_rl import SumoEnvironment

# Base class import
from envs.multiagentenv import MultiAgentEnv

BASE_PATH = "./src/envs/traffic/nets/"


class TrafficEnv(SumoEnvironment, MultiAgentEnv):
    def __init__(
            self,
            use_gui=False,
            exp_logger=None,
            learner_name="default_learner"
    ):
        self.use_gui = use_gui

        # Logging stuff
        self.exp_logger = exp_logger
        self.output_path = f"{self.exp_logger.learner_name_to_path(learner_name)}output"
        self.episode_data = []
        self.learner_name = learner_name

        super().__init__(
            f'{BASE_PATH}4x4-Lucas/4x4.net.xml',
            f'{BASE_PATH}4x4-Lucas/4x4c1c2c1c2.rou.xml',
            out_csv_name=None,
            use_gui=use_gui,
            num_seconds=1000,
            max_depart_delay=0
        )

        # Set PYMARL stuff
        self.episode_limit = self.sim_max_time // self.delta_time + 1
        self.episode_steps = 0
        self.n_agents = len(self.ts_ids)

    def get_state_size(self):
        return self.observation_space.shape[0] * self.n_agents

    def get_obs_size(self):
        return self.observation_space.shape[0]

    def get_avail_agent_actions(self, agent_id):
        return np.array([1] * self.action_space.n)

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_total_actions(self):
        return self.action_space.n

    def compute_all_observations(self):
        # Sumo RL does something weird in compute_observations,
        # It only sends observations for traffic lights that need to act
        # this returns everything instead
        # TODO: Relevant? Important?
        return {ts: self.traffic_signals[ts].compute_observation() for ts in self.ts_ids}

    def get_obs(self):
        dict_state = self.compute_all_observations()
        state = np.array([dict_state[ts] for ts in self.ts_ids])
        return state

    def get_state(self):
        return self.get_obs().flatten()

    def get_obs_agent(self, agent_id):
        return self._compute_observations()[agent_id]

    def to_sumo_dict(self, lst):
        return dict([(ts, lst[i]) for i, ts in enumerate(self.ts_ids)])

    def to_pymarl_arr(self, dct):
        return np.array([dct.get(ts, 0) for ts in self.ts_ids])

    def get_reward_size(self):
        return self.n_agents

    def step(self, action):
        _, rewards, done, info = SumoEnvironment.step(self, self.to_sumo_dict(action.detach().clone()))

        # TODO: Independent termination
        return self.to_pymarl_arr(rewards), done["__all__"], info

    def render(self):
        pass

    def reset(self):
        self.episode_data = []
        SumoEnvironment.reset(self)

    def save_step(self, t_env, step_reward):
        # Save the current step
        if self.exp_logger is not None:
            self.episode_data.append({
                "t_env": t_env,
                "step_reward": step_reward,
                "episode_reward": None,
            })

    def save_episode(self, t_env, episode_reward):
        # Save the episode
        if self.exp_logger is not None:
            self.episode_data.append({
                "t_env": t_env,
                "step_reward": 0.0,
                "episode_reward": episode_reward,
            })
            self.exp_logger.save_episode(self.learner_name, self.episode_data)
