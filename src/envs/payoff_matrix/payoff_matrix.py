from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import numpy as np

from components.locality_graph import DependencyGraph


class MatrixEnv(MultiAgentEnv):
    def __init__(
            self,
            a=2.0,
            episode_limit=2,
            reward_setup=0,
            exp_logger=None,
            learner_name="default_learner",
    ):
        # Define the agents and actions
        self.n_agents = 2
        self.n_actions = 2
        self.episode_limit = episode_limit
        self.episode_steps = 0

        if reward_setup == 0:
            self.r_1 = np.array([
                [1, 0.5 * a],
                [1, a],
            ])
            self.r_2 = np.array([
                [a, 0.5 * a],
                [0, 0],
            ])
        elif reward_setup == 1:
            self.r_1 = np.array([
                [0, 1],
                [1, 2],
            ])
            self.r_2 = np.array([
                [2, 1],
                [1, 1],
            ])
        elif reward_setup == 2:
            self.r_1 = np.array([
                [2, 1],
                [1, 2],
            ])
            self.r_2 = np.array([
                [0, 2],
                [2, 2],
            ])
        elif reward_setup == 3:
            self.r_1 = np.array([
                [0, 1],
                [0, 1],
            ])
            self.r_2 = np.array([
                [1, 0],
                [1, 0],
            ])

        self.payoff_matrix = self.r_1 + self.r_2

        self.state = np.ones(2)

        self.graph_obj = DependencyGraph(
            graph=DependencyGraph.build_simple_graph(self.n_agents, graph_type="full"),
            num_agents=self.n_agents,
        )

    def get_reward_size(self):
        return self.n_agents

    def get_graph_obj(self):
        return self.graph_obj

    def render(self):
        pass

    def reset(self):
        """ Returns initial observations and states"""
        self.episode_steps = 0
        return self.state, self.state

    def close(self):
        pass

    def step(self, actions):
        """ Returns reward, terminated, info """
        r_1 = self.r_1[actions[0], actions[1]]
        r_2 = self.r_2[actions[0], actions[1]]
        reward = np.array([r_1, r_2])
        self.episode_steps += 1

        done = self.episode_steps >= self.episode_limit
        return reward, done, {}

    def get_obs(self):
        return [self.state for _ in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.state

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.get_state_size()

    def get_state(self):
        return self.state

    def get_state_size(self):
        """ Returns the shape of the state"""
        return len(self.state)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.n_actions)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return self.n_actions

    def get_stats(self):
        raise NotImplementedError
