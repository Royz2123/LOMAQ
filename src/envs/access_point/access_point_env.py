"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
import time
from gym import spaces, logger
import numpy as np
from gym.envs.classic_control import rendering

import networkx as nx

import pyglet

# Base class import
from envs.multiagentenv import MultiAgentEnv
from components.locality_graph import DependencyGraph

# Multicartpole data
import envs.multi_cart.constants as constants
import envs.multi_cart.single_cart as single_cart


class AccessPointEnv(MultiAgentEnv):
    def __init__(self, grid_size, params, episode_limit, animation_speed, learner_name="default_learner",
                 exp_logger=None, ):
        # Save parameters
        self.params = {
            "grid_size": grid_size,
            "params": params,
            "episode_limit": episode_limit,
            "animation_speed": animation_speed
        }

        # Other important parameters for pymarl
        self.episode_limit = self.params["episode_limit"]
        self.episode_steps = 0

        self.grid_size = grid_size
        self.n_agents = grid_size * grid_size

        # self.seed()

        # Saving data per episode
        self.exp_logger = exp_logger

        # wait until all cartpoles are done?
        self.last_reward = -1

        # create state array
        self.state = dict()
        self.access_points = dict()
        self.reset()

        # In the AccessPointEnv enviroment we can model the interactions as a grid. This is the default
        # architecture
        self.graph_obj = self.graph_obj = DependencyGraph(
            graph=self.create_nx_graph(),
            num_agents=self.n_agents,
        )

        # define spaces
        # every user can send to any access point in it's vicinity
        self.action_space = spaces.MultiDiscrete([
            (1 + len(self.state[agent_index]["access_points"]))
            for agent_index in range(self.n_agents)
        ])
        self.observation_space = spaces.MultiDiscrete([
            self.state[agent_index]["di"]
            for agent_index in range(self.n_agents)
        ])

    def reset(self):
        self.state = dict()
        self.access_points = dict()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                agent_index = self.position_to_index((i, j))
                ap_positions = self.get_grid_neighbors_pos(agent_index, delta=0.5, cross=True)

                self.state[agent_index] = {
                    "position": (i, j),
                    "di": self.params["params"]["di"],
                    "pi": self.set_probability("pi"),
                    "queue": [0] * self.params["params"]["di"],
                    "access_points": ap_positions
                }

                for pos in ap_positions:
                    # might override existing uniform probabilities, doesnt matter
                    self.access_points[pos] = {
                        "qk": self.set_probability("qk"),
                        "waiting": []
                    }

    def set_probability(self, param_name):
        val = self.params["params"][param_name]

        if val == "uniform":
            return np.random.uniform(low=0, high=1, size=(1,))[0]
        else:
            try:
                return float(val)
            except:
                raise Exception(f"Unrecognised value for parameter {param_name}, {val}")

    def position_to_index(self, position):
        return int(position[0] * self.grid_size + position[1])

    def index_to_position(self, index):
        return index // self.grid_size, index % self.grid_size

    def get_grid_neighbors_pos(self, agent_index, delta=1.0, cross=False):
        position = self.index_to_position(agent_index)

        possible_nbrs = [
            (position[0] + delta, position[1] - cross * delta),
            (position[0] + cross * delta, position[1] + delta),
            (position[0] - delta, position[1] + cross * delta),
            (position[0] - cross * delta, position[1] - delta)
        ]
        nbrs = [pos for pos in possible_nbrs if self.is_valid_position(pos)]

        return nbrs

    def get_grid_neighbors(self, agent_index, delta=1.0):
        nbrs = self.get_grid_neighbors_pos(agent_index, delta=delta)
        nbrs = [self.position_to_index(pos) for pos in nbrs]
        return nbrs

    def is_valid_position(self, position):
        return all([self.is_valid_coordinate(coord) for coord in position])

    def is_valid_coordinate(self, coordinate):
        return (coordinate >= 0) and (coordinate <= (self.grid_size - 1))

    def create_nx_graph(self):
        graph = nx.Graph()

        # add all the nodes
        for agent_index in range(self.n_agents):
            graph.add_node(agent_index)

        # add all the edges
        for agent_index in range(self.n_agents):
            nbrs = self.get_grid_neighbors(agent_index)
            for nbr in nbrs:
                graph.add_edge(nbr, agent_index)

        return graph

    def step(self, action):
        self.episode_steps += 1

        action = action.detach().numpy()
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # At each time step, user ui can choose to send the earliest packet in its queue to one of the access
        # points in its available set Yi
        for agent_index, agent_info in self.state.items():
            # "where null represents the action of not sending"
            if action[agent_index] == 0:
                continue

            # "When the user has an empty queue, then all actions
            # will be understood as the null action."
            if not any(agent_info["queue"]):
                continue

            # " At each time step, if ui’s queue is non-empty and it takes
            # action ai = yk ∈ Yi
            # , i.e. sending the packet to access point yk"

            # action is not 0 - decrease 1 from the index and that is the index of the access_point in the neighborhood
            chosen_action = action[agent_index]
            # print(self.get_info())
            # print(agent_index)
            # print(chosen_action)
            # print(self.action_space)
            # print(self.state[agent_index]["access_points"])
            chosen_ap_pos = self.state[agent_index]["access_points"][chosen_action - 1]
            self.access_points[chosen_ap_pos]["waiting"].append(agent_index)

        rewards = [0] * self.n_agents
        for position, ap_info in self.access_points.items():
            # nothing to send
            if len(ap_info["waiting"]) == 0:
                continue

            # however, if another user chooses to send a packet to the same access point (i.e.
            # a collision), neither packet is sent
            if len(ap_info["waiting"]) > 1:
                continue

            # "then the packet is transmitted with
            # success probability qk that depends on the access point yk"
            if np.random.uniform(low=0, high=1, size=(1,))[0] > ap_info["qk"]:
                continue

            # packet successfully sent out
            agent_index = ap_info["waiting"][0]
            rewards[agent_index] += 1

            # At each time step, if
            # the packet is successfully sent out (we will define “send out” later), it will be removed from the
            # queue
            # queue is ordered as [e1, e2, ... , edi]
            queue = self.state[agent_index]["queue"]
            queue[queue.index(1)] = 0

        # clean waiting list for next iteration
        for _, ap_info in self.access_points.items():
            ap_info["waiting"] = []

        # Move up every packet by 1 in the queue
        for agent_index, agent_info in self.state.items():
            # packet arrival probability for user ui
            new_packet = 0
            if np.random.uniform(low=0, high=1, size=(1,))[0] > agent_info["pi"]:
                new_packet = 1

            # "it will be removed from the
            # queue; otherwise, its deadline will decrease by 1 and is discarded immediately from the queue if its
            # remaining deadline is zero. "
            queue = agent_info["queue"]
            agent_info["queue"] = queue[1:] + [new_packet]

        # check if should terminate simultation
        done = self.episode_steps >= self.params["episode_limit"]

        # if self.steps_beyond_done is None:
        #     # One of the poles has fallen
        #     self.steps_beyond_done = 0
        # else:
        #     if self.steps_beyond_done == 0:
        #         logger.warn(
        #             "You are calling 'step()' even though this "
        #             "environment has already returned done = True. You "
        #             "should always call 'reset()' once you receive 'done = "
        #             "True' -- any further steps are undefined behavior."
        #         )
        #     self.steps_beyond_done += 1

        # return results
        return rewards, done, {}

    def get_info(self):
        return "\n".join([
                             "Access Points Info:\n",
                         ] + [
                             f"User {i}: Position - {agent_info['position']}, Queue - {agent_info['queue']}"
                             + f", Access Points - {agent_info['access_points']}"
                             for i, agent_info in self.state.items()
                         ] + [
                             f"AP: Position - {pos}, Waiting - {ap_info['waiting']}"
                             for pos, ap_info in self.access_points.items()
                         ])

    def get_obs_agent(self, agent_id):
        return self.state[agent_id]["queue"]

    def get_obs(self):
        obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return np.array(obs)

    def get_state_size(self):
        return self.n_agents * self.params["params"]["di"]

    def get_reward_size(self):
        return self.n_agents

    def get_obs_size(self):
        return self.params["params"]["di"]

    def get_state(self):
        state = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return np.array(state).flatten()

    def get_avail_agent_actions(self, agent_id):
        return np.array([1] * self.get_total_actions())

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_total_actions(self):
        return self.params["params"]["di"] + 1

    def render(self, mode='human'):
        xs, ys, colors = [], [], []

        for agent_index, agent_info in self.state.items():
            xs.append(agent_info["position"][0])
            ys.append(agent_info["position"][1])

        for pos, info in self.access_points.items():
            xs.append(pos[0])
            ys.append(pos[1])

        plt.scatter(xs, ys)
        plt.show()
        time.sleep(self.params["animation_speed"])

    #     init = False
    #     if self.viewer is None:
    #         self.viewer = rendering.Viewer(self.params["screen"]["width"], self.params["screen"]["height"])
    #         init = True
    #
    #     if self.get_state() is None:
    #         return None
    #
    #     for cartpole in self.cartpoles:
    #         cartpole.render(viewer=self.viewer, init=init)
    #
    #     # render base springs
    #     low_idx = 0 if self.params["rules"]["edge_springs"] else 1
    #     high_idx = self.params["num_cartpoles"] if self.params["rules"]["edge_springs"] else (
    #                 self.params["num_cartpoles"] - 1)
    #
    #     # render springs and background
    #     if self.params["coupled"]["mode"]:
    #         if init:
    #             # render bases
    #             l1 = 0
    #             r1 = self.params["screen"]["cartwidth"] / 2
    #             l2 = self.params["screen"]["width"] - self.params["screen"]["cartwidth"] / 2
    #             r2 = self.params["screen"]["width"]
    #             b = self.params["screen"]["carty"] - self.params["screen"]["cartheight"] / 2
    #             t = self.params["screen"]["carty"] + 2 * self.params["screen"]["cartheight"]
    #
    #             base1 = rendering.FilledPolygon([(l1, b), (l1, t), (r1, t), (r1, b)])
    #             base2 = rendering.FilledPolygon([(l2, b), (l2, t), (r2, t), (r2, b)])
    #
    #             for base in [base1, base2]:
    #                 base.set_color(0, 0, 0)
    #                 self.viewer.add_geom(base)
    #
    #             # render springs
    #             for i in range(low_idx, high_idx + 1):
    #                 spring_pos, spring_color = self.get_spring_by_index(i)
    #
    #                 # create spring object
    #                 spring = rendering.FilledPolygon(spring_pos)
    #                 spring.set_color(*spring_color)
    #
    #                 self.viewer.add_geom(spring)
    #                 self.springs.append(spring)
    #
    #         # render springs again
    #         for i, spring_idx in enumerate(range(low_idx, high_idx + 1)):
    #             spring_pos, spring_color = self.get_spring_by_index(spring_idx)
    #
    #             self.springs[i].v = spring_pos
    #             self.springs[i].set_color(*spring_color)
    #
    #     time.sleep(self.params["animation_speed"])
    #     self.viewer.render(return_rgb_array=mode == 'rgb_array')
    #
    # def close(self):
    #     if self.viewer:
    #         self.viewer.close()
    #         self.viewer = None
    #
    # def save_step(self, t_env, step_reward):
    #     # No use for now in saving each time step, maybe later ther will be when we do independent cartpoles
    #     pass
    #
    # def save_episode(self, t_env, episode_reward):
    #     if self.exp_logger is not None:
    #         self.exp_logger.save_env_data(
    #             learner_name=self.learner_name,
    #             env_data={
    #                 "t_env": t_env,
    #                 "step_reward": None,
    #                 "episode_reward": episode_reward,
    #             }
    #         )
