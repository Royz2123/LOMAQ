"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import time

import numpy as np
import networkx as nx

from gym import spaces
import torch as th
# import matplotlib
# matplotlib.use('Agg')

# Base class import
from envs.multiagentenv import MultiAgentEnv
from components.locality_graph import DependencyGraph

# from multiagent.multi_discrete import MultiDiscrete
from envs.multi_particle.local_spread import LocalSpreadScenario
from envs.multi_particle.multiagent_particle_env.multiagent import rendering


# We ditched the scenario scheme since we had no need for all the complexity, and our reward scheme was quite different
# The foundations for the code are the same though
class MultiParticleEnv(MultiAgentEnv):
    def __init__(
            self,
            num_agents=1,
            num_landmarks=1,
            rules=None,
            episode_limit=500,
            animation_speed=0,
            exp_logger=None,
            learner_name="default_learner",
    ):
        self.params = {
            "num_agents": num_agents,
            "num_landmarks": num_landmarks,
            "animation_speed": animation_speed,
            "rules": rules
        }
        # if rules is not None:
        #     self.params["rules"] = rules
        # else:
        #     self.params["rules"] = {
        #         "show_landmarks": False,
        #         "collisions_reward": False,
        #         "binary_reward": True,
        #         "graph_type": "empty",
        #     }

        # Other important parameters for pymarl
        self.episode_limit = episode_limit
        self.episode_steps = 0

        # set the number of agents. If grid is used, override the num agents arg
        if self.params["rules"]["grid"]["use_grid"]:
            self.n_agents = self.params["rules"]["grid"]["num_x_agents"] * self.params["rules"]["grid"]["num_y_agents"]

            # Override
            self.params["num_agents"] = self.n_agents
            self.params["num_landmarks"] = self.params["num_agents"]
        else:
            self.n_agents = self.params["num_agents"]

        # Saving data per episode
        self.learner_name = learner_name
        self.exp_logger = exp_logger
        self.episode_data = []

        self.last_reward = -1
        self.reward_label = None

        # the scenario in our case will always be the local spread scenario under different params
        self.scenario = LocalSpreadScenario(params=self.params)
        self.world = self.scenario.make_world()
        self.agents = self.world.policy_agents

        # Build the graph based on the interactions of the particles (needs world to be created)
        self.graph_obj = self.build_graph()
        if self.params["rules"]["graph"]["show_graph"]:
            self.graph_obj.display()

        # define spaces
        # action space - 4 directions + noop
        self.action_space = spaces.Tuple(tuple([
            spaces.Discrete(self.get_total_actions()) for _ in range(self.n_agents)
        ]))

        # observation space - assume homogenous agents
        self.observation_space = spaces.Tuple(tuple([
            spaces.Box(low=-np.inf, high=+np.inf, shape=(self.get_obs_size(),), dtype=np.float32)
            for _ in range(self.n_agents)
        ]))

        # rendering - we allow only single viewer
        self.viewer = None
        self.render_geoms = None
        self.render_geoms_xform = None
        self._reset_render()

    def build_graph(self):
        graph_type = self.params["rules"]["graph"]["graph_type"]
        if graph_type in ["empty", "full", "line"]:
            graph = DependencyGraph.build_simple_graph(self.n_agents, graph_type=graph_type)
        elif graph_type == "auto":
            graph = self.build_auto_graph()
        else:
            raise Exception("Error: uknown graph type: %s" % graph_type)

        return DependencyGraph(num_agents=self.n_agents, graph=graph)

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # builds either the full graph or the empty graph
    def build_auto_graph(self):
        graph = nx.Graph()

        # add all the agents (necessary for the empty case)
        for i in range(self.n_agents):
            graph.add_node(i)

        # add all the agents based on the geometry of the problem
        for i in range(self.n_agents):
            agent = self.world.agents[i]
            for j in self.get_agent_neighbors(i, agent):
                graph.add_edge(i, j)
        return graph

    def get_agent_neighbors(self, agent_idx, agent):
        neighbors_idxs = []

        for agent2_idx, agent2 in enumerate(self.world.agents):
            # Don't want any self loops at this time
            if agent2_idx == agent_idx:
                continue

            # First off - if either of the agents is unbounded, then they can influence each other directly and
            # are considered neighbors
            if (not getattr(agent, "is_bound", False)) or (not getattr(agent2, "is_bound", False)):
                neighbors_idxs.append(agent2_idx)
                continue

            if agent.bound_type != agent2.bound_type:
                raise Exception("Unsupported auto setting for problem with mixed square and circle bounds")

            if agent.bound_type == "square":
                # Both are bounded - need to see if radii intersect since in that case they can collide
                agent1_ranges = LocalSpreadScenario.get_agent_ranges(agent)
                agent2_ranges = LocalSpreadScenario.get_agent_ranges(agent2)

                # We demand overlap in both ranges
                if (
                        LocalSpreadScenario.get_range_overlap(agent1_ranges[0], agent2_ranges[0]) > 0
                        and LocalSpreadScenario.get_range_overlap(agent1_ranges[1], agent2_ranges[1]) > 0
                ):
                    neighbors_idxs.append(agent2_idx)
            elif agent.bound_type == "circle":
                relative_pos = agent.initial_pos - agent2.initial_pos

                # We demand overlap between the 2 circles
                if np.sqrt(np.sum(np.square(relative_pos))) < agent.bound_dist + agent2.bound_dist:
                    neighbors_idxs.append(agent2_idx)

        return neighbors_idxs

    def step(self, actions):
        if th.is_tensor(actions):
            actions = actions.cpu().detach().numpy().tolist()
        else:
            actions = actions.tolist()
        self.episode_steps += 1
        err_msg = "%r (%s) invalid" % (actions, type(actions))
        assert self.action_space.contains(actions), err_msg

        # we now need to compute the global reward
        rewards = self.scenario.compute_all_rewards(self.world)

        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(actions[i], agent)

        # advance world state
        self.world.step()

        # print(self.episode_steps, ":\t", actions, ":\t", self.get_state())
        # time.sleep(0.3)

        # check if times up, and return done
        done = self.episode_steps >= self.episode_limit

        return rewards, done, {}

    # set env action for a particular agent
    def _set_action(self, action, agent):
        agent.action.u = np.zeros(self.world.dim_p)

        # process action, which is a discrete direction
        agent.action.u = np.zeros(self.world.dim_p)
        # process discrete action
        if action == 1:
            agent.action.u[0] = -1.0
        if action == 2:
            agent.action.u[0] = +1.0
        if action == 3:
            agent.action.u[1] = -1.0
        if action == 4:
            agent.action.u[1] = +1.0

        # set sensitivity if necessary
        sensitivity = 5.0
        if agent.accel is not None:
            sensitivity = agent.accel
        agent.action.u *= sensitivity

    def get_obs(self):
        return np.array([self.scenario.observation(agent, self.world, self.graph_obj) for agent in self.agents])

    def get_obs_agent(self, agent_id):
        return self.scenario.observation(self.agents[agent_id], self.world, self.graph_obj)

    def get_obs_size(self):
        return len(self.scenario.observation(self.agents[0], self.world, self.graph_obj))

    def get_state(self):
        return np.array(self.scenario.state(self.world)).flatten()

    def get_state_size(self):
        return len(self.get_state())

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(0) for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        avail_actions = np.array([1] * (self.get_total_actions()))
        return avail_actions

    def get_total_actions(self):
        return self.world.dim_p * 2 + 1

    def reset(self):
        self.episode_steps = 0

        # reset world
        self.scenario.reset_world(self.world)

        # reset renderer
        self._reset_render()

        return self.get_obs()

    # render environment
    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            self.render_geoms = []
            self.render_geoms_xform = []

            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size, res=getattr(entity, "res", 30))
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add the bounds visually - after the entities, shouldnt get updated
            for agent in self.world.agents:
                if agent.is_bound:
                    if agent.bound_type == "circle":
                        geom = rendering.make_circle(radius=agent.bound_dist)
                    elif agent.bound_type == "square":
                        geom = rendering.make_square(radius=agent.bound_dist)
                    else:
                        continue

                    xform = rendering.Transform()
                    geom.set_color(*np.array([0] * 3), alpha=0.05)
                    geom.add_attr(xform)
                    xform.set_translation(newx=agent.initial_pos[0], newy=agent.initial_pos[1])
                    self.render_geoms.append(geom)
                    self.render_geoms_xform.append(xform)

            # add geoms to viewer
            self.viewer.geoms = []
            for geom in self.render_geoms:
                self.viewer.add_geom(geom)

        results = []

        # update bounds to center around agent
        cam_margin = 0.5

        # define spawning margin
        if self.params["rules"]["bounding"]["is_bound"]:
            bound_dist = self.params["rules"]["bounding"]["bound_dist"]
        else:
            bound_dist = 1
        bound_dist += cam_margin

        # define the camera dimensions
        pos = np.array([-bound_dist, bound_dist, -bound_dist, bound_dist])

        if self.params["rules"]["grid"]["use_grid"]:
            x_dist = self.params["rules"]["grid"]["grid_dist_x"] * (self.params["rules"]["grid"]["num_x_agents"] - 1)
            y_dist = self.params["rules"]["grid"]["grid_dist_y"] * (self.params["rules"]["grid"]["num_y_agents"] - 1)
            max_dist = max(x_dist, y_dist) + self.params["rules"]["grid"]["grid_offset"]
            pos += np.array([0, max_dist, 0, max_dist])

        self.viewer.set_bounds(*pos)

        # update geometry positions
        for e, entity in enumerate(self.world.entities):
            self.render_geoms_xform[e].set_translation(*entity.state.p_pos)

        # update colors for landmarks that are occupied
        for e, entity in enumerate(self.world.entities):
            if entity in self.world.landmarks:
                if any([LocalSpreadScenario.is_collision(agent, entity) for agent in self.world.agents]):
                    self.render_geoms[e].set_color(*np.array([0.0, 0.0, 0.0]))
                else:
                    self.render_geoms[e].set_color(*entity.color)

        # render to display or array
        time.sleep(self.params["animation_speed"])
        self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def get_reward_size(self):
        return self.n_agents

    def get_graph_obj(self):
        return self.graph_obj
