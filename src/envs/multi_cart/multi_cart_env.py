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

import pyglet

# Base class import
from envs.multiagentenv import MultiAgentEnv
from components.locality_graph import DependencyGraph

# Multicartpole data
import envs.multi_cart.constants as constants
import envs.multi_cart.single_cart as single_cart


class MultiCartPoleEnv(MultiAgentEnv):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(
            self,
            cartdist=constants.CARTDIST,
            num_cartpoles=constants.CARTPOLES,
            obs_radius=constants.OBSERVATION_RADIUS,
            coupled=None,
            rules=None,
            test_physics=False,
            exp_logger=None,
            episode_limit=500,
            animation_speed=0,
            learner_name="default_learner",
    ):
        # Save parameters
        self.params = {
            "cartdist": cartdist if test_physics else cartdist,
            "num_cartpoles": num_cartpoles,
            "obs_radius": obs_radius,
            "test_physics": test_physics,
            "bottom_threshold": -cartdist,
            "top_threshold": cartdist * num_cartpoles,
            "episode_limit": episode_limit,
            "animation_speed": animation_speed,
            "screen": {
                "width": 700,
                "height": 400,
                "carty": 100,
                "polewidth": 10.0,
                "cartwidth": 50.0,
                "cartheight": 30.0,
                "springwidth": 10.0,
            },
            "physics": {
                "masscart": 1.0,
                "length": 0.5,
                "force_mag": 10.0,
                "tau": 0.02,  # resolution of the simulation (dt)
                "kinematics_integrator": 'euler'
            },
            "coupled": coupled,
            "rules": rules
        }

        # More parameter adjusting
        self.params["coupled"]["resting_dist"] = self.params["cartdist"]
        self.params["world_width"] = self.params["top_threshold"] - self.params["bottom_threshold"]
        self.params["scale"] = self.params["screen"]["width"] / self.params["world_width"]
        self.params["screen"]["polelen"] = self.params["scale"] * (2 * self.params["physics"]["length"])
        self.params["physics"]["masspole"] = 0.0 if self.params["test_physics"] else 0.1
        self.params["physics"]["total_mass"] = self.params["physics"]["masspole"] + self.params["physics"]["masscart"]
        self.params["physics"]["polemass_length"] = self.params["physics"]["masspole"] * self.params["physics"][
            "length"]

        # spring base positions, start and finish
        self.left_string_pos = self.params["bottom_threshold"] if self.params["rules"]["edge_springs"] else None
        self.right_string_pos = self.params["top_threshold"] if self.params["rules"]["edge_springs"] else None

        # Other important parameters for pymarl
        self.episode_limit = self.params["episode_limit"]
        self.episode_steps = 0
        self.n_agents = num_cartpoles

        # Crate a SingleCart object per CARTPOLE
        self.cartpoles = [
            single_cart.SingleCart(
                params=self.params,
                offset=i * self.params["cartdist"],
            )
            for i in range(self.params["num_cartpoles"])
        ]
        self.springs = []

        # define spaces
        self.action_space = spaces.MultiBinary(self.params["num_cartpoles"])
        self.observation_space = spaces.Tuple(tuple([
            spaces.Box(-constants.HIGH, constants.HIGH, dtype=np.float32)
            for _ in range(self.params["num_cartpoles"])
        ]))

        # self.seed()
        self.viewer = None
        self.steps_beyond_done = None

        # Saving data per episode
        self.learner_name = learner_name
        self.exp_logger = exp_logger
        self.episode_data = []

        # wait until all cartpoles are done?
        self.cart_alive = [True for _ in range(self.params["num_cartpoles"])]
        self.last_reward = -1
        self.reward_label = None

        # In the MultiCart enviroment we can model the interactions as a single line graph. This is the default
        # architecture
        self.graph_obj = DependencyGraph(
            graph=DependencyGraph.build_simple_graph(self.params["num_cartpoles"], graph_type="line"),
            num_agents=self.params["num_cartpoles"],
        )

    def get_graph_obj(self):
        return self.graph_obj

    def step(self, action):
        # STEP 1: logistics for this stage

        self.episode_steps += 1

        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        # STEP 2: calculate the reward of this current state - currently assume only r(s)
        # Here we compute the reward as a function of the current state - not the new one

        # get rewards for this run
        rewards = [not cartpole.is_done() for cartpole in self.cartpoles]
        # cont_rewards = [cartpole.cont_reward() for cartpole in self.cartpoles]

        # for endless mode, punish cartpoles more harshly so the average reward is more affected
        if self.params["rules"]["terminate"] == "endless":
            rewards = [r if r else -20 for r in rewards]

        self.last_reward = sum(rewards)

        # update status of all cartpoles
        self.cart_alive = rewards

        # STEP 3: take a step for each active cartpole
        for i in range(self.params["num_cartpoles"]):
            if self.cart_alive[i]:
                self.cartpoles[i].step(
                    action[i],
                    left_pos=self.left_string_pos if i == 0 else self.cartpoles[i - 1].get_absolute_state()[0],
                    right_pos=self.right_string_pos if i == (self.params["num_cartpoles"] - 1) else
                    self.cartpoles[i + 1].get_absolute_state()[0]
                )

            # if cart is dead, reset it to it's current location
            elif self.params["rules"]["terminate"] == "endless":
                self.cartpoles[i].reset(save_x=True)

        # STEP 4: Check if should terminate simultation
        times_up = self.episode_steps >= self.params["episode_limit"]
        if self.params["rules"]["terminate"] == "all":
            any_alive = any(self.cart_alive)
            done = times_up or (not any_alive)
        elif self.params["rules"]["terminate"] == "any":
            all_alive = all(self.cart_alive)
            done = times_up or (not all_alive)
        elif self.params["rules"]["terminate"] == "endless":
            done = times_up
        else:
            raise Exception("Termination condition not recognized, muse be one of [any, all, endless]")

        if self.steps_beyond_done is None:
            # One of the poles has fallen
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1

        # return results
        # rewards = cont_rewards  # do continuous rewards for now
        return rewards, done, {}

    def get_info(self):
        return "\n".join([
                             "CARTPOLES INFO\n",
                         ] + [
                             f"CARTPOLE {i}: {self.get_sub_states(i, 0, relative=False)}"
                             for i in range(self.params["num_cartpoles"])
                         ])

    def get_obs(self):
        obs = [self.get_obs_agent(i) for i in range(self.params["num_cartpoles"])]
        return np.array(obs)

    # Recieves all the substates from an agent_id in a certain radius. Returns the RELATIVE x by default
    def get_sub_states(self, agent_id, radius, pad=False, relative=True):
        # relative or absolute position on x axis - doing this thanks to Adi <3
        if relative:
            get_state_func = single_cart.SingleCart.get_relative_state
        else:
            get_state_func = single_cart.SingleCart.get_absolute_state

        # left obs not including the agent
        left_obs = np.array([
            get_state_func(cartpole) for cartpole in
            self.cartpoles[max(0, agent_id - radius): agent_id]
        ])
        right_obs = np.array([
            get_state_func(cartpole) for cartpole in
            self.cartpoles[agent_id: min(self.params["num_cartpoles"], agent_id + radius + 1)]
        ])

        # flatten both sides
        left_obs = left_obs.flatten()
        right_obs = right_obs.flatten()

        # If we are required to pad (the edges) then we do this to either side
        # Note: that the 0's are not there by mistake! they convey the state of
        # a perfect cartpole on either side (assuming relative positions). For
        # either end as far the eye can see.
        # (Historically this happened thanks to a happy mistake)

        if pad:
            # left obs need to include "radius" cartpoles
            left_obs = np.pad(
                left_obs,
                (radius * 4 - left_obs.shape[0], 0),
                "constant",
                constant_values=(0, 0)
            )

            # right obs need to include "radius + 1" cartpoles
            right_obs = np.pad(
                right_obs,
                (0, (radius + 1) * 4 - right_obs.shape[0]),
                "constant",
                constant_values=(0, 0)
            )

        agent_obs = np.concatenate((left_obs, right_obs))
        return agent_obs.flatten()

    def get_obs_agent(self, agent_id):
        return self.get_sub_states(agent_id, self.params["obs_radius"], pad=True)

    def get_state_size(self):
        return self.params["num_cartpoles"] * 4

    def get_reward_size(self):
        return self.params["num_cartpoles"]

    def get_obs_size(self):
        return (2 * self.params["obs_radius"] + 1) * 4

    def get_state(self):
        state = [self.get_sub_states(i, 0) for i in range(self.params["num_cartpoles"])]
        return np.array(state).flatten()

    def get_avail_agent_actions(self, agent_id):
        return np.array([1, 1])

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.params["num_cartpoles"])]

    def get_total_actions(self):
        return 2

    def reset(self):
        for cartpole in self.cartpoles:
            cartpole.reset()

        self.episode_data = []
        self.steps_beyond_done = None
        self.episode_steps = 0
        return self.get_state()

    def get_spring_by_index(self, idx):
        # 0 and n represent the edge springs, all the other ones are the inner springs
        if idx == 0:
            start_spring_x = self.params["bottom_threshold"]
            end_spring_x = self.cartpoles[0].get_absolute_state()[0]
        elif idx == self.params["num_cartpoles"]:
            start_spring_x = self.cartpoles[idx - 1].get_absolute_state()[0]
            end_spring_x = self.params["top_threshold"]
        else:
            start_spring_x = self.cartpoles[idx - 1].get_absolute_state()[0]
            end_spring_x = self.cartpoles[idx].get_absolute_state()[0]

        # find the spring position
        spring_length = end_spring_x - start_spring_x

        l = (start_spring_x + abs(self.params["bottom_threshold"])) * self.params["scale"]
        l += self.params["screen"]["cartwidth"] / 2

        r = (end_spring_x + abs(self.params["bottom_threshold"])) * self.params["scale"]
        r -= self.params["screen"]["cartwidth"] / 2

        t = self.params["screen"]["carty"] + self.params["screen"]["springwidth"] / 2
        b = self.params["screen"]["carty"] - self.params["screen"]["springwidth"] / 2

        spring_pos = [(l, b), (l, t), (r, t), (r, b)]

        # create spring color
        stretch = abs(spring_length - self.params["cartdist"])
        red = 0.5 + min(0.5, stretch / self.params["cartdist"])
        spring_color = (red, 0.1, 0.1)

        # return spring positions and color
        return spring_pos, spring_color

    def render(self, mode='human'):
        init = False
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.params["screen"]["width"], self.params["screen"]["height"])
            init = True

        if self.get_state() is None:
            return None

        for cartpole in self.cartpoles:
            cartpole.render(viewer=self.viewer, init=init)

        # render base springs
        low_idx = 0 if self.params["rules"]["edge_springs"] else 1
        high_idx = self.params["num_cartpoles"] if self.params["rules"]["edge_springs"] else (
                self.params["num_cartpoles"] - 1)

        # render springs and background
        if self.params["coupled"]["mode"]:
            if init:
                # render bases
                l1 = 0
                r1 = self.params["screen"]["cartwidth"] / 2
                l2 = self.params["screen"]["width"] - self.params["screen"]["cartwidth"] / 2
                r2 = self.params["screen"]["width"]
                b = self.params["screen"]["carty"] - self.params["screen"]["cartheight"] / 2
                t = self.params["screen"]["carty"] + 2 * self.params["screen"]["cartheight"]

                base1 = rendering.FilledPolygon([(l1, b), (l1, t), (r1, t), (r1, b)])
                base2 = rendering.FilledPolygon([(l2, b), (l2, t), (r2, t), (r2, b)])

                for base in [base1, base2]:
                    base.set_color(0, 0, 0)
                    self.viewer.add_geom(base)

                # render springs
                for i in range(low_idx, high_idx + 1):
                    spring_pos, spring_color = self.get_spring_by_index(i)

                    # create spring object
                    spring = rendering.FilledPolygon(spring_pos)
                    spring.set_color(*spring_color)

                    self.viewer.add_geom(spring)
                    self.springs.append(spring)

            # render springs again
            for i, spring_idx in enumerate(range(low_idx, high_idx + 1)):
                spring_pos, spring_color = self.get_spring_by_index(spring_idx)

                self.springs[i].v = spring_pos
                self.springs[i].set_color(*spring_color)

        time.sleep(self.params["animation_speed"])
        self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def save_step(self, t_env, step_reward):
        # No use for now in saving each time step, maybe later ther will be when we do independent cartpoles
        pass

    def save_episode(self, t_env, episode_reward):
        if self.exp_logger is not None:
            self.exp_logger.save_env_data(
                learner_name=self.learner_name,
                env_data={
                    "t_env": t_env,
                    "step_reward": None,
                    "episode_reward": episode_reward,
                }
            )
