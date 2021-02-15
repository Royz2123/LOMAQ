"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering

# Base class import
from envs.multiagentenv import MultiAgentEnv

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
            coupled=True,
            test_physics=False,
            seed=None
    ):
        # Save parameters
        self.params = {
            "cartdist": cartdist,
            "num_cartpoles": num_cartpoles,
            "obs_radius": obs_radius,
            "coupled": {
                "mode": coupled,
                "spring_k": 0.5,
                "resting_dist": cartdist
            },
            "seed": seed,
            "test_physics": test_physics,
            "bottom_threshold": -constants.X_MARGIN,
            "top_threshold": (num_cartpoles - 1) * cartdist + constants.X_MARGIN,
            "episode_limit": 1000,
            "screen": {
                "width": 1000,
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
                "tau": 0.02,
                "kinematics_integrator": 'euler'
            }
        }
        self.params["world_width"] = self.params["top_threshold"] - self.params["bottom_threshold"]
        self.params["scale"] = self.params["screen"]["width"] / self.params["world_width"]
        self.params["screen"]["polelen"] = self.params["scale"] * (2 * self.params["physics"]["length"])
        self.params["physics"]["masspole"] = 0.0 if self.params["test_physics"] else 0.1
        self.params["physics"]["total_mass"] = self.params["physics"]["masspole"] + self.params["physics"]["masscart"]
        self.params["physics"]["polemass_length"] = self.params["physics"]["masspole"] * self.params["physics"][
            "length"]

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
        self.state = None
        self.steps_beyond_done = None

    def step(self, action):
        self.episode_steps += 1

        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        for i in range(self.params["num_cartpoles"]):
            self.cartpoles[i].step(
                action[i],
                left_pos=None if i == 0 else self.cartpoles[i - 1].state[0],
                right_pos=None if i == (self.params["num_cartpoles"] - 1) else self.cartpoles[i + 1].state[0]
            )

        done = (
                any([cartpole.is_done() for cartpole in self.cartpoles])
                or self.episode_steps >= self.params["episode_limit"]
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # One of the poles has fallen
            self.steps_beyond_done = 0
            reward = 0.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 1.0

        # return results
        return [reward] * self.params["num_cartpoles"], done, {}

    def get_info(self):
        return "\n".join([
                             "CARTPOLES INFO\n",
                         ] + [
                             f"CARTPOLE {i}: {self.get_sub_states(i, 0)}"
                             for i in range(self.params["num_cartpoles"])
                         ])

    def get_obs(self):
        obs = [self.get_obs_agent(i) for i in range(self.params["num_cartpoles"])]
        return np.array(obs)

    def get_sub_states(self, agent_id, radius, pad=False):
        agent_obs = np.array([
            cartpole.state for cartpole in
            self.cartpoles[
            max(0, agent_id - radius)
            : min(self.params["num_cartpoles"], agent_id + radius + 1)
            ]
        ])
        agent_obs = agent_obs.flatten()

        if pad:
            agent_obs = np.pad(
                agent_obs,
                (0, self.get_obs_size() - agent_obs.shape[0]),
                "constant",
                constant_values=(0, 0)
            )
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

        self.steps_beyond_done = None
        self.episode_steps = 0
        return self.get_state()

    def render(self, mode='human'):
        init = False
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.params["screen"]["width"], self.params["screen"]["height"])
            init = True

        if self.get_state() is None:
            return None

        for cartpole in self.cartpoles:
            cartpole.render(viewer=self.viewer, init=init)

        if self.params["coupled"]["mode"]:
            # render springs and background
            if init:
                for i in range(self.params["num_cartpoles"] - 1):
                    start_spring_x = self.cartpoles[i].state[0]
                    end_spring_x = self.cartpoles[i + 1].state[0]
                    spring_length = end_spring_x - start_spring_x

                    l, r = -spring_length / 2, spring_length / 2
                    t, b = self.params["screen"]["springwidth"] / 2, -self.params["screen"]["springwidth"] / 2

                    l *= self.params["scale"]
                    r *= self.params["scale"]

                    spring = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    self.viewer.add_geom(spring)
                    self.springs.append(spring)

            for i in range(self.params["num_cartpoles"] - 1):
                start_spring_x = self.cartpoles[i].state[0]
                end_spring_x = self.cartpoles[i + 1].state[0]
                spring_length = end_spring_x - start_spring_x

                l = (start_spring_x + abs(self.params["bottom_threshold"]))*self.params["scale"]
                l += self.params["screen"]["cartwidth"] / 2

                r = (end_spring_x + abs(self.params["bottom_threshold"]))*self.params["scale"]
                r -= self.params["screen"]["cartwidth"] / 2

                t = self.params["screen"]["carty"] + self.params["screen"]["springwidth"] / 2
                b = self.params["screen"]["carty"] - self.params["screen"]["springwidth"] / 2
                self.springs[i].v = [(l, b), (l, t), (r, t), (r, b)]

                # set spring color
                stretch = abs(spring_length - self.params["coupled"]["resting_dist"])
                red = 0.5 + min(0.5, stretch / self.params["coupled"]["resting_dist"])
                self.springs[i].set_color(red, 0.1, 0.1)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
