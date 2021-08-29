from functools import partial
from envs.multi_cart.multi_cart_env import MultiCartPoleEnv
from envs.access_point.access_point_env import AccessPointEnv
from envs.multi_particle.multi_particle_env import MultiParticleEnv
from envs.payoff_matrix.payoff_matrix import MatrixEnv

# If we are using Traffic Environments
# from envs.traffic.traffic_env import TrafficEnv

# If we are using StarCraft Environments
# from smac.env import MultiAgentEnv, StarCraft2Env

# Otherwise
from envs.multiagentenv import MultiAgentEnv

import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}

REGISTRY["multi_cart"] = partial(env_fn, env=MultiCartPoleEnv)
REGISTRY["access_point"] = partial(env_fn, env=AccessPointEnv)
REGISTRY["multi_particle"] = partial(env_fn, env=MultiParticleEnv)
REGISTRY["payoff_matrix"] = partial(env_fn, env=MatrixEnv)

# Only use if we have starcraft installed
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

# Only use if we have traffic SUMO installed
# REGISTRY["traffic"] = partial(env_fn, env=TrafficEnv)

# I think it doesnt go in here
# if sys.platform == "linux" or True:
#     os.environ.setdefault("SC2PATH",
#                           os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
