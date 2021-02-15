from functools import partial
from envs.multi_cart.multi_cart_env import MultiCartPoleEnv

# If we are using StarCraft Environments
# from smac.env import MultiAgentEnv, StarCraft2Env

# Otherwise
from envs.multiagentenv import MultiAgentEnv

import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}

# Only use if we have starcraft installed
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

REGISTRY["multi_cart"] = partial(env_fn, env=MultiCartPoleEnv)

# I think it doesnt go in here
# if sys.platform == "linux" or True:
#     os.environ.setdefault("SC2PATH",
#                           os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
