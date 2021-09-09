import random

import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
from utils.logging import Logger
import yaml
import threading

from components.exp_logger import ExperimentLogger
from types import SimpleNamespace as SN

import run

from main_util import *

SETTINGS['CAPTURE_MODE'] = "no"  # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


def run_name(env_name, alg_name, test_num, iteration_num, run_num):
    name = f"{test_num}-" if test_num is not None else ""
    name += f"{iteration_num}-" if iteration_num is not None else ""
    name += f"{run_num}-" if run_num is not None else ""
    name += f"{alg_name}-{env_name}"
    return name


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return int(np.argmax(memory_available))


def single_run(env_name, alg_name, seed, override_config=None, test_num=None, iteration_num=None, run_num=None):
    # Load algorithm and env base configs
    default_config = get_config_dict("default")
    env_config = get_config_dict(env_name, "envs")
    alg_config = get_config_dict(alg_name, "algs")
    if override_config is None:
        override_config = dict()

    # Load my Experiment Logger object for personal testing
    config_dict = default_config.copy()
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    config_dict = recursive_dict_update(config_dict, override_config)

    # Create local Experiment logger
    exp_logger = ExperimentLogger(
        env_name=env_config["env"],
        exp_name=None,
    )
    exp_logger.add_learner(config_dict["name"])
    exp_logger.log_config(config_dict)
    config_dict["exp_logger"] = exp_logger

    # Setting the random seed throughout the modules
    np.random.seed(seed)
    th.manual_seed(seed)
    random.seed(seed)
    config_dict["seed"] = seed

    config_dict = run.args_sanity_check(config_dict, logger)

    # Set device for this experiment
    config_dict["device"] = "cpu"
    if config_dict["use_cuda"]:
        try:
            free_gpu_id = get_freer_gpu()
            config_dict["device"] = f"cuda:{free_gpu_id}"
            th.cuda.set_device(free_gpu_id)
        except Exception as e:
            print(f"Resorting to default cuda device, {e}")
            config_dict["device"] = "cuda"
    print(f"Running the test on device: {config_dict['device']}")

    # Setup logger and wandb. Modify current run name
    logger_obj = Logger(logger)
    curr_run_name = run_name(env_name, alg_name, test_num, iteration_num, run_num)
    if not config_dict["human_mode"]:
        curr_run_name = logger_obj.setup_wandb(config=config_dict, run_name=curr_run_name)
    config_dict["run_name"] = curr_run_name

    # Run the current test
    run.run_sequential(args=SN(**config_dict), logger=logger_obj)

    # Kill eveything
    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    # Setup WANDB for new run
    logger_obj.new_run_wandb()


def main():
    params = deepcopy(sys.argv)

    env_name = get_param(params, "--env-name")
    alg_name = get_param(params, "--alg-name")

    # Seed is randomly generated for a single run like this
    seed = random.randrange(2 ** 32 - 1)

    single_run(env_name, alg_name, seed)


if __name__ == '__main__':
    main()
