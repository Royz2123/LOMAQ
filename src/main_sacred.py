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
import yaml

from components.exp_logger import ExperimentLogger
from run import run

SETTINGS['CAPTURE_MODE'] = "no"  # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if "," in config_name:
        config_d = [get_config_dict(name, subfolder) for name in config_name.split(",")]
    else:
        config_d = [get_config_dict(config_name, subfolder)]

    return config_d


def get_config_dict(config_name, subfolder):
    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)),
                  "r") as f:
            try:
                config_d = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_d


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def main():
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            default_config = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")[0]
    alg_configs = _get_config(params, "--config", "algs")

    # Load my Experiment Logger object for personal testing
    exp_logger = ExperimentLogger(
        env_name=env_config["env"],
        exp_name=None,
        env_config=env_config,
        alg_configs=alg_configs,
    )
    default_config["exp_logger"] = exp_logger

    for alg_config in alg_configs:
        config_dict = default_config.copy()
        config_dict = recursive_dict_update(config_dict, env_config)
        config_dict = recursive_dict_update(config_dict, alg_config)

        # Here we want to see if there are any super configs that we need to update
        # Super configs will usually come from the multi_main.py where we change parameters in a single run
        # print(config_dict)
        try:
            with open(os.path.join(os.path.dirname(__file__), "config", "global", "super_config.yaml"), "r") as f:
                super_config = yaml.load(f)
                config_dict = recursive_dict_update(config_dict, super_config)

                # TODO: Save super_config in each experiment
        except Exception as e:
            print(f"Failed updating super config, might not exist which is OK {e}")

        exp_logger.add_learner(config_dict["name"])

        ex = Experiment("pymarl")
        ex.logger = logger
        ex.captured_out_filter = apply_backspaces_and_linefeeds

        @ex.main
        def my_main(_run, _config, _log):
            # Setting the random seed throughout the modules
            config = config_copy(_config)
            np.random.seed(config["seed"])
            th.manual_seed(config["seed"])
            config['env_args']['seed'] = config["seed"]

            # run the framework
            run(_run, config, _log)

        # now add all the config to sacred
        ex.add_config(config_dict)

        # Save to disk by default for sacred
        logger.info("Saving to FileStorageObserver in results/sacred.")
        file_obs_path = os.path.join(results_path, "sacred")

        ex.observers.append(FileStorageObserver(file_obs_path))

        ex.run_commandline(params)


if __name__ == '__main__':
    main()
