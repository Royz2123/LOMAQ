import main as single_test
import yaml
import time
import sys
import collections

import os
import subprocess

MULTI_MAIN_PATH = os.path.join(os.path.dirname(__file__), "config", "global", "multi_main.yaml")
SUPER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "global", "super_config.yaml")

DEFAULT_PYTHON_NAME = "python37"
DEFAULT_MAIN_NAME = "src/main.py"
DEFAULT_PLOT_NAME = "src/plot.py"


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_current_test_data(orig_dict, test_num, num_tests):
    # If this is still a dictionary, we need to go deeper for each new argument
    if type(orig_dict) == dict:
        single_dict = {}
        for k, v in orig_dict.items():
            single_dict[k] = get_current_test_data(v, test_num, num_tests)
        return single_dict

    # In this case, just assume constant value
    elif type(orig_dict) == int:
        return orig_dict

    elif type(orig_dict) == list:
        # Assume [min, max]
        if len(orig_dict) == 2:
            delta = orig_dict[1] - orig_dict[0]
            return orig_dict[0] + delta * (test_num / (num_tests - 1))

        elif len(orig_dict) != num_tests:
            raise Exception(f"List length must be equal to num_tests or 2. Params: {orig_dict}, Required: {num_tests}")

        else:
            return orig_dict[test_num]

    else:
        raise Exception(f"Unsupported type in {MULTI_MAIN_PATH}")


def main():
    with open(MULTI_MAIN_PATH, "r") as f:
        try:
            multi_config = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    num_tests = multi_config["num_tests"]

    for test_num in range(num_tests):
        super_config = {}
        for config_type in ["env_config", "alg_config"]:
            if multi_config[config_type] is not None:
                curr_dict = get_current_test_data(multi_config[config_type], test_num, num_tests)
                super_config = recursive_dict_update(super_config, curr_dict)

        # Now we need to run the main with curr_config. Save under config global super_config
        with open(SUPER_CONFIG_PATH, 'w') as f:
            _ = yaml.dump(super_config, f)

        # Run main (will take args from the super config file)
        subprocess.call([DEFAULT_PYTHON_NAME, DEFAULT_MAIN_NAME] + sys.argv[1:])
        subprocess.call([DEFAULT_PYTHON_NAME, DEFAULT_PLOT_NAME] + sys.argv[1:])

        # Destroy the super config file for later runs
        os.remove(SUPER_CONFIG_PATH)


# This is a module who's goal is to run multiple test in one run
# The arguments are
if __name__ == '__main__':
    main()
