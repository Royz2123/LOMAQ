import yaml
import time
import sys
import collections

import os
import subprocess

from multiprocessing import Process

from main_util import *
from main import single_run

TESTS_PATH = os.path.join(os.path.dirname(__file__), "config", "tests")
SUPER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "super_config.yaml")


def get_current_run_override_config(orig_dict, test_num, num_tests):
    # If this is still a dictionary, we need to go deeper for each new argument
    if type(orig_dict) == dict:
        single_dict = {}
        for k, v in orig_dict.items():
            single_dict[k] = get_current_run_override_config(v, test_num, num_tests)
        return single_dict

    elif type(orig_dict) == list:
        # Assume [min, max] if ints
        if len(orig_dict) == 2 and type(orig_dict[0]) == int and type(orig_dict[1]) == int:
            delta = orig_dict[1] - orig_dict[0]
            return orig_dict[0] + delta * (test_num / (num_tests - 1))

        elif len(orig_dict) != num_tests:
            raise Exception(f"List length must be equal to num_tests or 2. Params: {orig_dict}, Required: {num_tests}")

        else:
            return orig_dict[test_num]

    # In this case, just assume constant value
    else:
        return orig_dict


def main():
    # First try to see what test we're dealing with
    params = deepcopy(sys.argv)
    test_num = get_param(params, "--test-num")

    # If test num not specified, raise an error so we don't have any problems
    if test_num is None:
        raise Exception("Please specify a test_num")

    # Now try to read the test and see if it's valid
    test_config = get_config_dict(f"test{test_num}", "tests")
    if test_config is None:
        raise Exception("Invalid test_num, exiting...")

    # Parse the test config, and run single_run that many times
    runs = []
    for run_num in range(test_config["num_runs"]):
        override_config = get_current_run_override_config(test_config["override"], run_num, test_config["num_runs"])
        env_name = get_current_run_override_config(test_config["env_name"], run_num, test_config["num_runs"])
        alg_name = get_current_run_override_config(test_config["alg_name"], run_num, test_config["num_runs"])

        # Run main
        run_process = Process(target=single_run, args=(env_name, alg_name, override_config))
        run_process.start()
        runs.append(run_process)

    for run in runs:
        run.join()

    # single_run(env_name, alg_name, override_config)


# This is a module who's goal is to run multiple test in one run
if __name__ == '__main__':
    main()
