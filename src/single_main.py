import yaml
import time
import sys
import collections

import os
import subprocess

from subprocess import Popen

from main_util import *
from main import single_run

TESTS_PATH = os.path.join(os.path.dirname(__file__), "config", "tests")
SUPER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "super_config.yaml")


def convert_to_int(s):
    try:
        return int(s)
    except ValueError:
        raise Exception("Please specify a valid test/run num")


def main():
    # First try to see what test we're dealing with
    params = deepcopy(sys.argv)
    test_num = get_param(params, "--test-num")
    run_num = get_param(params, "--run-num")

    test_num = convert_to_int(test_num)
    run_num = convert_to_int(run_num)

    # Now try to read the test and see if it's valid
    test_config = get_config_dict(f"test{test_num}", "tests")
    if test_config is None:
        raise Exception("Invalid test_num, exiting...")

    # Parse the test config, and run single_run that many times
    override_config = get_current_run_override_config(test_config["override"], run_num, test_config["num_runs"])
    env_name = get_current_run_override_config(test_config["env_name"], run_num, test_config["num_runs"])
    alg_name = get_current_run_override_config(test_config["alg_name"], run_num, test_config["num_runs"])

    single_run(env_name, alg_name, override_config, test_num=test_num, run_num=run_num)


# This is a module who's goal is to run multiple test in one run
if __name__ == '__main__':
    main()
