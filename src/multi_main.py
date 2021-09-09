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


def make_command(test_num, iteration_num, run_num, platform):
    return f"sh scripts/single_{platform}.sh {test_num} {iteration_num} {run_num}"


def main():
    # First try to see what test we're dealing with
    params = deepcopy(sys.argv)
    test_num = get_param(params, "--test-num")
    iteration_num = get_param(params, "--iteration-num")
    platform = get_param(params, "--platform")

    # If test num not specified, raise an error so we don't have any problems
    if test_num is None:
        raise Exception("Please specify a test_num")

    # If platform is not specified, assume it's on the server machine
    if platform is None:
        print("Platform is not specified, assuming running on a server")
        platform = "server"

    # Now try to read the test and see if it's valid
    test_config = get_config_dict(f"test{test_num}", "tests")
    if test_config is None:
        raise Exception("Invalid test_num, exiting...")

    # Create command for every testnum-iterationnum-runnum. If iteration num is not specified, run all of them
    if iteration_num is None:
        commands = [
            make_command(test_num, curr_iteration_num, run_num, platform)
            for run_num in range(test_config["num_runs"])
            for curr_iteration_num in range(test_config.get("num_iterations", 1))
        ]
    else:
        commands = [
            make_command(test_num, iteration_num, run_num, platform)
            for run_num in range(test_config["num_runs"])
        ]

    print(commands)

    procs = []
    for i in commands:
        print(f"Running:\t{i}")
        procs.append(Popen(i, shell=True))
        time.sleep(5)

    for p in procs:
        p.wait()


# This is a module who's goal is to run multiple test in one run
if __name__ == '__main__':
    main()
