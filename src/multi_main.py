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


def make_command(test_num, run_num, platform):
    return f"sh scripts/single_{platform}.sh {test_num} {run_num}"


def main():
    # First try to see what test we're dealing with
    params = deepcopy(sys.argv)
    test_num = get_param(params, "--test-num")
    platform = get_param(params, "--platform")

    # If test num not specified, raise an error so we don't have any problems
    if test_num is None:
        raise Exception("Please specify a test_num")

    # If platform is not specified, assume it's on the technion machine
    if platform is None:
        print("Platform is not specified, assuming default platform")
        platform = "technion"

    # Now try to read the test and see if it's valid
    test_config = get_config_dict(f"test{test_num}", "tests")
    if test_config is None:
        raise Exception("Invalid test_num, exiting...")

    # Parse the test config, and run single_run that many times
    num_iterations = test_config.get("num_iterations", default=1)
    test_names = [f"{test_num}-{iteration_num}" for iteration_num in range(num_iterations)]

    # Create command for every testnum-iterationnum-runnum
    commands = [
        make_command(test_name, run_num, platform)
        for run_num in range(test_config["num_runs"])
        for test_name in test_names
    ]

    procs = []
    for i in commands:
        print(f"Running:\t{i}")
        procs.append(Popen(i, shell=True))
        time.sleep(10)

    for p in procs:
        p.wait()


# This is a module who's goal is to run multiple test in one run
if __name__ == '__main__':
    main()
