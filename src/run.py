import datetime
import os
import pprint
import time
import threading
import torch as th
import numpy as np

from types import SimpleNamespace as SN

from matplotlib import pyplot as plt

from reward_decomposition import decompose_viz
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
from torch.optim import RMSprop
from torch.optim import Adam

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

import reward_decomposition.decompose as decompose
from reward_decomposition.decomposer import RewardDecomposer




def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)

    # configure which device, attempting to use freest gpu available
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    # print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):
    args.env_args["learner_name"] = args.name
    args.env_args["exp_logger"] = args.exp_logger

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()

    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.obs_shape = env_info["obs_shape"]
    args.state_shape = env_info["state_shape"]
    args.graph_obj = env_info["graph_obj"]

    # Support Local Rewards also
    reward_shape = (1,)
    try:
        reward_shape = env_info["reward_shape"]
    except Exception as e:
        print("WARNING: Reward shape not specified in Enviroment, Assuming global reward")

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": reward_shape},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Setup Reward decomposition
    if not hasattr(args, "decompose_reward"):
        args.decompose_reward = False
    if not hasattr(args, "reward_parameter_sharing"):
        args.reward_parameter_sharing = True
    if not hasattr(args, "reward_batch_size"):
        args.reward_batch_size = 10
    if not hasattr(args, "reward_updates_per_batch"):
        args.reward_updates_per_batch = 10
    if not hasattr(args, "reward_diff_threshold"):
        args.reward_diff_threshold = 0.2
    if not hasattr(args, "reward_acc"):
        args.reward_acc = 0.95

    args.reward_decomposer = RewardDecomposer(buffer.scheme, args) if args.decompose_reward else None
    # args.reward_optimiser = RMSprop(params=args.reward_decomposer.parameters(), lr=0.005, alpha=args.optim_alpha,
    #                                 eps=args.optim_eps) if args.decompose_reward else None
    args.reward_optimiser = Adam(params=args.reward_decomposer.parameters(),
                                 lr=0.01) if args.decompose_reward else None

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    last_viz_reward_t = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:
        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        # First train the reward decomposer if necessary
        if args.decompose_reward and buffer.can_sample(args.reward_batch_size):
            for reward_update_idx in range(args.reward_updates_per_batch):
                reward_sample = buffer.sample(args.reward_batch_size)
                reward_sample.to(args.device)
                decompose.train_decomposer(args.reward_decomposer, reward_sample, args.reward_optimiser)

        if (
                getattr(args, "viz_reward_decomposition", False)
                and (runner.t_env - last_viz_reward_t) >= getattr(args, "reward_viz_interval", 5000)
                and buffer.can_sample(args.reward_batch_size)
        ):
            # Visualize the reward models
            reward_sample = buffer.sample(args.reward_batch_size)
            decompose_viz.visualize_decomposer(args.reward_decomposer, reward_sample, env_name=args.env)
            last_viz_reward_t = runner.t_env

            # Save models to default directory
            args.reward_decomposer.save_models()

        # Next Train the learner
        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            # logger.console_logger.info("Starting Training")
            learner.train(episode_sample, runner.t_env, episode)
            # logger.console_logger.info("Ending Training")

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            logger.console_logger.info("")
            logger.console_logger.info("")
            logger.console_logger.info("Starting Test")
            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)
            logger.console_logger.info("Ending Test")
            logger.console_logger.info("")
            logger.console_logger.info("")

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config
