import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
import itertools

import envs.multi_cart.constants as constants

from reward_decomposition.decomposer import *
from reward_decomposition.decompose import *

from mpl_toolkits.axes_grid1 import make_axes_locatable

import time

# wierd bug where matplotlib spits out a ton of debug messages for no apparent reason
import logging

logging.getLogger('matplotlib').setLevel(logging.ERROR)


# Shape Conventions:

# Observations: [batch_size, ep_length, num_agents, obs_length]
# Raw Outputs (Regression): List of List of [batch_size, ep_length, 1]
# Raw Outputs (Regression): List of List of [batch_size, ep_length, num_classes (for that func)]
# Flat Raw Outputs: Same as Raw outputs, only with a flattened list
# Pred (Regression): [batch_size, ep_length, 1]
# Pred (Classification): [batch_size, ep_length, num_global_classes]
# Classes (Only Classification): [batch_size, ep_length, num_reward_functions]
# Local Rewards (Both): [batch_size, ep_length, num_reward_functions, 1]
# Agent Rewards (Both): [batch_size, ep_length, num_agents, 1]
# Global Rewards: [batch_size, ep_length, 1]


def visualize_decomposer(decomposer, batch, env_name="multi_particle"):
    if env_name == "multi_particle":
        visualize_decomposer_2d(decomposer, batch, env_name)
    elif env_name == "multi_cart":
        visualize_decomposer_1d(decomposer, batch, env_name)
    else:
        print("Can't visualize decomposer for this enviroment...")
        return


def visualize_decomposer_2d(decomposer, batch, env_name="multi_particle"):
    example_input = decomposer.build_input(batch, 0, 0, 0)

    if env_name == "multi_particle":
        example_input_method = create_example_inputs_multi_particle
    elif env_name == "multi_cart":
        example_input_method = create_example_inputs_multi_cart
    else:
        print("Can't visualize decomposer for this enviroment...")
        return

    if decomposer.n_agents != 2:
        print("Can only visualize 2d decomposer for 2 agent case")
        return

    xs1, example_inputs1 = example_input_method(decomposer, example_input)
    # xs = itertools.product(*[xs1, xs1])

    example_inputs = list(itertools.product(*[example_inputs1, example_inputs1]))
    example_inputs = [th.stack(list(pair)) for pair in example_inputs]
    example_inputs = th.stack(example_inputs)

    # Reshape example_inputs so it looks like a batch
    example_inputs = th.reshape(example_inputs, shape=(1, *example_inputs.shape))

    # Obtain local rewards for visualizations
    raw_outputs = decomposer.forward(example_inputs)
    local_rewards = decomposer.convert_raw_outputs(raw_outputs, output_type=LOCAL_REWARDS)

    total_ys = []
    for reward_idx in range(local_rewards.shape[-2]):
        ys = local_rewards[:, :, reward_idx].detach().numpy()
        ys = np.reshape(ys, (xs1.shape[0], xs1.shape[0]))
        total_ys.append(ys)

    all_ys = total_ys + [None] * (3 - len(total_ys))
    all_ys.append(sum(total_ys))

    draw_multiple_2d([all_ys])


# For plotting purposes
def draw_multiple_2d(arr_list):
    plt.close("all")
    fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(11, 3))
    fig.tight_layout(pad=3.0)

    titles = [r"$r_1(s_1, a_1)$", r"$r_2(s_2, a_2)$", r"$r_{ \{1\cup2\} }(s, a)$", r"$r(s,a)$"]

    vmin = min(-1, min([np.min(arr) for arr in arr_list[0] if arr is not None]))
    vmax = max(2, max([np.max(arr) for arr in arr_list[0] if arr is not None]))

    for col_idx, col in enumerate(ax):
        plt.setp(col.get_xticklabels(), visible=False)
        plt.setp(col.get_yticklabels(), visible=False)
        col.tick_params(axis=u'both', which=u'both', length=0)

        if arr_list[0][col_idx] is None:
            col.axis("off")
            continue

        im = col.imshow(arr_list[0][col_idx], cmap='bone', vmin=vmin, vmax=vmax, origin="lower")
        divider = make_axes_locatable(col)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        col.set_xlabel(r"Agent 2 $\Delta x$")
        col.set_ylabel(r"Agent 1 $\Delta x$")
        col.set_title(titles[col_idx])

    plt.draw()
    plt.savefig("src/reward_decomposition/figs/decomposition.png")
    plt.pause(0.001)


# Returns the theta coordinate
def create_example_inputs_multi_cart(decomposer, example_input):
    xs = np.arange(-np.pi / 4, np.pi / 4, 0.01)
    example_inputs = []
    for x_val in xs:
        if decomposer.args.reward_index_in_obs == -1:
            temp_input = th.tensor(example_input)
            temp_input[2] = x_val
        else:
            temp_input = th.tensor(example_input[:1])
            temp_input[0] = x_val

        example_inputs.append(temp_input)
    return xs, th.stack(example_inputs)


# Returns the theta coordinate
def create_example_inputs_multi_particle(decomposer, example_input):
    xs = np.arange(0, 1.5, 0.1)
    example_inputs = []
    for x_val in xs:
        if decomposer.args.reward_index_in_obs == -1:
            temp_input = th.tensor(example_input)
            temp_input[-1] = x_val
            temp_input[-2] = x_val / np.sqrt(2)
            temp_input[-3] = x_val / np.sqrt(2)
        else:
            temp_input = th.tensor(example_input[:1])
            temp_input[0] = x_val
        example_inputs.append(temp_input)
    return xs, example_inputs


def visualize_decomposer_1d(decomposer, batch, env_name="multi_particle"):
    example_input = decomposer.build_input(batch, 0, 0, 0)

    if env_name == "multi_particle":
        example_input_method = create_example_inputs_multi_particle
    elif env_name == "multi_cart":
        example_input_method = create_example_inputs_multi_cart
    else:
        print("Can't visualize decomposer for this enviroment...")
        return

    xs, example_inputs = example_input_method(decomposer, example_input)

    # Reshape example_inputs so it looks like a batch, duplicate along agent axis so forward knows how to eat this
    example_inputs = th.reshape(example_inputs, shape=(1, example_inputs.shape[0], 1, example_inputs.shape[1]))
    example_inputs = example_inputs.repeat(1, 1, decomposer.n_agents, 1)

    # Obtain local rewards for visualizations
    raw_outputs = decomposer.forward(example_inputs)
    agent_rewards = decomposer.convert_raw_outputs(raw_outputs, output_type=AGENT_REWARDS)

    ys = [agent_reward.detach().numpy() for agent_reward in agent_rewards[0, :, 0]]

    plt.close("all")
    visualize_batch_1d(decomposer, batch, env_name)
    draw_1d_updating(xs, ys)


# Returns the theta coordinate
def create_multi_cart_viz_input(reward_inputs):
    return reward_inputs[:, :, :, 2]


# # Returns the distance from the first landmark, which is the distance from the point in coordinates (4-5)
# def create_multi_particle_viz_input(reward_inputs):
#     return th.sqrt(th.sum(th.square(reward_inputs[:, :, :, 4:6]), dim=-1))

# Returns the distance from the first landmark, which is the distance from the point in coordinates (4-5)
def create_multi_particle_viz_input(reward_inputs):
    return reward_inputs[:, :, :, -1]


def visualize_batch_1d(decomposer, batch, env_name="multi_particle"):
    reward_inputs, global_rewards, mask, real_local_rewards = build_reward_data(decomposer, batch, include_last=False)

    # Get the x coordinate for the visualization
    if decomposer.args.reward_index_in_obs != -1:
        reward_inputs = reward_inputs[:, :, :, 0]
    else:
        if env_name == "multi_particle":
            reward_inputs = create_multi_particle_viz_input(reward_inputs)
        elif env_name == "multi_cart":
            reward_inputs = create_multi_cart_viz_input(reward_inputs)
        else:
            print("Can't visualize batch for this enviroment...")
            return

    if env_name == "multi_particle":
        plt.axvline(x=0.3)
    elif env_name == "multi_cart":
        plt.axvline(x=-constants.THETA_THRESHOLD_RADIANS)
        plt.axvline(x=constants.THETA_THRESHOLD_RADIANS)

    global_rewards = global_rewards.expand_as(reward_inputs)
    mask = mask.expand_as(reward_inputs)

    # reshape for visualization
    reward_inputs = reward_inputs.flatten()
    global_rewards = global_rewards.flatten()
    mask = mask.flatten()
    real_local_rewards = real_local_rewards.flatten()

    xs = []
    ys = []

    for obs_idx in range(len(reward_inputs)):
        # We include the step that led to the terminated state, but now exit
        # TODO: make sure this still happens with new build data
        if mask[obs_idx]:
            xs.append(reward_inputs[obs_idx])
            ys.append(real_local_rewards[obs_idx])

    draw_1d_updating(xs, ys)


def visualize_diff(diff, mask, horiz_line=0.2):
    diff = diff.flatten()
    mask = mask.flatten()

    ys = []

    for diff_idx in range(len(diff)):
        if mask[diff_idx]:
            ys.append(diff[diff_idx].detach().numpy())

    plt.clf()
    plt.ylim()
    plt.axhline(y=horiz_line)
    plt.plot(np.sort(np.abs(ys)))
    plt.show()


def draw_1d_updating(xs, ys):
    plt.scatter(xs, ys)
    plt.ylim((-0.2, 1.2))
    plt.draw()
    plt.pause(0.001)
