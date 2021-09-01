import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
import itertools

import envs.multi_cart.constants as constants

from reward_decomposition.decomposer import *

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


def train_decomposer(decomposer, batch, reward_optimizer):
    # organize the data
    reward_inputs, global_rewards, mask, _ = build_reward_data(batch)
    raw_outputs = decomposer.forward(reward_inputs)

    reward_pred = decomposer.convert_raw_outputs(raw_outputs, output_type=PRED)
    local_rewards = decomposer.convert_raw_outputs(raw_outputs, output_type=LOCAL_REWARDS)

    if decomposer.args.assume_binary_reward:
        # First, create to global probabilities
        reward_pred = th.log(reward_pred + 1e-8)
        reward_pred = th.reshape(reward_pred, shape=(-1, reward_pred.shape[-1]))

        # Next ready the global targets
        global_rewards = global_rewards.long()
        global_rewards = global_rewards.flatten()

        loss = nn.NLLLoss()
        output = loss(reward_pred, global_rewards)
    else:
        # compute the loss
        output = compute_loss(local_rewards, global_rewards, mask)

    print(output)

    # Now add the regularizing loss
    output += decomposer.compute_regularization(raw_outputs)

    reward_optimizer.zero_grad()
    output.backward()
    reward_optimizer.step()

    # visualize_decomposer_1d(decomposer, batch)
    # visualize_data(reward_inputs, global_rewards)


# decomposes the global rewards into local rewards
# Returns (status, reward_mask, local_rewards) where
# status - refers to the total reliability of the local rewards
# reward_mask - if status is True, which rewards should be used
def decompose(decomposer, batch):
    # decompose the reward
    reward_inputs, global_rewards, mask, _ = build_reward_data(batch, include_last=False)
    raw_outputs = decomposer.forward(reward_inputs)

    agent_rewards = decomposer.convert_raw_outputs(raw_outputs, output_type=AGENT_REWARDS)

    if decomposer.args.assume_binary_reward:
        global_rewards = global_rewards.long()

    # now we assume that the local rewards
    status, reward_mask = build_reward_mask(decomposer, agent_rewards, global_rewards, mask)

    # return the rewards and the status. Just in case, return None as local rewards to make sure they arent used even
    # if status is False
    if not status:
        agent_rewards = None
    return status, reward_mask, agent_rewards


# Function recvs the true global reward and the outputs generated by the decomposer, and checks how similar they are
# This will determine if the QLearner should use each modified sample
def build_reward_mask(decomposer, local_rewards, global_rewards, mask):
    diff = compute_diff(local_rewards, global_rewards, mask)

    # Determine the reward mask and the status
    reward_mask = th.where(th.logical_and(mask, (th.abs(diff) < decomposer.args.reward_diff_threshold)), 1., 0.)
    reward_decomposition_acc = th.sum(reward_mask) / th.sum(mask)
    status = reward_decomposition_acc > decomposer.args.reward_acc

    # Visualize inverse histogram if necessary
    print(f"Decomposition Accuracy: {reward_decomposition_acc}")
    # visualize_diff(diff, mask, horiz_line=decomposer.args.reward_diff_threshold)

    return status, reward_mask


def build_reward_data(batch, include_last=True):
    # For now, define the input for the reward decomposition network as just the observations
    # note that some of these aren't relevant, so we additionally supply a mask for pairs that shouldn't be learnt
    if include_last:
        inputs = batch["obs"][:, :, :, :]
        outputs = local_to_global(batch["reward"])
        truth = batch["reward"]
        mask = batch["filled"].float()
    else:
        inputs = batch["obs"][:, :-1, :, :]
        outputs = local_to_global(batch["reward"][:, :-1])
        truth = batch["reward"][:, :-1]
        mask = batch["filled"][:, :-1].float()

    return inputs, outputs, mask, truth


# huber loss
def huber(diff, delta=0.1):
    loss = th.where(th.abs(diff) < delta, 0.5 * (diff ** 2),
                    delta * th.abs(diff) - 0.5 * (delta ** 2))
    return th.sum(loss) / len(diff)


# log cosh loss
def logcosh(diff):
    loss = th.log(th.cosh(diff))
    return th.sum(loss) / len(diff)


def mse(diff):
    return th.sum(diff ** 2) / len(diff)


def mae(diff):
    return th.sum(th.abs(diff)) / len(diff)


def compute_loss(local_rewards, global_rewards, mask):
    diff = compute_diff(local_rewards, global_rewards, mask).flatten()
    loss = logcosh(diff)
    return loss


def compute_diff(local_rewards, global_rewards, mask):
    # reshape local rewards
    local_rewards = th.reshape(local_rewards, shape=(*local_rewards.shape[:2], -1))
    summed_local_rewards = local_to_global(local_rewards)
    global_rewards = th.reshape(global_rewards, summed_local_rewards.shape)
    return th.mul(th.subtract(summed_local_rewards, global_rewards), mask)


def visualize_decomposer_2d(decomposer, batch, env_name="multi_particle"):
    example_input = decomposer.build_input(batch, 0, 0, 0)

    if env_name == "multi_particle":
        example_input_method = create_example_inputs_multi_particle
    elif env_name == "multi_cart":
        example_input_method = create_example_inputs_multi_cart
    else:
        print("Can't visualize decomposer for this enviroment...")
        return

    xs1, example_inputs1 = example_input_method(example_input)
    xs = itertools.product(*[xs1, xs1])

    # TODO: check if this is correct for larger ones
    example_inputs = th.tensor(list(itertools.product(*[example_inputs1, example_inputs1])))

    # Reshape example_inputs so it looks like a batch
    example_inputs = th.reshape(example_inputs, shape=(1, *example_inputs.shape, 1))

    # Obtain local rewards for visualizations
    raw_outputs = decomposer.forward(example_inputs)
    local_rewards = decomposer.convert_raw_outputs(raw_outputs, output_type=LOCAL_REWARDS)

    total_ys = []
    for reward_idx in range(local_rewards.shape[-2]):
        ys = local_rewards[:, :, reward_idx].detach().numpy()
        ys = np.reshape(ys, (xs1.shape[0], xs1.shape[0]))
        total_ys.append(ys)

    total_ys.append(sum(total_ys))
    total_ys += [None] * (4 - len(total_ys))

    draw_multiple_2d([total_ys[:2], total_ys[2:]])


# For plotting purposes
def draw_multiple_2d(arr_list):
    plt.close("all")
    fig, ax = plt.subplots(ncols=2, nrows=2)

    for row_idx, row in enumerate(ax):
        for col_idx, col in enumerate(row):
            if arr_list[row_idx][col_idx] is None:
                continue
            im = col.imshow(arr_list[row_idx][col_idx], cmap='bone')
            divider = make_axes_locatable(col)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
    plt.draw()
    plt.pause(0.001)


# Returns the theta coordinate
def create_example_inputs_multi_cart(example_input):
    xs = np.arange(-np.pi / 4, np.pi / 4, 0.01)
    example_inputs = []
    for x_val in xs:
        temp_input = th.tensor(example_input)
        temp_input[2] = x_val
        example_inputs.append(temp_input)
    return xs, th.stack(example_inputs)


# Returns the theta coordinate
def create_example_inputs_multi_particle(example_input):
    xs = np.arange(0, 3, 0.1)
    example_inputs = []
    for x_val in xs:
        temp_input = th.tensor(example_input)
        temp_input[-1] = x_val
        # temp_input[-2] = x_val / np.sqrt(2)
        # temp_input[-3] = x_val / np.sqrt(2)
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

    xs, example_inputs = example_input_method(example_input)

    # Reshape example_inputs so it looks like a batch
    example_inputs = th.reshape(example_inputs, shape=(1, example_inputs.shape[0], 1, example_inputs.shape[1]))

    # Duplicate along agent axis so forward knows how to eat this
    example_inputs = example_inputs.repeat(1, 1, decomposer.n_agents, 1)

    # Obtain local rewards for visualizations
    raw_outputs = decomposer.forward(example_inputs)
    agent_rewards = decomposer.convert_raw_outputs(raw_outputs, output_type=AGENT_REWARDS)

    ys = [agent_reward.detach().numpy() for agent_reward in agent_rewards[0, :, 0]]

    plt.close("all")
    visualize_batch_1d(batch, env_name)
    draw_1d_updating(xs, ys)


def almost_flatten(arr):
    return arr.reshape(-1, arr.shape[-1])


def local_to_global(arr):
    return th.sum(arr, dim=-1, keepdims=True)


# Returns the theta coordinate
def create_multi_cart_viz_input(reward_inputs):
    return reward_inputs[:, :, :, 2]


# # Returns the distance from the first landmark, which is the distance from the point in coordinates (4-5)
# def create_multi_particle_viz_input(reward_inputs):
#     return th.sqrt(th.sum(th.square(reward_inputs[:, :, :, 4:6]), dim=-1))

# Returns the distance from the first landmark, which is the distance from the point in coordinates (4-5)
def create_multi_particle_viz_input(reward_inputs):
    return reward_inputs[:, :, :, -1]


def visualize_batch_1d(batch, env_name="multi_particle"):
    reward_inputs, global_rewards, mask, real_local_rewards = build_reward_data(batch, include_last=False)

    # Get the x coordinate for the visualization
    if env_name == "multi_particle":
        reward_inputs = create_multi_particle_viz_input(reward_inputs)

        plt.axvline(x=0.3)
    elif env_name == "multi_cart":
        reward_inputs = create_multi_cart_viz_input(reward_inputs)

        plt.axvline(x=-constants.THETA_THRESHOLD_RADIANS)
        plt.axvline(x=constants.THETA_THRESHOLD_RADIANS)
    else:
        print("Can't visualize batch for this enviroment...")
        return

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
