import torch.nn as nn
import torch as th
import torch.nn.functional as F

import numpy as np
from itertools import combinations
from itertools import product

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# We define here some terms before we get started
# RewardNetwork - A single reward network that represents a certain n_j
# RewardGroup - A group of reward networks that represent all n_j of size l
# RewardGroups - All of the reward networks

# Possible output types (mostly when using the classification scheme)
(
    RAW,
    FLAT_RAW,
    PRED,
    CLASSES,
    LOCAL_REWARDS,
    AGENT_REWARDS
) = range(6)


class RewardDecomposer:
    DEFAULT_DIR = "src/reward_decomposition/saved_models"
    ARCHIVE_DIR = "src/reward_decomposition/saved_models/archive"

    def __init__(self, scheme, args, try_load=True):
        self.n_agents = args.n_agents
        self.args = args
        self.input_shape_one_obs = self._get_input_shape(scheme)

        # We will now build all the reward networks. Currently done with parameter sharing between networks of the same
        # input size. Our reward networks is a module_list of size l where index i marks the reward functions of size
        # i + 1
        self.reward_groups_indices = args.graph_obj.find_reward_groups(l=args.reward_l, beta2=args.reward_beta2)
        self.reward_groups = self.build_reward_groups()
        self.reward_networks = nn.ModuleList([reward_group.reward_network for reward_group in self.reward_groups])

        self.reward_combos = self.get_all_combinations(filter_combos=True)
        self.regularizing_weights = self.get_regularizing_weights()

        # try loading pretrained reward decomposition model
        if try_load:
            path = self.create_path(path=RewardDecomposer.ARCHIVE_DIR)
            print(f"Trying to load reward decomposition model from {path}")
            try:
                self.load_models(path=path)
            except Exception as e:
                print("Loading model from archive failed... training from scratch")
                print(e)

    # Let the reward function observe the state and the last action
    def _get_input_shape(self, scheme):
        # observe the last state
        input_shape = scheme["obs"]["vshape"]

        # # observe the last action
        # input_shape += scheme["actions_onehot"]["vshape"][0]
        #
        # if self.args.obs_agent_id:
        #     input_shape += self.n_agents

        if self.args.reward_index_in_obs != -1:
            input_shape = 1

        return input_shape

    def build_team_inputs(self, batch, ep_idx, t_idx):
        team_inputs = list()
        for agent_idx in range(self.n_agents):
            team_inputs.append(self.build_input(batch, ep_idx, t_idx, agent_idx))
        return th.stack(team_inputs)

    def build_input(self, batch, ep_idx, t_idx, agent_idx):
        agent_input = list()

        # observe the last state
        agent_input.append(batch["obs"][ep_idx, t_idx, agent_idx])
        # agent_input.append(batch["obs"][ep_idx, t_idx, agent_idx][-1:])

        # observe the last action
        # agent_input.append(batch["actions_onehot"][ep_idx, t_idx, agent_idx])

        # if self.args.obs_agent_id:
        #     agent_input.append(F.one_hot(th.tensor([agent_idx])[0], num_classes=self.n_agents))

        agent_input = th.cat(agent_input)
        return agent_input

    def build_reward_groups(self):
        reward_groups = []
        for idx, indices_groups in enumerate(self.reward_groups_indices):
            reward_groups.append(RewardGroup(
                self.args,
                indices_groups,
                self.input_shape_one_obs,
                idx + 1,
            ))
        return reward_groups

    def single_forward(self, reward_input, reward_group_idx=0):
        return self.reward_groups[reward_group_idx].single_forward(reward_input)

    def forward(self, reward_inputs):
        # Get the reward output for every reward group based on the output type
        reward_group_outputs = []
        for reward_group in self.reward_groups:
            reward_group_outputs.append(reward_group.forward(reward_inputs))
        return reward_group_outputs

    def convert_raw_outputs(self, raw_outputs, output_type=AGENT_REWARDS):
        # Note that local_rewards and raw are the same option for the regression scheme, therefore the type in that case
        # can be local rewards
        if not self.args.assume_binary_reward and output_type == RAW:
            output_type = LOCAL_REWARDS

        converted_outputs = [
            self.reward_groups[idx].convert_raw_outputs(raw_output, output_type=output_type)
            for idx, raw_output in enumerate(raw_outputs)
        ]
        # Flatten the outputs nicely if we're in raw or pred
        if output_type in [FLAT_RAW, PRED]:
            converted_outputs = [
                indicies_output for reward_group_output in converted_outputs
                for indicies_output in reward_group_output
            ]
        if output_type == PRED:
            if not self.args.assume_binary_reward:
                converted_outputs = th.stack(converted_outputs, dim=2)
                converted_outputs = th.sum(converted_outputs, dim=2)
            else:
                converted_outputs = self.raw_to_pred(converted_outputs)
        elif output_type in [CLASSES, LOCAL_REWARDS]:
            converted_outputs = th.cat(converted_outputs, dim=2)
        elif output_type == AGENT_REWARDS:
            converted_outputs = sum(converted_outputs)
        return converted_outputs

    def get_all_combinations(self, filter_combos=True):
        # This functions returns a list of pairs, where the first element is the indices that we should take in the
        # output, and the second element is the sum that they represent (global class). We will provide an options off
        # limiting all the global classes to be between 0 and n called "filter combos

        # Start by making a list of all the options
        reward_classes = []
        reward_values = []
        for idx, reward_group in enumerate(self.reward_groups):
            reward_classes += [list(range(len(reward_group.output_vals)))] * len(reward_group.indices_groups)
            reward_values += [reward_group.output_vals] * len(reward_group.indices_groups)

        # Now, we compute all of these combinations by doing the product of all these lists
        # WARNING: This list is huge. exponentially large with the number of agents
        reward_combos_classes = list(product(*reward_classes))
        reward_combos_values = list(product(*reward_values))

        # Now compute the sum of every combo
        output = [
            (np.array(classes), sum(values)) for classes, values in
            zip(reward_combos_classes, reward_combos_values)
        ]

        # Filter if necessary
        if filter_combos:
            output = [pair for pair in output if (0 <= pair[1] <= self.n_agents and int(pair[1]) == float(pair[1]))]
            output = [(pair[0], int(pair[1])) for pair in output]

        return output

    def raw_to_pred(self, local_probs):
        if not len(local_probs):
            return local_probs

        num_classes = self.n_agents + 1
        global_prob_class_shape = local_probs[0].shape[:2]
        global_probs = th.zeros(*global_prob_class_shape, num_classes)

        for indices_list, class_num in self.reward_combos:
            # Compute probability for this indices group
            curr_probs = th.ones(*global_prob_class_shape)
            for reward_func_idx, class_idx in enumerate(indices_list):
                curr_probs *= local_probs[reward_func_idx][:, :, class_idx]

            # Add this combination to the total probability
            global_probs[:, :, class_num] += curr_probs

        return global_probs

    def get_regularizing_weights(self):
        regularizing_weights = []
        for idx, reward_group in enumerate(self.reward_groups):
            if idx == 0:
                regularizing_weights.append(0)
            else:
                regularizing_weights.append(self.args.regularizing_weight)
        return regularizing_weights

    def compute_regularization(self, raw_outputs):
        reg_loss = 0
        for idx, raw_output in enumerate(raw_outputs):
            reg_loss += self.regularizing_weights[idx] * self.reward_groups[idx].compute_regularization(raw_output)
        return reg_loss

    def parameters(self):
        return self.reward_networks.parameters()

    def load_state(self, other_mac):
        self.reward_networks.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.reward_networks.cuda()

    def create_path(self, path=None):
        if path is None:
            path = RewardDecomposer.DEFAULT_DIR
        return f"{path}/reward_decomposition_{self.n_agents}_agents.th"

    def save_models(self, path=None):
        if path is None:
            path = self.create_path()
        th.save(self.reward_networks.state_dict(), path)

    def load_models(self, path=None):
        if path is None:
            path = self.create_path()
        self.reward_networks.load_state_dict(th.load(path, map_location=lambda storage, loc: storage))


# This class represents all of the NN that are of the same size n_j
class RewardGroup(nn.Module):
    def __init__(self, args, indices_groups, input_shape_one_obs, num_reward_agents):
        super(RewardGroup, self).__init__()

        self.args = args
        self.indices_groups = indices_groups

        self.input_shape_one_obs = input_shape_one_obs
        self.num_reward_agents = num_reward_agents
        self.contrib_weight = 1 / self.num_reward_agents

        self.input_shape = self.input_shape_one_obs * num_reward_agents
        self.output_shape, self.output_vals = self.compute_output_scheme()

        self.reward_network = self.build_reward_network().to(self.args.device)

    # If the reward network is a classifier, then outputs like classes otherwise 1
    def compute_output_scheme(self):
        output_vals = None
        output_shape = 1

        if self.args.reward_parameter_sharing:
            # If reward is an integer that's either 0 or 1, then we know that for a pair of agents, their
            # reward together could be either -1, 0 or 1 for every agent. This means the for agent group
            # of size k, the output shape will be 2*k + 1, except for the first one.
            # For the sake of convenience, we allow single agent reward functions to be -1 as well
            if self.num_reward_agents == 1:
                if self.args.env == "multi_particle":
                    output_vals = [0, 0.5, 1]
                elif self.args.env == "multi_cart":
                    output_vals = [0, 1]
                else:
                    raise Exception("Unsupported env name")
            else:
                output_vals = list(range(-self.num_reward_agents, self.num_reward_agents + 1))
                # output_vals = [-1, -0.5, 0, 0.5, 1]
            output_shape = len(output_vals)
        return output_shape, output_vals

    def build_reward_network(self):
        if len(self.indices_groups) == 0:
            return None

        if self.args.reward_parameter_sharing:
            if not self.args.assume_binary_reward:
                module_sub_list = RewardNetwork(self.args, self.input_shape)
            else:
                module_sub_list = RewardClassificationNetwork(self.args, self.input_shape, self.output_vals)
        else:
            raise Exception("Currently not supporting no parameter sharing for the reward network, exiting...")
        return module_sub_list

    def single_forward(self, reward_input):
        return self.reward_network(reward_input)

    def forward(self, reward_inputs):
        # Recieves reward inputs of shape [num_batches, ep_length, team_size, input_size]
        # Runs through the reward network for every indices_group in indices_groups
        # Returns a tensors of shape [num_batches, ep_length, indices_group_length]
        reward_outputs = []
        for indices_group in self.indices_groups:
            # get the reward inputs of every agent in indices group
            reward_input = reward_inputs[:, :, indices_group]

            # reshape the reward input by concatenating agents inputs
            reward_input = th.reshape(reward_input, shape=(*reward_input.shape[:-2], -1))

            # get outputs from the network for this combination
            reward_output = self.reward_network(reward_input)
            reward_outputs.append(reward_output)
        return reward_outputs

    def convert_raw_outputs(self, reward_outputs, output_type=AGENT_REWARDS):
        if output_type >= CLASSES:
            reward_outputs = th.stack(reward_outputs, dim=2)

        if not self.args.assume_binary_reward:
            if output_type == CLASSES:
                raise Exception("Regression networks cannot output classes")
        else:
            if output_type >= CLASSES:
                reward_outputs = RewardGroup.raw_to_classes(reward_outputs)
            if output_type >= LOCAL_REWARDS:
                reward_outputs = self.classes_to_local_rewards(reward_outputs)

        # For both classification and regression, we offer converting into agent_rewards
        if output_type == AGENT_REWARDS:
            reward_outputs = self.local_rewards_to_agent_rewards(reward_outputs)
        return reward_outputs

    @staticmethod
    def raw_to_classes(probs):
        return th.argmax(probs, dim=-1, keepdim=True)

    def classes_to_local_rewards(self, classes):
        # mask = th.tensor(np.array([min(self.output_vals)])).repeat(*classes.shape)
        return classes.apply_(lambda x: self.output_vals[x])

    def local_rewards_to_agent_rewards(self, local_rewards):
        agent_rewards = th.zeros(*local_rewards.shape[:2], self.args.n_agents).to(self.args.device)
        local_rewards = th.reshape(local_rewards, shape=(*local_rewards.shape[:2], -1))

        for idx, indices_group in enumerate(self.indices_groups):
            weighted_reward = self.contrib_weight * local_rewards[:, :, [idx]]
            weighted_reward = weighted_reward.repeat(1, 1, len(indices_group))
            agent_rewards[:, :, indices_group] += weighted_reward

        return agent_rewards

    def compute_regularization(self, raw_outputs):
        # Gets as input a list of outputs representing every rnj output
        # a list of length num_rnj functions with tensors of shape [num_batches, ep_len, output_shape]
        # Now add the regularizing loss
        reg_loss = 0

        for output in raw_outputs:
            # For regression this is fairly easy
            if not self.args.assume_binary_reward:
                reg_loss += th.sum(th.abs(output))
            # A little more difficult for classification, we want to punish every probability as they travel farther
            # 0. An ideal approach would be pi * val[i]
            else:
                values = th.square(th.tensor(np.array([self.output_vals])))
                values = th.reshape(values, shape=(1, 1, -1))
                values = values.repeat(*output.shape[:-1], 1)
                reg_loss += th.sum(output * values)
        return reg_loss


class RewardNetwork(nn.Module):
    def __init__(self, args, input_shape):
        super(RewardNetwork, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        y = F.leaky_relu(self.fc2(x))
        h = F.tanh(self.fc3(y))
        q = self.fc4(h)

        # reward is bounded between 0 and 1
        if getattr(self.args, "reward_clamp", False):
            q = th.clamp(q, min=0, max=1)
        return q


# This is a classifcation network for the binary reward
class RewardClassificationNetwork(nn.Module):
    def __init__(self, args, input_shape, output_vals):
        super(RewardClassificationNetwork, self).__init__()
        self.args = args
        self.output_vals = output_vals

        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, len(output_vals))

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        y = F.leaky_relu(self.fc2(x))
        h = F.tanh(self.fc3(y))
        q = F.softmax(self.fc4(h), dim=-1)
        return q
