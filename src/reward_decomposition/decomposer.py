import torch.nn as nn
import torch as th
import torch.nn.functional as F

import numpy as np
from itertools import combinations
from itertools import product

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class RewardDecomposer:
    DEFAULT_DIR = "src/reward_decomposition/saved_models"
    ARCHIVE_DIR = "src/reward_decomposition/saved_models/archive"

    def __init__(self, scheme, args, try_load=True):
        self.n_agents = args.n_agents
        self.args = args

        # We will now build all the reward networks. Currently done with parameter sharing between networks of the same
        # input size. Our reward networks is a module_list of size l where index i marks the reward functions of size
        # i + 1
        input_shape_one_obs = self._get_input_shape(scheme)
        self.reward_groups = args.graph_obj.find_reward_groups(l=args.reward_l, beta2=args.reward_beta2)
        self.reward_networks = self.build_reward_networks(input_shape_one_obs)

        self.reward_combos, self.reward_to_class_idx = self.get_all_combinations(filter_combos=True)

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

        input_shape = 1

        return input_shape

    # def build_data_old(self, batch):
    #     inputs = []
    #     outputs = []
    #     for ep_idx in range(batch.batch_size):
    #         for t_idx in range(batch.max_seq_length):
    #             team_inputs = self.build_team_inputs(batch, ep_idx, t_idx)
    #             inputs.append(team_inputs)
    #
    #             if self.args.assume_binary_reward:
    #                 # We have effectively n+1 classes, all the values between [0, n] including
    #                 # outputs.append(F.one_hot(batch["reward"][ep_idx, t_idx].int(), num_classes=(self.n_agents + 1)))
    #                 outputs.append(batch["reward"][ep_idx, t_idx].int())
    #             else:
    #                 outputs.append(th.sum(batch["reward"][ep_idx, t_idx]))
    #
    #             # we include the step that led to the terminated state, but now exit
    #             if batch["terminated"][ep_idx, t_idx][0]:
    #                 break
    #
    #     inputs = th.stack(inputs)
    #     outputs = th.stack(outputs)
    #     return inputs, outputs

    def build_team_inputs(self, batch, ep_idx, t_idx):
        team_inputs = list()
        for agent_idx in range(self.n_agents):
            team_inputs.append(self.build_input(batch, ep_idx, t_idx, agent_idx))
        return th.stack(team_inputs)

    def build_input(self, batch, ep_idx, t_idx, agent_idx):
        agent_input = list()

        # observe the last state
        # agent_input.append(batch["obs"][ep_idx, t_idx, agent_idx])
        agent_input.append(batch["obs"][ep_idx, t_idx, agent_idx][-1:])

        # observe the last action
        # agent_input.append(batch["actions_onehot"][ep_idx, t_idx, agent_idx])

        # if self.args.obs_agent_id:
        #     agent_input.append(F.one_hot(th.tensor([agent_idx])[0], num_classes=self.n_agents))

        agent_input = th.cat(agent_input)
        return agent_input

    def build_reward_networks(self, input_shape_one_obs):
        module_list = []
        for idx, reward_groups in enumerate(self.reward_groups):
            if len(reward_groups) == 0:
                module_list.append(None)
            else:
                # reward group has idx + 1 observations
                module_list.append(self.build_reward_network(input_shape_one_obs, (idx + 1)))
        return nn.ModuleList(module_list)

    def build_reward_network(self, input_shape_one_obs, num_reward_agents):
        input_shape = input_shape_one_obs * num_reward_agents

        if self.args.reward_parameter_sharing:
            if not self.args.assume_binary_reward:
                module_sub_list = RewardNetwork(self.args, input_shape)
            else:
                # If reward is an integer that's either 0 or 1, then we know that for a pair of agents, their
                # reward together could be either -1, 0 or 1 for every agent. This means the for agent group
                # of size k, the output shape will be 2*k + 1, except for the first one.
                # For the sake of convenience, we allow single agent reward functions to be -1 as well
                if num_reward_agents == 1:
                    output_vals = [0, 1]
                else:
                    output_vals = list(range(-num_reward_agents, num_reward_agents + 1))
                    output_vals = [-1, 0, 1]
                module_sub_list = RewardClassificationNetwork(self.args, input_shape, output_vals)
        else:
            raise Exception("Currently not supporting no parameter sharing for the reward network, exiting...")
        return module_sub_list

    def single_network_forward(self, reward_input, network_idx=0):
        return self.reward_networks[network_idx](reward_input)

    def forward(self, reward_inputs):
        # Run through the reward networks
        # Note that the reward networks aren't sequential so no need for hidden states

        # We need to run every reward indices
        reward_outputs = []
        for idx, reward_groups in enumerate(self.reward_groups):
            for indices_group in reward_groups:
                # get the reward inputs of every agent in indices group
                reward_input = reward_inputs[:, :, indices_group]

                # reshape the reward input by concatenating agents inputs
                reward_input = th.reshape(reward_input, shape=(*reward_input.shape[:-2], -1))

                # get outputs from the network for this combination
                reward_output = self.reward_networks[idx](reward_input)
                reward_outputs.append(reward_output)

        # Return the output rewards
        return reward_outputs

    def get_all_combinations(self, filter_combos=True):
        # This functions returns a list of pairs, where the first element is the indices that we should take in the
        # output, and the second element is the sum that they represent (global class). We will provide an options off
        # limiting all the global classes to be between 0 and n called "filter combos

        # Start by making a list of all the options
        reward_options = []
        for idx, reward_group in enumerate(self.reward_groups):
            for idx2, _ in enumerate(reward_group):
                # Parameter Sharing assumed so using idx instead of idx2
                if self.reward_networks[idx] is not None:
                    reward_options.append(self.reward_networks[idx].output_vals)

        # Now, we compute all of these combinations by doing the product of all these lists
        # WARNING: This list is huge. exponentially large with the number of agents
        reward_combos = list(product(*reward_options))

        # Now compute the sum of every combo
        output = [(combo, sum(combo)) for combo in reward_combos]

        # Filter if necessary
        if filter_combos:
            output = [pair for pair in output if (0 <= pair[1] <= self.n_agents)]

        # Finally, convert the represent indices into actual indices
        # Create mapping array for this
        reward_to_class_idx = np.array([-min(output_val) for output_val in reward_options])
        output = [(np.array(pair[0]), pair[1]) for pair in output]
        output = [(pair[0] + reward_to_class_idx, pair[1]) for pair in output]
        return output, th.tensor(reward_to_class_idx)

    def local_probs_to_global_probs(self, local_probs):
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

    def class_probs_to_local_rewards(self, class_probs):
        # local rewards is a list of tensors of different lengths (different num of classes per reward group). We wish
        # to find the maximal class indices for every reward group, so just loop through them
        class_indices = []
        for reward_group in class_probs:
            class_indices.append(self.probs_to_class_idx(reward_group))
        class_indices = th.stack(class_indices, dim=2)
        local_rewards = self.class_idx_to_reward(class_indices)
        return local_rewards

    def probs_to_class_idx(self, probs):
        return th.argmax(probs, dim=-1)

    # Assumes that the last dimension of the arr is indices
    def class_idx_to_reward(self, class_idx):
        return class_idx - self.reward_to_class_idx.repeat(*class_idx.shape[:-1], 1)

    def reward_to_class_idx(self, reward):
        return reward + self.reward_to_class_idx.repeat(*reward.shape[:-1], 1)

    def get_regularizing_weights(self):
        regularizing_weights = []
        for idx, reward_group in enumerate(self.reward_groups):
            for _ in reward_group:
                if idx == 0:
                    regularizing_weights.append(0)
                else:
                    regularizing_weights.append(self.args.regularizing_weight)
        return th.tensor(regularizing_weights)

    def local_rewards_to_agent_rewards(self, reward_outputs):
        # Basically turns [50, 50, n_reward_groups] -> [50, 50, n_agents]

        # start by reshaping into 3 dimensinal
        local_rewards = th.zeros(*reward_outputs.shape[:2], self.n_agents)
        reward_outputs = th.reshape(reward_outputs, shape=(*reward_outputs.shape[:2], -1))

        total_idx = 0
        for idx, reward_groups in enumerate(self.reward_groups):
            for indices_group in reward_groups:
                # For every agent in the group, add the weighted reward contribution
                contrib_weight = (1 / (idx + 1))
                weighted_reward = contrib_weight * reward_outputs[:, :, [total_idx]]
                weighted_reward = weighted_reward.repeat(1, 1, (idx + 1))
                local_rewards[:, :, indices_group] += weighted_reward
                total_idx += 1

        # Return the output rewards
        return local_rewards

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
        q = F.softmax(self.fc4(h))
        return q
