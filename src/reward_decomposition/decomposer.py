import torch.nn as nn
import torch as th
import torch.nn.functional as F

import numpy as np
from itertools import combinations


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

        # input_shape = 2

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
        agent_input.append(batch["obs"][ep_idx, t_idx, agent_idx])
        # agent_input.append(batch["obs"][ep_idx, t_idx, agent_idx][2:4])

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
                input_shape = input_shape_one_obs * (idx + 1)
                module_list.append(self.build_reward_network(input_shape))
        return nn.ModuleList(module_list)

    def build_reward_network(self, input_shape):
        if self.args.reward_parameter_sharing:
            if not self.args.assume_binary_reward:
                module_sub_list = RewardNetwork(input_shape, self.args)
            else:
                module_sub_list = RewardClassificationNetwork(input_shape, self.args)
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
        return th.stack(reward_outputs, dim=2)

    def local_probs_to_global_probs(self, local_probs):
        local_probs = th.squeeze(local_probs)
        num_classes = self.n_agents + 1

        if local_probs.shape[2] != self.n_agents:
            raise Exception("Can't assume classification & large reward functions")

        global_probs = th.zeros(*local_probs.shape[:2], num_classes)
        for class_num in range(num_classes):
            global_prob_is_class = th.zeros(*local_probs.shape[:2])

            # Find every indices group that is of size class_num, and compute their probability
            indices_group = list(combinations(range(self.n_agents), class_num))
            for indicies in indices_group:
                # Compute probability for this indices group
                curr_probs = th.ones(*local_probs.shape[:2])
                for idx in range(self.n_agents):
                    if idx in indicies:
                        curr_probs *= local_probs[:, :, idx]
                    else:
                        curr_probs *= (1 - local_probs[:, :, idx])
                global_prob_is_class += curr_probs

            global_probs[:, :, class_num] = global_prob_is_class

        return global_probs

    @staticmethod
    def probs_to_reward(probs):
        return th.where(probs > 0.5, 1., 0.)

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
    def __init__(self, input_shape, args):
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
    def __init__(self, input_shape, args):
        super(RewardClassificationNetwork, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        y = F.leaky_relu(self.fc2(x))
        h = F.tanh(self.fc3(y))
        q = F.softmax(self.fc4(h))
        return q