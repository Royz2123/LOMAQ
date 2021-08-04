import torch.nn as nn
import torch as th
import torch.nn.functional as F


class RewardDecomposer:
    def __init__(self, scheme, args):
        self.n_agents = args.n_agents
        self.args = args

        input_shape = self._get_input_shape(scheme)
        self.reward_networks = nn.ModuleList(self.build_reward_networks(input_shape))

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

    def build_data(self, batch):
        inputs = []
        outputs = []
        for ep_idx in range(batch.batch_size):
            for t_idx in range(batch.max_seq_length):
                team_inputs = list()
                for agent_idx in range(self.n_agents):
                    team_inputs.append(self.build_input(batch, ep_idx, t_idx, agent_idx))
                inputs.append(th.stack(team_inputs))
                outputs.append(th.sum(batch["reward"][ep_idx, t_idx]))

                # we include the step that led to the terminated state, but now exit
                if batch["terminated"][ep_idx, t_idx][0]:
                    break

        inputs = th.stack(inputs)
        outputs = th.stack(outputs)
        return inputs, outputs

    def build_input(self, batch, ep_idx, t_idx, agent_idx):
        agent_input = list()

        # observe the last state
        # agent_input.append(batch["obs"][ep_idx, t_idx, agent_idx])
        agent_input.append(batch["obs"][ep_idx, t_idx, agent_idx][2:3])

        # observe the last action
        # agent_input.append(batch["actions_onehot"][ep_idx, t_idx, agent_idx])

        # if self.args.obs_agent_id:
        #     agent_input.append(F.one_hot(th.tensor([agent_idx])[0], num_classes=self.n_agents))

        agent_input = th.cat(agent_input)
        return agent_input

    def build_reward_networks(self, input_shape):
        if self.args.reward_parameter_sharing:
            module_list = [RewardNetwork(input_shape, self.args)]
        else:
            # module_list = [FFAgent(input_shape, self.args)] * self.args.n_agents
            raise Exception("Currently not supporting no parameter sharing for the reward network, exiting...")
        return module_list

    def forward(self, reward_inputs):
        # Run through the reward networks
        # Note that the reward networks aren't sequential so no need for hidden states
        local_rewards = self.reward_networks[0](reward_inputs)

        # Return the output rewards
        return local_rewards

    def parameters(self):
        return self.reward_networks.parameters()

    def load_state(self, other_mac):
        self.reward_networks.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.reward_networks.cuda()

    def save_models(self, path):
        th.save(self.reward_networks.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.reward_networks.load_state_dict(
            th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))


class RewardNetwork(nn.Module):
    def __init__(self, input_shape, args):
        super(RewardNetwork, self).__init__()
        self.args = args

        # Easiest to reuse rnn_hidden_dim variable
        self.fc1 = nn.Linear(input_shape, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        y = F.relu(self.fc2(x))
        h = F.tanh(self.fc3(y))
        q = self.fc4(h)
        return q
