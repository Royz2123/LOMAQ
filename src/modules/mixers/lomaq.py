import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# So what does the LocalQMixer look like coceptually?
# We have a mixing layer that redirects inputs based on the graph and k
# and then we have an array of submixers
from modules.mixers.absnetwork import AbsNetwork
from modules.mixers.hypernetwork import HyperNetwork


class LocalQMixer(nn.Module):
    def __init__(self, args):
        super(LocalQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim

        # Now optimally we will need a graph dependency between the nodes
        # For now, lets assume that all agents are in a line like in the multi_cart_pole setting
        self.value_depth_k = int(self.args.value_depth_k)
        self.graph_obj = args.graph_obj

        # create list of nbrhds in advance for every agent
        self.nbrhds = [
            self.graph_obj.get_nbrhood(agent_index, self.value_depth_k)
            for agent_index in range(self.n_agents)
        ]

        # Each submixer needs to know the relevant agents that it is getting as input
        # TODO: Consider GNN, Convolution, Not just redirecting outputs

        # Notice that we store all the submixers in a nn.ModuleList. This is a f**ing
        # genius feature of Pytorch which I just found, lets you store a list of nn.Modules
        # in a single list and it recognizes their parameters automatically. Had some problems
        # doing this manually, I hope this will work now.

        # parameter sharing
        submixer_non_share_depth = getattr(self.args, "submixer_non_share_depth", 0)
        sharing_submixers = self.get_sharing_submixers(submixer_non_share_depth)

        # We assume that every agent has the same subgraph apart from the non-sharing submixers
        shared_module = None
        if self.args.submixer_parameter_sharing and len(sharing_submixers) > 0:
            shared_module = SharedSubMixer(args=args, nbrhds=self.nbrhds, shared_idx_example=submixer_non_share_depth)

        # create the module_list base on who wants to share and who doesn't
        module_list = []
        for agent_index in range(self.n_agents):
            if shared_module is None or (agent_index not in sharing_submixers):
                module_list.append(SubMixer(
                    agent_index=agent_index,
                    agent_nbrhood=self.nbrhds[agent_index],
                    args=args
                ))
            else:
                module_list.append(shared_module)

        self.sub_mixers = nn.ModuleList(module_list)

    def get_sharing_submixers(self, submixer_non_share_depth):
        # Here we want to return all the indices of submixers that can share parameters
        all_submixers = list(range(self.args.n_agents))
        num_non_sharing = 2 * submixer_non_share_depth

        # No point in sharing parameters if we only have "edges" submixers
        # We should have at least 2 sharing submixers in order for this to be useful
        if len(all_submixers) <= (num_non_sharing + 1):
            return []
        elif submixer_non_share_depth == 0:
            return all_submixers
        else:
            return all_submixers[submixer_non_share_depth:-submixer_non_share_depth]

    def forward(self, agent_qs, states, obs=None):
        qs = []
        for idx, sub_mixer in enumerate(self.sub_mixers):
            # States shape is: [batch_size, max_ep_len, num_agents * state_size]
            # Agent_qs shape is: [batch_size, max_ep_len, num_agents]
            #
            # If I understand this correctly, agent_qs are the outputs of the individual
            # agent networks that are merged in the mixer, and the states is just whats
            # being fed from the side to the mixer. In that case, we can feed the mixer
            # either the whole state or the truncated state and compare those. Lets start
            # off with the whole state and see what happens
            # TODO: Check truncated state.
            #
            # As for the agent_qs, we only want to take the q_s that are in the relevant
            # nbrhood. The submixer has this property stored (so that we don't compute
            # shortest path every time)

            relevant_qs = agent_qs[:, :, sub_mixer.get_input_indexes(submixer_idx=idx)]

            # So now the updated states are
            # States shape is: [batch_size, max_ep_len, num_agents * state_size] - consider truncated
            # Agent_qs shape is: [batch_size, max_ep_len, len(sub_mixer.agent_nbrhood)]

            # Redirect States to SubMixers
            qs.append(sub_mixer.forward(relevant_qs, states))

        qs = th.stack(qs, dim=2)
        qs = th.reshape(qs, shape=qs.shape[:3])
        return qs


class SubMixer(nn.Module):
    def __init__(self, agent_index, agent_nbrhood, args):
        super(SubMixer, self).__init__()

        self.agent_index = agent_index
        self.agent_nbrhood = agent_nbrhood
        self.args = args
        self.state_dim = int(np.prod(args.state_shape))

        self.use_hyper_network = getattr(args, "monotonicity_network", "hyper") == "hyper"
        self.use_abs_network = getattr(args, "monotonicity_network", "hyper") in ["abs", "relu", "leaky_relu"]
        self.positive_weights = getattr(args, "monotonicity_method", "weights") == "weights"

        # This part is critical for the submixers, could be the source of problems!
        # In the original architecture, the mixer (i.e: submixer) recieves the inputs
        # of all the different agents. This is why self.n_agents is saved.

        # We do not want this! For ensuring purposes, I have deleted this row:
        # self.n_agents = args.n_agents
        # and instead switched it with the property of submixer_qs_size, which I set to
        # be the size of the appropriate neighbourhood
        self.submixer_qs_size = len(self.agent_nbrhood)

        self.input_size = len(self.agent_nbrhood)
        self.output_size = 1
        self.hidden_size = args.mixing_embed_dim

        # breaking the MLP to hypernetworks for deriving the weights and biases
        # We will implement a small 2-layer network for every submixer
        # The dimensions will be (feature_size, sub_mixer_embed_dim, 1)
        if self.positive_weights:
            if self.use_hyper_network:
                self.hyper_input_size = int(np.prod(args.state_shape))
                self.hyper_hidden_size = self.args.hypernet_embed
                self.hyper_layers = getattr(args, "hypernet_layers", 1)

                self.hyper_network = HyperNetwork(
                    args,
                    self.hyper_input_size,
                    self.input_size,
                    self.hyper_hidden_size,
                    self.hidden_size,
                    self.output_size,
                    self.hyper_layers,
                )
            elif self.use_abs_network:
                self.abs_network = AbsNetwork(
                    args,
                    self.input_size,
                    self.hidden_size,
                    self.output_size,
                    3
                )
            else:
                raise Exception("Unrecognized monotonicity_network")
        else:
            self.network = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, self.output_size)
            )

    def __repr__(self):
        return f"\nSubmixer for Agent {self.agent_index} reporting for duty\nNeighborhood: {self.agent_nbrhood}\n"

    def get_input_indexes(self, submixer_idx):
        if submixer_idx != self.agent_index:
            raise Exception("Can't return the input indexes for a different Submixer! No shared parameters")

        return self.agent_nbrhood

    def forward(self, utilities, states):
        # Now flatten the last 2 dimensions for the network
        utilities = th.reshape(utilities, shape=(
            *utilities.shape[:2], self.input_size)
        )

        if self.use_hyper_network:
            q_i = self.hyper_network(utilities, states)
        elif self.use_abs_network:
            q_i = self.abs_network(utilities)
        else:
            q_i = self.network(utilities)

        # Return to original shape
        q_i = th.reshape(q_i, shape=(
            *utilities.shape[:2], self.output_size)
        )
        return q_i


class SharedSubMixer(SubMixer):
    def __init__(self, args, nbrhds, shared_idx_example):
        # call super using default args, just create a submixer for the first one
        SubMixer.__init__(
            self,
            agent_index=shared_idx_example,
            agent_nbrhood=nbrhds[shared_idx_example],
            args=args
        )

        self.args = args
        self.nbrhds = nbrhds

    def __repr__(self):
        return f"\nShared Submixer reporting for duty\nMultiple neighborhoods relevant\n"

    def get_input_indexes(self, submixer_idx):
        if submixer_idx is None:
            raise Exception("Working with shared parameters, need to specify the submixer_idx")

        return self.nbrhds[submixer_idx]
