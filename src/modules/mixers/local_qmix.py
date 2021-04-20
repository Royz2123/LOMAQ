import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# So what does the LocalQMixer look like coceptually?
# We have a mixing layer that redirects inputs based on the graph and k
# and then we have an array of submixers

class LocalQMixer(nn.Module):
    def __init__(self, args):
        super(LocalQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim

        # Now optimally we will need a graph dependency between the nodes
        # For now, lets assume that all agents are in a line like in the multi_cart_pole setting
        self.depth_k = self.args.depth_k
        self.graph_obj = args.graph_obj

        # create list of nbrhds in advance for every agent
        self.nbrhds = [self.graph_obj.get_nbrhood(agent_index, self.depth_k) for agent_index in range(self.n_agents)]

        # Each submixer needs to know the relevant agents that it is getting as input
        # TODO: Consider GNN, Convolution, Not just redirecting outputs

        # Notice that we store all the submixers in a nn.ModuleList. This is a f**ing
        # genius feature of Pytorch which I just found, lets you store a list of nn.Modules
        # in a single list and it recognizes their parameters automatically. Had some problems
        # doing this manually, I hope this will work now.

        # parameter sharing
        sharing_submixers = self.get_sharing_submixers()

        # We assume that every agent has the same subgraph apart from the non-sharing submixers
        shared_module = None
        if self.args.parameter_sharing and len(sharing_submixers) > 0:
            shared_module = SharedSubMixer(args=args, nbrhds=self.nbrhds, shared_idx_example=self.depth_k)

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

    def get_sharing_submixers(self):
        # Here we want to return all the indices of submixers that can share parameters
        all_submixers = list(range(self.args.n_agents))
        num_non_sharing = 2 * self.depth_k

        # No point in sharing parameters if we only have "edges" submixers
        # We should have at least 2 sharing submixers in order for this to be useful
        if len(all_submixers) <= (num_non_sharing + 1):
            return []
        else:
            return all_submixers[self.depth_k:-self.depth_k]


    def forward(self, agent_qs, states):
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

        return th.squeeze(th.stack(qs, dim=2))


class SubMixer(nn.Module):
    def __init__(self, agent_index, agent_nbrhood, args):
        super(SubMixer, self).__init__()

        self.agent_index = agent_index
        self.agent_nbrhood = agent_nbrhood
        self.args = args
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        # This part is critical for the submixers, could be the source of problems!
        # In the original architecture, the mixer (i.e: submixer) recieves the inputs
        # of all the different agents. This is why self.n_agents is saved.

        # We do not want this! For ensuring purposes, I have deleted this row:
        # self.n_agents = args.n_agents
        # and instead switched it with the property of submixer_qs_size, which I set to
        # be the size of the appropriate neighbourhood
        self.submixer_qs_size = len(self.agent_nbrhood)

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.submixer_qs_size)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.submixer_qs_size))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.ReLU(),
                                               nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def __repr__(self):
        return f"\nSubmixer for Agent {self.agent_index} reporting for duty\nNeighborhood: {self.agent_nbrhood}\n"

    def get_input_indexes(self, submixer_idx):
        if submixer_idx != self.agent_index:
            raise Exception("Can't return the input indexes for a different Submixer! No shared parameters")

        return self.agent_nbrhood

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.submixer_qs_size)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.submixer_qs_size, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot


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
