import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from components.locality_graph import DependencyGraph


# So what does the LocalQMixer look like coceptually?
# We have a mixing layer that redirects inputs based on the graph and k
# and then we have an array of submixers

class LocalQMixer(nn.Module):
    def __init__(self, args, depth_k=2):
        super(LocalQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = args.mixing_embed_dim

        # Now optimally we will need a graph dependency between the nodes
        # For now, lets assume that all agents are in a line like in the multi_cart_pole setting
        self.depth_k = depth_k
        self.graph_obj = DependencyGraph(graph=None, num_agents=args.n_agents)

        # Each submixer needs to know the relevant agents that it is getting as input
        # TODO: Consider GNN, Convolution, Not just redirecting outputs
        self.sub_mixers = []
        for agent_index in range(self.n_agents):
            agent_nbrhood = self.graph_obj.get_nbrhood(agent_index, depth_k)
            self.sub_mixers.append(SubMixer(
                agent_index=agent_index,
                agent_nbrhood=agent_nbrhood,
                args=args
            ))

    def forward(self, agent_qs, states):
        qs = []
        for sub_mixer in self.sub_mixers:
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

            relevant_qs = agent_qs[:, :, sub_mixer.agent_nbrhood]

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
        return f"Submixer for Agent {self.agent_index} reporting for duty"

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
