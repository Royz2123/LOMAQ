import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.mixers.gcn.gcn_network import MonotonicGCN
from modules.mixers.gcn.gcn_submixer import MonotonicSubMixer


class GraphQMixer(nn.Module):
    def __init__(self, args):
        super(GraphQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents

        # Now optimally we will need a graph dependency between the nodes
        # For now, lets assume that all agents are in a line like in the multi_cart_pole setting
        self.graph_obj = args.graph_obj
        self.value_depth_k = int(self.args.value_depth_k)
        self.nbrhds = [self.graph_obj.get_nbrhood(agent_index, self.value_depth_k) for agent_index in range(self.n_agents)]

        # Create all relevant submodules
        self.mixing_gcn_network = MonotonicGCN(args)

        # We assume a full partition, and create n sub_mixers
        self.sub_mixers = nn.ModuleList()
        for mixer_index in range(self.n_agents):
            self.sub_mixers.append(MonotonicSubMixer(
                mixer_index=mixer_index,
                mixer_neighborhood=self.nbrhds[mixer_index],
                args=args
            ))

        # Implement parameter sharing for sub_mixers
        if args.share_first_layer:
            hyper_w_1 = self.sub_mixers[0].network.hyper_w_1
            hyper_b_1 = self.sub_mixers[0].network.hyper_b_1

            for sub_mixer in self.sub_mixers:
                sub_mixer.network.hyper_w_1 = hyper_w_1
                sub_mixer.network.hyper_b_1 = hyper_b_1

    def forward(self, utilities, states, obs=None):
        mixed_utilities = self.mixing_gcn_network(utilities, states)

        qs = []
        for idx, sub_mixer in enumerate(self.sub_mixers):
            qs.append(sub_mixer.forward(mixed_utilities, states))

        return th.squeeze(th.stack(qs, dim=2))




