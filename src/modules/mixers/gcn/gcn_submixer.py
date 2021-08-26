import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.mixers.gcn.hypernetwork import HyperNetwork


# Quite a simple component - basically recieves the mixed node feature from the GCN and does some fine tuning so that
# it really matches Q_i. Since this unit is small, and will often not be parameter shared, we input into it only the
# local state s_nk, since this is what it should represnt! This is a nice tweak for scalability.
class MonotonicSubMixer(nn.Module):
    def __init__(self, args, mixer_index, mixer_neighborhood):
        super(MonotonicSubMixer, self).__init__()

        self.args = args
        self.use_local_state = getattr(args, "submixer_use_local_state", True)

        # We denote i as the mixer index, and mixer_neighborhood as the relevant indices of utilities. Note that
        # mixer neighborhood is also exactly what we need for extracting S_Nik
        self.mixer_index = mixer_index
        self.mixer_neighborhood = mixer_neighborhood

        # We will implement a small 2-layer network for every submixer
        # The dimensions will be (feature_size, sub_mixer_embed_dim, 1)
        self.hyper_input_size = int(np.prod(args.state_shape))
        self.input_size = getattr(args, "gnn_feature_size", 1)
        self.hyper_hidden_size = getattr(args, "submixer_hypernet_hidden_size", 1)
        self.hidden_size = getattr(args, "submixer_hidden_size", 1)
        self.output_size = 1
        self.hyper_layers = getattr(args, "submixer_hypernet_layers", 1)

        self.network = HyperNetwork(
            args,
            self.hyper_input_size,
            self.input_size,
            self.hyper_hidden_size,
            self.hidden_size,
            self.output_size,
            self.hyper_layers,
        )

    def __repr__(self):
        return f"\nSubmixer {self.mixer_index} reporting for duty\nNeighborhood: {self.mixer_neighborhood}\n"

    def get_input_indexes(self, submixer_idx):
        if submixer_idx != self.agent_index:
            raise Exception("Can't return the input indexes for a different Submixer! No shared parameters")
        return self.agent_nbrhood

    def forward(self, mixed_utilities, states):
        # Retriving only the relevant mixer nieghborhood
        # TODO: Local state shit

        # Running through the hypernetwork
        q_i = self.network(mixed_utilities[:, :, self.mixer_index], states)
        return q_i
