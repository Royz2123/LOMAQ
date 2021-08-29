import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.mixers.hypernetwork import HyperNetwork


class MonotonicGCNLayer(nn.Module):
    def __init__(self, args, input_feature_size, output_feature_size, adj_matrix):
        super(MonotonicGCNLayer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.input_feature_size = input_feature_size
        self.output_feature_size = output_feature_size

        # check if hypernetwork should be used
        self.use_hyper_network = getattr(args, "gcn_use_hypernetwork", False)

        # Save adjacency matrix
        self.adj_matrix = adj_matrix

        # Set global parameters
        self.input_size = input_feature_size * self.n_agents
        self.output_size = output_feature_size * self.n_agents
        self.hidden_size = int((self.input_size + self.output_size) / 2)

        # breaking the MLP to hypernetworks for deriving the weights and biases
        # We will implement a small 2-layer network for every submixer
        # The dimensions will be (feature_size, sub_mixer_embed_dim, 1)
        if self.use_hyper_network:
            self.hyper_input_size = int(np.prod(args.state_shape))
            self.hyper_hidden_size = args.gnn_hyper_hidden_size
            self.hyper_layers = args.gnn_hyper_layers

            self.network = HyperNetwork(
                args,
                self.hyper_input_size,
                self.input_size,
                self.hyper_hidden_size,
                self.hidden_size,
                self.output_size,
                self.hyper_layers,
            )
        else:
            self.weight = nn.Parameter(th.randn((self.input_size, self.output_size)))
            self.bias = nn.Parameter(th.zeros((self.output_size,)))

    def forward(self, input_features, states):
        # First, multiply the input_features with adj matrix
        input_features = th.reshape(input_features, shape=(
            *input_features.shape[:2], self.n_agents, self.input_feature_size)
        )
        features = th.matmul(self.adj_matrix, input_features)

        # Now flatten the last 2 dimensions for the gcn weights
        features = th.reshape(features, shape=(
            *input_features.shape[:2], self.input_size)
        )

        if self.use_hyper_network:
            output_features = self.network(features, states)
        else:
            output_features = th.matmul(features, th.abs(self.weight)) + self.bias

        # Return to original shape
        output_features = th.reshape(output_features, shape=(
            *output_features.shape[:2], self.n_agents, self.output_feature_size)
        )
        return output_features
