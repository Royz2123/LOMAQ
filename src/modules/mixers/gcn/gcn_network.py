import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.mixers.gcn.gcn_layer import MonotonicGCNLayer


class MonotonicGCN(nn.Module):
    def __init__(self, args):
        super(MonotonicGCN, self).__init__()

        self.args = args
        self.n_agents = args.n_agents

        self.value_depth_k = int(self.args.value_depth_k)
        self.graph_obj = args.graph_obj
        self.adj_matrix = th.tensor(self.graph_obj.get_adjacency_matrix()).to(args.device)

        self.layers = nn.ModuleList()
        for i in range(self.value_depth_k):
            input_feature_size = 1 if i == 0 else args.gnn_feature_size
            output_feature_size = args.gnn_feature_size

            self.layers.append(MonotonicGCNLayer(
                args=args,
                input_feature_size=input_feature_size,
                output_feature_size=output_feature_size,
                adj_matrix=self.adj_matrix
            ))

    def forward(self, input_features, states):
        x = input_features
        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x, states))
        return x
