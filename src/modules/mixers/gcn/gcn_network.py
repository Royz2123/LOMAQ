import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.mixers.gcn.gcn_layer import MonotonicGCNLayer


class MonotonicGCN(nn.Module):
    def __init__(self, args):
        super(MonotonicGCN, self).__init__()

        self.args = args
        self.n_agents = args.n_agents

        self.depth_k = int(self.args.depth_k)
        self.graph_obj = args.graph_obj
        self.adj_matrix = th.tensor(self.graph_obj.get_adjacency_matrix())

        self.layers = nn.ModuleList()
        for i in range(self.depth_k):
            input_feature_size = 1 if i == 0 else getattr(args, "gnn_feature_size", 8)
            output_feature_size = getattr(args, "gnn_feature_size", 8)

            self.layers.append(MonotonicGCNLayer(
                args=args,
                input_feature_size=input_feature_size,
                output_feature_size=output_feature_size,
                adj_matrix=self.adj_matrix
            ))

    def forward(self, input_features, states):
        x = input_features
        for i in range(len(self.layers)):
            x = F.elu(self.layers[i](x, states))
        return x
