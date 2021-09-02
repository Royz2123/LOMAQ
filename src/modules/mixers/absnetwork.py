import torch as th
import torch.nn as nn
import torch.nn.functional as F


class AbsNetwork(nn.Module):
    def __init__(self, args, input_size, hidden_size, output_size, total_layers=2):
        super(AbsNetwork, self).__init__()

        self.args = args

        # We will implement a small 2-layer network
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.total_layers = total_layers

        self.layers = []

        for layer_idx in range(total_layers):
            layer_input_size, layer_output_size = self.get_layer_dim(layer_idx)

            # Add the layer
            self.layers.append((
                nn.Parameter(th.randn((layer_input_size, layer_output_size)).to(self.args.device)),
                nn.Parameter(th.zeros((layer_output_size,)).to(self.args.device))
            ))

    def get_layer_dim(self, layer_idx):
        layer_input_size = self.hidden_size
        layer_output_size = self.hidden_size
        if layer_idx == 0:
            layer_input_size = self.input_size
        elif layer_idx == (self.total_layers - 1):
            layer_output_size = self.output_size
        return layer_input_size, layer_output_size

    def forward(self, regular_input):
        # Save batch_size for resizing
        batch_size = regular_input.size(0)
        curr_input = regular_input.view(-1, 1, self.input_size)

        for layer_idx, layer in enumerate(self.layers):
            layer_input_size, layer_output_size = self.get_layer_dim(layer_idx)
            w = F.relu(layer[0]).view(-1, layer_input_size, layer_output_size)
            b = layer[1].view(-1, 1, layer_output_size)

            # repeat both biases and weights to fit size
            w = w.repeat(curr_input.shape[0], 1, 1)
            b = b.repeat(curr_input.shape[0], 1, 1)

            curr_input = th.bmm(curr_input, w) + b

            if layer_idx != (self.total_layers - 1):
                curr_input = F.elu(curr_input)

        output = curr_input.view(batch_size, -1, self.output_size)
        return output
