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

        self.layers = []

        for i in range(total_layers):
            layer_input_size = hidden_size
            layer_output_size = hidden_size
            if i == 0:
                layer_input_size = input_size
            elif i == (total_layers - 1):
                layer_output_size = output_size

            # Add the layer
            self.layers.append((
                nn.Parameter(th.randn((layer_input_size, layer_output_size))).to(self.args.device),
                nn.Parameter(th.zeros((layer_output_size,)).to(self.args.device))
            ))

    def forward(self, regular_input):
        # Save batch_size for resizing
        batch_size = regular_input.size(0)
        curr_input = regular_input.view(-1, 1, self.input_size)

        for layer in self.layers:
            curr_input = th.matmul(curr_input, th.abs(layer[0])) + layer[1]

        output = curr_input.view(batch_size, -1, self.output_size)
        return output
