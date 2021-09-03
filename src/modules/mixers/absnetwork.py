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

        self.abs_method = getattr(args, "monotonicity_network", "hyper")

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for layer_idx in range(total_layers):
            layer_input_size, layer_output_size = self.get_layer_dim(layer_idx)

            # Add the layer
            self.weights.append(nn.Parameter(th.randn((layer_input_size, layer_output_size)).to(self.args.device)))
            self.biases.append(nn.Parameter(th.zeros((layer_output_size,)).to(self.args.device)))

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

        for layer_idx in range(self.total_layers):
            layer_input_size, layer_output_size = self.get_layer_dim(layer_idx)

            w = self.weights[layer_idx]
            if self.abs_method == "relu":
                w = F.relu(w)
            elif self.abs_method == "abs":
                w = th.abs(w)
            elif self.abs_method == "leaky_relu":
                w = F.leaky_relu(w)
            else:
                raise Exception("Unrecognized abs method: %s" % self.abs_method)
            w = w.view(-1, layer_input_size, layer_output_size)
            b = self.biases[layer_idx].view(-1, 1, layer_output_size)

            # repeat both biases and weights to fit size
            w = w.repeat(curr_input.shape[0], 1, 1)
            b = b.repeat(curr_input.shape[0], 1, 1)

            curr_input = th.bmm(curr_input, w) + b

            if layer_idx != (self.total_layers - 1):
                curr_input = F.elu(curr_input)

        output = curr_input.view(batch_size, -1, self.output_size)
        return output
