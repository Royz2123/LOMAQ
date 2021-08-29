import torch as th
import torch.nn as nn
import torch.nn.functional as F


class HyperNetwork(nn.Module):
    def __init__(self, args, hyper_input_size, input_size, hyper_hidden_size, hidden_size, output_size, hyper_layers):
        super(HyperNetwork, self).__init__()

        self.args = args

        # We will implement a small 2-layer network
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Parameters for hypernetworks
        self.hyper_input_size = hyper_input_size
        self.hyper_hidden_size = hyper_hidden_size

        # Build networks. Note that setting large values here will result in very large networks, unrecommended.
        if hyper_layers == 1:
            self.hyper_w_1 = nn.Linear(self.hyper_input_size,  self.input_size * self.hidden_size)
            self.hyper_w_final = nn.Linear(self.hyper_input_size, self.hidden_size * self.output_size)
        elif hyper_layers == 2:
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.hyper_input_size, self.hyper_hidden_size),
                nn.ReLU(),
                nn.Linear(self.hyper_hidden_size, self.input_size * self.hidden_size)
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.hyper_input_size, self.hyper_hidden_size),
                nn.ReLU(),
                nn.Linear(self.hyper_hidden_size, self.hidden_size * self.output_size)
            )
        elif hyper_layers > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.hyper_input_size, self.hidden_size)

        # V(s) instead of a bias for the last layer
        self.hyper_b_final = nn.Sequential(
            nn.Linear(self.hyper_input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, regular_input, hyper_input):
        # Save batch_size for resizing
        batch_size = regular_input.size(0)
        regular_input = regular_input.view(-1, 1, self.input_size)
        hyper_input = hyper_input.reshape(-1, self.hyper_input_size)

        # First layer
        w1 = th.abs(self.hyper_w_1(hyper_input))
        b1 = self.hyper_b_1(hyper_input)
        w1 = w1.view(-1, self.input_size, self.hidden_size)
        b1 = b1.view(-1, 1, self.hidden_size)
        hidden = F.elu(th.bmm(regular_input, w1) + b1)

        # Second layer
        w_final = th.abs(self.hyper_w_final(hyper_input))
        w_final = w_final.view(-1, self.hidden_size, self.output_size)
        b_final = self.hyper_b_final(hyper_input)
        b_final = b_final.view(-1, 1, self.output_size)

        # Compute final output
        y = th.bmm(hidden, w_final) + b_final
        output = y.view(batch_size, -1, self.output_size)
        return output
