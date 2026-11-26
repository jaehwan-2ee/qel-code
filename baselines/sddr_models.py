"""Score networks of the SDDR baselines. A network reads the features of a whole instance as one vector x and returns
one score per decision variable.

  shared        s(x) = W2 tanh(W1 x + b1) + b2
  independent   s_k(x) = w2_k . tanh(W1_k x + b1_k) + b2_k, one MLP per decision variable k

Both use the default initialization of nn.Linear, drawn in construction order right after seeding with the model seed.
"""

import numpy as np
import torch
from torch import nn

from data import BM_NODES


def flatten_context(problem, X):
    """The network input of every instance, float32. BA and KP: the item features, flattened. BM: the features of the
    10 left and the 10 right nodes; X holds them once per pair (pair 10 i + j is [left_i | right_j]), so nine of the ten
    copies of each node are dropped."""
    X = np.asarray(X, np.float32)
    if problem != "bm":
        return X.reshape(len(X), -1)
    half = X.shape[-1] // 2
    left = X[..., :half].reshape(len(X), BM_NODES, BM_NODES, half)[:, :, 0]
    right = X[..., half:].reshape(len(X), BM_NODES, BM_NODES, half)[:, 0]
    return np.concatenate((left.reshape(len(X), -1), right.reshape(len(X), -1)), axis=1)


class FullContextMLP(nn.Module):
    """shared: one MLP for all decision variables."""

    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.hidden_layer = nn.Linear(in_dim, hidden)
        self.output_layer = nn.Linear(hidden, out_dim)

    def forward(self, inputs):
        return self.output_layer(torch.tanh(self.hidden_layer(inputs)))


class IndependentFullContextMLP(nn.Module):
    """independent: one MLP per decision variable, stored as stacked weights. They are initialized one after another
    as nn.Linear(D, H) and nn.Linear(H, 1), which fixes the random draws."""

    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        hidden_weight, hidden_bias, output_weight, output_bias = [], [], [], []
        for _ in range(out_dim):
            first = nn.Linear(in_dim, hidden)
            second = nn.Linear(hidden, 1)
            hidden_weight.append(first.weight.detach())
            hidden_bias.append(first.bias.detach())
            output_weight.append(second.weight.detach().reshape(hidden))
            output_bias.append(second.bias.detach().reshape(()))
        self.hidden_weight = nn.Parameter(torch.stack(hidden_weight))
        self.hidden_bias = nn.Parameter(torch.stack(hidden_bias))
        self.output_weight = nn.Parameter(torch.stack(output_weight))
        self.output_bias = nn.Parameter(torch.stack(output_bias))

    def forward(self, inputs):
        context = inputs.unsqueeze(0).expand(len(self.hidden_weight), -1, -1)  # the same input for every MLP
        hidden = torch.bmm(context, self.hidden_weight.transpose(1, 2)) + self.hidden_bias.unsqueeze(1)
        activated = torch.tanh(hidden)
        output = torch.bmm(activated, self.output_weight.unsqueeze(-1)).squeeze(-1)
        return (output + self.output_bias.unsqueeze(1)).transpose(0, 1).contiguous()


def make_model(structure, in_dim, hidden, out_dim, seed):
    torch.manual_seed(seed)
    return (IndependentFullContextMLP if structure == "independent" else FullContextMLP)(in_dim, hidden, out_dim)
