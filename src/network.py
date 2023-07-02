import torch
from torch import Tensor
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing


class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="max")  # "Max" aggregation.
        self.mlp = Sequential(
            Linear(2 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        return self.propagate(edge_index, x=x)  # shape [num_nodes, out_channels]

    def message(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        # x_j: Source node features of shape [num_edges, in_channels]
        # x_i: Target node features of shape [num_edges, in_channels]
        edge_features = torch.cat([x_i, x_j - x_i], dim=-1)
        return self.mlp(edge_features)  # shape [num_edges, out_channels]


class GCN(torch.nn.Module):
    channels = [32, 64, 128, 128, 128]

    def __init__(self):
        super().__init__()
        for i in range(len(self.channels) - 1):
            in_channels = self.channels[i]
            out_channels = self.channels[i + 1]
            setattr(self, "conv{}".format(i), EdgeConv(in_channels, out_channels))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, num_node_features]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        for i in range(len(self.channels) - 1):
            conv = getattr(self, "conv{}".format(i))
            x = conv(x, edge_index)
            if i != len(self.channels) - 2:
                x = x.relu()
        return x
