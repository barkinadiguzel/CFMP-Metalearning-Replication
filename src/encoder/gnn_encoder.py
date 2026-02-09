import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from config import HIDDEN_DIM, NUM_GNN_LAYERS, DROPOUT


class GNNEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.convs = nn.ModuleList()

        for i in range(NUM_GNN_LAYERS):
            mlp = nn.Sequential(
                nn.Linear(in_dim if i == 0 else HIDDEN_DIM, HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            )
            self.convs.append(GINConv(mlp))

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, edge_index, batch):

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        # molecule-level embedding
        g = global_mean_pool(x, batch)

        return g
