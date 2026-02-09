import torch
import torch.nn as nn


class RelationUpdate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, H, R):

        agg = torch.matmul(R, H)
        H_new = H + self.lin(agg)

        return H_new
