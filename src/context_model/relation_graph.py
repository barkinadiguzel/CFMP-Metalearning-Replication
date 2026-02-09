import torch
import torch.nn as nn


class RelationGraph(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.edge_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, H, A):
        N, D = H.shape
        R = torch.zeros(N, N, device=H.device)

        for i in range(N):
            for j in range(N):
                if A[i, j] > 0:
                    pair = torch.cat([H[i], H[j]])
                    R[i, j] = self.edge_mlp(pair)

        return R
