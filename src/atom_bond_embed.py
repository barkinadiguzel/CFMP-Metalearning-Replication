import torch
import torch.nn as nn
from config import ATOM_FEATURE_DIM, BOND_FEATURE_DIM


class AtomBondEmbedding(nn.Module):
    def __init__(self, atom_in_dim, bond_in_dim):
        super().__init__()

        self.atom_proj = nn.Linear(atom_in_dim, ATOM_FEATURE_DIM)
        self.bond_proj = nn.Linear(bond_in_dim, BOND_FEATURE_DIM)

    def forward(self, x_atom, x_bond):
        atom_emb = self.atom_proj(x_atom)
        bond_emb = self.bond_proj(x_bond)

        return atom_emb, bond_emb
