import torch.nn as nn

from src.encoder.atom_bond_embed import AtomBondEmbedding
from src.encoder.gnn_encoder import GNNEncoder
from src.context_model.relation_update import RelationUpdate
from src.classifier.classifier import Classifier


class MetaModel(nn.Module):
    def __init__(self, config, atom_dim, bond_dim):
        super().__init__()

        self.embed = AtomBondEmbedding(atom_dim, bond_dim, config.hidden_dim)
        self.encoder = GNNEncoder(config.hidden_dim, config.num_layers)
        self.context_update = RelationUpdate(config.hidden_dim)
        self.classifier = Classifier(config.hidden_dim, config.num_classes)

    def forward(self, atom_feat, bond_feat, edge_index, relation_mat):
        atom_h, _ = self.embed(atom_feat, bond_feat)
        h = self.encoder(atom_h, edge_index)
        h = self.context_update(h, relation_mat)
        out = self.classifier(h)
        return out
