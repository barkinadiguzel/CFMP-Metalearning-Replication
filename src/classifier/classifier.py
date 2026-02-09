import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, h):
        return self.fc(h)
