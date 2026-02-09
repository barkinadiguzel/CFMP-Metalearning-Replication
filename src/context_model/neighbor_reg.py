import torch


def neighbor_alignment_loss(H, A):

    diff = H.unsqueeze(1) - H.unsqueeze(0)
    dist = (diff ** 2).sum(-1)

    loss = (A * dist).mean()

    return loss
