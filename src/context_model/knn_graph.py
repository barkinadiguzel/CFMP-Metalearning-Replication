import torch
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(embeddings, k=5):
    emb_np = embeddings.detach().cpu().numpy()

    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine")
    nbrs.fit(emb_np)

    _, indices = nbrs.kneighbors(emb_np)

    N = embeddings.size(0)
    A = torch.zeros(N, N, device=embeddings.device)

    for i in range(N):
        A[i, indices[i]] = 1.0

    return A
