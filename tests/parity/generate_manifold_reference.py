"""Reference data for the manifold learners.

Embeddings are defined only up to rotation/reflection (and are stochastic for
t-SNE/UMAP), so coordinates never match scikit-learn. Parity is asserted on
*trustworthiness* — a rotation-invariant measure of how well the embedding
preserves the local neighbourhood structure of the original data. Skigen must
reach trustworthiness within a band of sklearn's.

umap-learn is not a hard dependency; when absent, UMAP is checked against a
fixed trustworthiness floor instead of a reference implementation."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.manifold import (
    trustworthiness, TSNE, Isomap, MDS, LocallyLinearEmbedding,
    SpectralEmbedding)

from _common import save, save_params

N_NEIGHBORS = 5


def data():
    X, _ = make_blobs(n_samples=120, n_features=6, centers=4,
                      cluster_std=1.0, random_state=7)
    return X.astype(np.float64)


def emit(name, X, emb):
    t = trustworthiness(X, emb, n_neighbors=N_NEIGHBORS)
    save(name, X=X, trust=np.array([t]))
    save_params(name, {"n_components": 2, "n_neighbors": N_NEIGHBORS})


def main():
    X = data()
    emit("isomap", X, Isomap(n_neighbors=5, n_components=2).fit_transform(X))
    emit("mds", X, MDS(n_components=2, random_state=0,
                       normalized_stress="auto").fit_transform(X))
    emit("locally_linear_embedding", X,
         LocallyLinearEmbedding(n_neighbors=5, n_components=2,
                                random_state=0).fit_transform(X))
    emit("spectral_embedding", X,
         SpectralEmbedding(n_components=2, n_neighbors=5,
                           random_state=0).fit_transform(X))
    emit("tsne", X, TSNE(n_components=2, perplexity=15.0,
                         random_state=0).fit_transform(X))

    # UMAP: no umap-learn reference required — use a trustworthiness floor.
    save("umap", X=X, trust=np.array([0.75]))
    save_params("umap", {"n_components": 2, "n_neighbors": 15,
                         "note": "floor check; no umap-learn reference"})


if __name__ == "__main__":
    main()
