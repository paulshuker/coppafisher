import numpy as np
import torch

from coppafisher.omp import discriminality


def test_score() -> None:
    rng = np.random.RandomState(0)
    n_spots = 2
    n_genes_assigned = 3
    n_genes = 4
    n_rounds_use = 5
    n_channels_use = 2

    residual_spot_colours = rng.rand(n_spots, n_genes_assigned, n_rounds_use, n_channels_use).astype(np.float32)
    assigned_bled_codes = rng.rand(n_spots, n_genes_assigned, n_rounds_use, n_channels_use).astype(np.float32)
    bled_codes = rng.rand(n_genes, n_rounds_use, n_channels_use).astype(np.float32)
    gene_indices = rng.randint(n_genes, size=(n_spots, n_genes_assigned)).astype(np.int32)

    score = discriminality.score(
        torch.from_numpy(residual_spot_colours),
        torch.from_numpy(assigned_bled_codes),
        torch.from_numpy(bled_codes),
        torch.from_numpy(gene_indices),
    )
    assert type(score) is torch.Tensor
    assert score.ndim == 2
    assert score.shape == (n_spots, n_genes_assigned)


def test_spearman_score() -> None:
    rng = np.random.RandomState(0)
    xs = rng.rand(3, 4, 5).astype(np.float32)
    ys = rng.rand(3, 2, 5).astype(np.float32)
    results = discriminality.spearman_score(torch.from_numpy(xs), torch.from_numpy(ys))
    expected_results = np.array(
        [
            [[-0.3, 0.3], [0.0, 0.6], [-0.2, 0.7], [0.8, -0.7]],
            [[-0.2, -0.1], [0.9, -0.3], [-0.6, 0.8], [0.0, 0.1]],
            [[-0.6, -0.3], [0.3, 0.4], [-1.0, -0.9], [0.3, 0.1]],
        ],
        np.float32,
    )

    assert type(results) is torch.Tensor
    assert results.dtype == torch.float32
    assert results.shape == (3, 4, 2)
    assert np.allclose(results.numpy(), expected_results)
