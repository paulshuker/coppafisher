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
    gene_indices = np.zeros((n_spots, n_genes_assigned), np.int32)
    gene_indices[0] = [0, 2, 3]
    gene_indices[1] = [3, 0, 1]

    score_1 = discriminality.score(
        torch.from_numpy(residual_spot_colours),
        torch.from_numpy(assigned_bled_codes),
        torch.from_numpy(bled_codes),
        torch.from_numpy(gene_indices),
    )
    assert type(score_1) is torch.Tensor
    assert score_1.ndim == 2
    assert score_1.shape == (n_spots, n_genes_assigned)

    residual_spot_colours_2 = residual_spot_colours.copy()
    assigned_bled_codes_2 = assigned_bled_codes.copy()
    gene_indices_2 = gene_indices.copy()
    temp = residual_spot_colours_2[0].copy()
    residual_spot_colours_2[0] = residual_spot_colours_2[1]
    residual_spot_colours_2[1] = temp
    temp = assigned_bled_codes_2[0].copy()
    assigned_bled_codes_2[0] = assigned_bled_codes_2[1]
    assigned_bled_codes_2[1] = temp
    temp = gene_indices_2[0].copy()
    gene_indices_2[0] = gene_indices_2[1]
    gene_indices_2[1] = temp

    score_2 = discriminality.score(
        torch.from_numpy(residual_spot_colours_2),
        torch.from_numpy(assigned_bled_codes_2),
        torch.from_numpy(bled_codes),
        torch.from_numpy(gene_indices_2),
    )
    assert type(score_2) is torch.Tensor
    assert score_2.ndim == 2
    assert score_2.shape == (n_spots, n_genes_assigned)
    # Ensure that the order of spots does not change results.
    assert torch.allclose(score_1[0], score_2[1])
    assert torch.allclose(score_1[1], score_2[0])
    assert torch.allclose(score_1[2:], score_2[2:])

    random_shuffles = [np.arange(n_genes_assigned) for _ in range(n_spots)]
    for spot in range(n_spots):
        rng.shuffle(random_shuffles[spot])
        residual_spot_colours[spot] = residual_spot_colours[spot][random_shuffles[spot]]
        assigned_bled_codes[spot] = assigned_bled_codes[spot][random_shuffles[spot]]
        gene_indices[spot] = gene_indices[spot][random_shuffles[spot]]
    score_3 = discriminality.score(
        torch.from_numpy(residual_spot_colours),
        torch.from_numpy(assigned_bled_codes),
        torch.from_numpy(bled_codes),
        torch.from_numpy(gene_indices),
    )
    assert type(score_3) is torch.Tensor
    assert score_3.ndim == 2
    assert score_3.shape == (n_spots, n_genes_assigned)
    score_3 = score_3.numpy()
    for spot in range(n_spots):
        score_3[spot] = score_3[spot][random_shuffles[spot]]
    score_3 = torch.from_numpy(score_3)
    # Ensure that the order of gene_indices[s] does not change the final results.
    assert torch.allclose(score_1, score_3)


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
