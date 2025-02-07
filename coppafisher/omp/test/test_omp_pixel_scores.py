import numpy as np
import torch

from coppafisher.omp.pixel_scores import PixelScoreSolver
from coppafisher.utils import base
from coppafisher.utils import intensity as utils_intensity


def test_solve() -> None:
    rng = np.random.RandomState(0)

    dtype = np.float32
    n_pixels = 5
    n_genes = 7
    n_rounds = 2
    n_channels = 3

    solver = PixelScoreSolver()
    pixel_colours = rng.rand(n_pixels, n_rounds, n_channels).astype(dtype)
    bled_codes = rng.rand(n_genes, n_rounds, n_channels).astype(dtype)
    bled_codes /= np.linalg.norm(bled_codes, axis=(-1, -2), keepdims=True)
    bg_codes = solver.create_background_bled_codes(n_rounds, n_channels)
    maximum_iterations = 4
    dot_product_threshold = 0.001
    minimum_intensity = 0.0
    alpha = 0.0
    beta = 1.0

    # Simple checks for consistent results and correct shapes.
    previous_result = None
    for return_all_scores in (True, False):
        for return_all_residuals in (True, False):
            result = solver.solve(
                pixel_colours,
                bled_codes,
                bg_codes,
                maximum_iterations,
                dot_product_threshold,
                minimum_intensity,
                alpha,
                beta,
                return_all_scores=return_all_scores,
                return_all_residuals=return_all_residuals,
            )
            if return_all_scores:
                assert type(result[1]) is np.ndarray
                assert result[1].shape[0] >= 1
                assert result[1].shape[1:] == (n_pixels, n_genes + n_channels)
                assert result[1].dtype == dtype
                assert (result[1] >= 0).all()
            if return_all_residuals:
                assert type(result[1 + int(return_all_scores)]) is np.ndarray
                assert result[1 + int(return_all_scores)].shape == (n_pixels, n_genes, n_rounds, n_channels)
                assert result[1 + int(return_all_scores)].dtype == dtype
            if type(result) is tuple:
                result = result[0]
            assert type(result) is np.ndarray
            assert result.shape == (n_pixels, n_genes)
            assert result.dtype == dtype
            if previous_result is not None:
                assert np.allclose(result, previous_result)
            previous_result = result

    # Ensure the number of assigned genes only decreases as the dot product threshold increases.
    previous_n_genes_assigned = n_pixels * n_genes + 1
    for dp_threshold in [dot_product_threshold + 0.001 * i for i in range(1, 100)] + [10.0]:
        result = solver.solve(
            pixel_colours, bled_codes, bg_codes, maximum_iterations, dp_threshold, minimum_intensity, alpha, beta
        )
        n_genes_assigned = (~np.isclose(result, 0)).sum()
        assert n_genes_assigned < n_pixels * n_genes + 1
        assert n_genes_assigned <= previous_n_genes_assigned

        previous_n_genes_assigned = n_genes_assigned

    # Run with obvious expected gene assignments.
    n_channels = 4
    n_rounds = 5
    n_pixels = 8
    pixel_colours = np.zeros((n_pixels, n_rounds, n_channels), dtype)
    bled_codes = np.zeros((n_genes, n_rounds, n_channels), dtype)
    bg_codes = solver.create_background_bled_codes(n_rounds, n_channels)
    minimum_intensity = 0.2
    reed_bled_codes = base.reed_solomon_codes(n_genes, n_rounds, n_channels)
    for g, gene_code in enumerate(reed_bled_codes.values()):
        for r, digit in enumerate(gene_code):
            bled_codes[g, r, int(digit)] = 2
    bled_codes += rng.rand(n_genes, n_rounds, n_channels) * 0.02
    bled_codes /= np.linalg.norm(bled_codes, axis=(-1, -2), keepdims=True)
    maximum_iterations = 2
    dot_product_threshold = 0.5
    expected_gene_assignments = [(0, 2), (1, 2), (4, 5), (4,), (6,), (0, 6), tuple(), (0,)]
    for p, gene_assignments in enumerate(expected_gene_assignments):
        pixel_colours[p] = 0
        for g in gene_assignments:
            pixel_colours[p] += (rng.rand() + 2) * bled_codes[g]
    result = solver.solve(
        pixel_colours, bled_codes, bg_codes, maximum_iterations, dot_product_threshold, minimum_intensity, alpha, beta
    )
    for p in range(n_pixels):
        assert (~np.isclose(result[p], 0)).sum() == len(expected_gene_assignments[p])
        for g in expected_gene_assignments[p]:
            assert (result[p] > 0)[g]

    # Check gene assignments fail if the pixel colour is too low in intensity.
    dim_gene_assignments = [True, False, True, True, False, False, False, True]
    for p, dim in enumerate(dim_gene_assignments):
        if not dim:
            continue
        intensity = utils_intensity.compute_intensity(pixel_colours[[p]]).item()
        pixel_colours[p] *= (0.9 + rng.rand() * 0.1) * minimum_intensity / intensity
        intensity = utils_intensity.compute_intensity(pixel_colours[[p]]).item()
        assert intensity < minimum_intensity
    result = solver.solve(
        pixel_colours, bled_codes, bg_codes, maximum_iterations, dot_product_threshold, minimum_intensity, alpha, beta
    )
    for p, dim in enumerate(dim_gene_assignments):
        if dim:
            assert (~np.isclose(result[p], 0)).sum() == 0
            continue
        assert (~np.isclose(result[p], 0)).sum() == len(expected_gene_assignments[p])
        for g in expected_gene_assignments[p]:
            assert (result[p] > 0)[g]


def test_get_next_gene_assignments() -> None:
    n_pixels = 6
    n_rounds = 1
    n_channels = 5
    residual_colours = torch.zeros((n_pixels, n_rounds, n_channels), dtype=torch.float32)
    # Pixel 0 should pass score for first gene.
    residual_colours[0, 0, 0] = 1
    # Pixel 1 will contain high scores for two genes, expecting first to be selected.
    residual_colours[1, 0, 0] = 2
    residual_colours[1, 0, 1] = 2
    # # Pixel 2 will contain high scores for all genes, expecting it to fail selection.
    # residual_colours[2, 0] = 1
    # residual_colours[2, 1] = 1
    # residual_colours[2, 2] = 1
    # residual_colours[2, 3] = 1
    # Pixel 3 contains no intensity, expecting to fail selection.
    # Pixel 4 scores in a gene on the fail list, expecting to fail selection.
    residual_colours[4, 0, 4] = 0.6
    # Pixel 5 scores on fail gene, scores higher on second gene, expecting it to pass.
    residual_colours[5, 0, 1] = 0.7
    residual_colours[5, 0, 4] = 0.6

    all_bled_codes = torch.zeros((4, n_rounds, n_channels), dtype=torch.float32)
    all_bled_codes[0, 0, 0] = 1
    all_bled_codes[1, 0, 1] = 1
    all_bled_codes[2, 0, 2] = 1 / torch.sqrt(torch.tensor(2))
    all_bled_codes[2, 0, 3] = 1 / torch.sqrt(torch.tensor(2))
    all_bled_codes[3, 0, 4] = 1

    fail_gene_indices = torch.ones((n_pixels, 1), dtype=torch.int32)
    fail_gene_indices[:, 0] = 3
    dot_product_threshold = 0.5

    residual_colours_previous = residual_colours.detach().clone()
    all_bled_codes_previous = all_bled_codes.detach().clone()
    fail_gene_indices_previous = fail_gene_indices.detach().clone()
    kwargs = dict(
        residual_colours=residual_colours,
        all_bled_codes=all_bled_codes,
        fail_gene_indices=fail_gene_indices,
        dot_product_threshold=dot_product_threshold,
        minimum_intensity=0.0,
    )
    omp_solver = PixelScoreSolver()
    best_genes = omp_solver.get_next_gene_assignments(**kwargs)
    assert type(best_genes) is tuple
    assert len(best_genes) == 1
    best_genes = best_genes[0]
    kwargs["return_all_scores"] = True
    other_result = omp_solver.get_next_gene_assignments(**kwargs)
    assert type(other_result) is tuple
    assert len(other_result) == 2
    assert all([type(result) is torch.Tensor for result in other_result])
    assert type(best_genes) is torch.Tensor
    assert best_genes.shape == (n_pixels,), f"Got shape {best_genes.shape}"
    assert best_genes[0] == 0, f"Got {best_genes[0]}"
    assert best_genes[1] == 0
    assert best_genes[2] == omp_solver.NO_GENE_ASSIGNMENT
    assert best_genes[3] == omp_solver.NO_GENE_ASSIGNMENT
    assert best_genes[4] == omp_solver.NO_GENE_ASSIGNMENT
    assert best_genes[5] == 1
    # Since tensors are mutable, check that the parameter tensors have not changed.
    assert torch.allclose(residual_colours_previous, residual_colours)
    assert torch.allclose(all_bled_codes_previous, all_bled_codes)
    assert torch.allclose(fail_gene_indices_previous, fail_gene_indices)


def test_get_next_residual_colours() -> None:
    n_pixels = 1
    n_genes_added = 2
    n_rounds_channels_use = 3
    pixel_colours = torch.zeros((n_pixels, n_rounds_channels_use, 1)).float()
    pixel_colours[0, 0, 0] = 1
    pixel_colours[0, 1, 0] = 0.5
    pixel_colours[0, 2, 0] = 0
    bled_codes = torch.zeros((n_pixels, n_rounds_channels_use, n_genes_added)).float()
    bled_codes[0, 0, 0] = 1
    bled_codes[0, 1, 0] = 1
    bled_codes[0, 2, 0] = 0
    bled_codes[0, 0, 1] = 0
    bled_codes[0, 1, 1] = 1
    bled_codes[0, 2, 1] = 1
    alpha = 2.0
    beta = 1.0
    pixel_colours_copy = pixel_colours.detach().clone()
    bled_codes_copy = bled_codes.detach().clone()
    solver = PixelScoreSolver()
    results = solver.get_next_gene_weights(pixel_colours, bled_codes, alpha, beta)
    assert type(results) is tuple
    assert len(results) == 3
    assert all([type(r) is torch.Tensor for r in results])
    residuals, epsilon_squared, weights = results
    assert residuals.ndim == 2
    assert residuals.shape == (n_pixels, n_rounds_channels_use)
    assert epsilon_squared.ndim == 2
    assert epsilon_squared.shape == (n_pixels, n_rounds_channels_use)
    assert weights.ndim == 2
    assert weights.shape == (n_pixels, n_genes_added)
    # Check that the input tensors are left unchanged.
    assert torch.allclose(pixel_colours, pixel_colours_copy)
    assert torch.allclose(bled_codes, bled_codes_copy)
    # Check against calculations done by hand.
    expected_residuals = torch.ones(3).float()
    expected_residuals[1] = -1
    expected_residuals /= 6
    assert torch.allclose(residuals[0], expected_residuals)
    expected_weights = torch.ones(2).float()
    expected_weights[0] = 5 / 6
    expected_weights[1] = -1 / 6
    assert torch.allclose(weights, expected_weights)
    expected_epsilon_squared = torch.ones(3).float()
    expected_epsilon_squared[0] = 0.707475
    expected_epsilon_squared[1] = 0.691396
    expected_epsilon_squared[2] = 1.601128
    assert torch.allclose(epsilon_squared[0], expected_epsilon_squared)


def test_get_gene_pixel_scores() -> None:
    n_pixels = 2
    n_rounds_use = 3
    n_channels_use = 4
    n_genes_assigned = 2
    pixel_colours = np.zeros((n_pixels, n_rounds_use, n_channels_use), np.float32)
    pixel_colours[0, 0] = [1, 2, 0, 2]
    pixel_colours[0, 1] = [2, 0, 0, 0]
    pixel_colours[0, 2] = [2, 1, 2, 0]
    pixel_colours[1, 0] = [2, 2, 1, 0]
    pixel_colours[1, 1] = [2, 2, 0, 1]
    pixel_colours[1, 2] = [1, 0, 0, 1]

    weighted_bled_codes = np.zeros((n_pixels, n_genes_assigned, n_rounds_use, n_channels_use), np.float32)
    weighted_bled_codes[0, 0, 0] = [0, 1, 0, 0]
    weighted_bled_codes[0, 0, 1] = [1, 0, 0, 0]
    weighted_bled_codes[0, 0, 2] = [2, 2, 1, 0]
    weighted_bled_codes[0, 1, 0] = [2, 1, 0, 2]
    weighted_bled_codes[0, 1, 1] = [0.2, 0.2, 0, 0.1]
    weighted_bled_codes[0, 1, 2] = [0, 0, 0, 0.1]

    weighted_bled_codes[1, 0, 0] = [0, 2, 0, 0]
    weighted_bled_codes[1, 0, 1] = [2, 1, 0, 0]
    weighted_bled_codes[1, 0, 2] = [1, 0, 0, 0]
    weighted_bled_codes[1, 1, 0] = [2, 0, 0.5, 0]
    weighted_bled_codes[1, 1, 1] = [0, 1, 0, 0.5]
    weighted_bled_codes[1, 1, 2] = [0, 0, 1, 1]

    weights = np.zeros((n_pixels, n_genes_assigned), np.float32)
    weights = np.linalg.norm(weighted_bled_codes, axis=(-1, -2))

    bled_codes = np.zeros_like(weighted_bled_codes, np.float32)
    bled_codes = weighted_bled_codes.copy() / weights[:, :, np.newaxis, np.newaxis]

    pixel_colours = torch.from_numpy(pixel_colours)
    weights = torch.from_numpy(weights)
    bled_codes = torch.from_numpy(bled_codes)
    pixel_colours_copy = pixel_colours.detach().clone()
    weights_copy = weights.detach().clone()
    bled_codes_copy = bled_codes.detach().clone()

    solver = PixelScoreSolver()
    pixel_scores = solver.get_gene_pixel_scores(pixel_colours, bled_codes, weights, 0.0, 2.0)
    assert type(pixel_scores) is tuple
    assert len(pixel_scores) == 1
    pixel_scores = pixel_scores[0]
    assert pixel_scores.ndim == 2
    assert pixel_scores.shape == (n_pixels, n_genes_assigned)
    assert torch.allclose(pixel_colours, pixel_colours_copy)
    assert torch.allclose(bled_codes, bled_codes_copy)
    assert torch.allclose(weights, weights_copy)
    assert (pixel_scores >= 0).all()

    # Check against calculations done by hand.
    assert torch.isclose(pixel_scores[0, 0], torch.tensor(0.8626247925).float())

    # TODO: Check when alpha and beta are both non-zero.


def test_get_uncertainty_weights() -> None:
    n_batches = 2
    n_pixels = 3
    n_genes_assigned = 4
    n_rounds_channels_use = 5

    rng = np.random.RandomState(0)

    gene_weights = rng.rand(n_batches, n_pixels, n_genes_assigned).astype(np.float32)
    gene_weights = torch.from_numpy(gene_weights)
    bled_codes = rng.rand(n_batches, n_pixels, n_rounds_channels_use, n_genes_assigned).astype(np.float32)
    bled_codes = torch.from_numpy(bled_codes)
    alpha = 1.1
    beta = 2.3

    gene_weights_copy = gene_weights.detach().clone()
    bled_codes_copy = bled_codes.detach().clone()

    solver = PixelScoreSolver()
    epsilon_squared = solver.get_uncertainty_weights(gene_weights, bled_codes, alpha, beta)
    assert type(epsilon_squared) is torch.Tensor
    assert epsilon_squared.shape == (n_batches, n_pixels, n_rounds_channels_use)
    assert torch.allclose(gene_weights, gene_weights_copy)
    assert torch.allclose(bled_codes, bled_codes_copy)

    for b in range(n_batches):
        for p in range(n_pixels):
            # We require sigma squared for every round/channel pair for each epsilon squared computation.
            sigma_squared_values = []
            for rc in range(n_rounds_channels_use):
                sigma_squared = beta**2 + alpha * torch.square(gene_weights[b, p] * bled_codes[b, p, rc]).sum()
                sigma_squared_values.append(sigma_squared)
            sigma_squared_values = torch.tensor(sigma_squared_values)
            sigma_squared_values = torch.reciprocal(sigma_squared_values)

            # Now compute epsilon squared for each round/channel pair and check the function's values are correct.
            for rc in range(n_rounds_channels_use):
                epsilon_squared_expected = n_rounds_channels_use * sigma_squared_values[rc] / sigma_squared_values.sum()
                assert torch.isclose(epsilon_squared[b, p, rc], epsilon_squared_expected)
