import itertools

import numpy as np
import pytest

from coppafisher.omp.pixel_scores import PixelScoreSolver
from coppafisher.utils import base
from coppafisher.utils import intensity as utils_intensity


@pytest.mark.slow
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
    assert bg_codes.ndim == 3
    assert bg_codes.shape == (n_channels, n_rounds, n_channels)
    assert bg_codes.dtype == np.float32
    maximum_iterations = 4
    dot_product_threshold = 0.001
    minimum_intensity = 0.0
    background_subtract_percentile = 10.0
    alpha = 0.0
    beta = 1.0

    # Simple checks for consistent results and correct shapes.
    previous_result = None
    for return_all_scores, return_all_residuals, return_stopping_criteria in itertools.product((True, False), repeat=3):
        print(return_all_scores)
        print(return_all_residuals)
        print(return_stopping_criteria)
        print("")

        result = solver.solve(
            pixel_colours,
            bled_codes,
            bg_codes,
            maximum_iterations,
            dot_product_threshold,
            minimum_intensity,
            background_subtract_percentile,
            alpha,
            beta,
            return_all_scores=return_all_scores,
            return_all_residuals=return_all_residuals,
            return_stopping_criteria=return_stopping_criteria,
        )
        if return_all_scores:
            assert type(result[1]) is np.ndarray
            assert result[1].shape[0] >= 1
            assert result[1].shape[1:] == (n_pixels, n_genes + n_channels)
            assert result[1].dtype == dtype
            assert not np.isnan(result[1]).any()
            assert (result[1] >= 0).all()
        if return_all_residuals:
            assert type(result[1 + int(return_all_scores)]) is np.ndarray
            assert result[1 + int(return_all_scores)].shape == (n_pixels, n_genes, n_rounds, n_channels)
            assert result[1 + int(return_all_scores)].dtype == dtype
        if return_stopping_criteria:
            assert type(result[1 + int(return_all_scores) + int(return_all_residuals)]) is np.ndarray
            assert result[1 + int(return_all_scores) + int(return_all_residuals)].shape == (n_pixels,)
            assert result[1 + int(return_all_scores) + int(return_all_residuals)].dtype == np.int8
        if type(result) is tuple:
            result = result[0]
        assert type(result) is np.ndarray
        assert result.shape == (n_pixels, n_genes)
        assert result.dtype == dtype
        if previous_result is not None:
            assert np.allclose(result, previous_result), f"{np.abs(result - previous_result).max()}"
        previous_result = result

    # Ensure the number of assigned genes only decreases as the dot product threshold increases.
    previous_n_genes_assigned = n_pixels * n_genes + 1
    for dp_threshold in [dot_product_threshold + 0.001 * i for i in range(1, 100)] + [10.0]:
        result = solver.solve(
            pixel_colours,
            bled_codes,
            bg_codes,
            maximum_iterations,
            dp_threshold,
            minimum_intensity,
            background_subtract_percentile,
            alpha,
            beta,
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
    assert bg_codes.dtype == np.float32
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
        pixel_colours,
        bled_codes,
        bg_codes,
        maximum_iterations,
        dot_product_threshold,
        minimum_intensity,
        background_subtract_percentile,
        alpha,
        beta,
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
        pixel_colours,
        bled_codes,
        bg_codes,
        maximum_iterations,
        dot_product_threshold,
        minimum_intensity,
        background_subtract_percentile,
        alpha,
        beta,
    )
    for p, dim in enumerate(dim_gene_assignments):
        if dim:
            assert (~np.isclose(result[p], 0)).sum() == 0
            continue
        assert (~np.isclose(result[p], 0)).sum() == len(expected_gene_assignments[p])
        for g in expected_gene_assignments[p]:
            assert (result[p] > 0)[g]


def test_get_next_gene_assignments() -> None:
    import torch

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

    gene_bled_codes = torch.zeros((4, n_rounds, n_channels), dtype=torch.float32)
    gene_bled_codes[0, 0, 0] = 1
    gene_bled_codes[1, 0, 1] = 1
    gene_bled_codes[2, 0, 2] = 1 / torch.sqrt(torch.tensor(2))
    gene_bled_codes[2, 0, 3] = 1 / torch.sqrt(torch.tensor(2))
    gene_bled_codes[3, 0, 4] = 1

    fail_gene_indices = torch.ones((n_pixels, 1), dtype=torch.int32)
    fail_gene_indices[:, 0] = 3
    dot_product_threshold = 0.5

    residual_colours_previous = residual_colours.detach().clone()
    gene_bled_codes_previous = gene_bled_codes.detach().clone()
    fail_gene_indices_previous = fail_gene_indices.detach().clone()
    kwargs = dict(
        residual_colours=residual_colours,
        gene_bled_codes=gene_bled_codes,
        fail_gene_indices=fail_gene_indices,
        dot_product_threshold=dot_product_threshold,
        minimum_intensity=0.0,
        bg_subtraction_percentile=25.0,
    )
    omp_solver = PixelScoreSolver()
    best_genes = omp_solver.get_next_gene_assignments(**kwargs)
    assert type(best_genes) is tuple
    assert len(best_genes) == 1
    best_genes = best_genes[0]
    kwargs["return_all_scores"] = True
    kwargs["return_stopping_criteria"] = True
    other_result = omp_solver.get_next_gene_assignments(**kwargs)
    assert type(other_result) is tuple
    assert len(other_result) == 3
    assert all(type(result) is torch.Tensor for result in other_result)
    assert not any(torch.isnan(result).any() for result in other_result)
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
    assert torch.allclose(gene_bled_codes_previous, gene_bled_codes)
    assert torch.allclose(fail_gene_indices_previous, fail_gene_indices)


def test_get_next_gene_weights() -> None:
    import torch

    rng = np.random.RandomState(0)
    n_pixels = 6
    n_genes_added = 2
    n_rounds_use = 3
    n_channels_use = 2

    pixel_colours = torch.from_numpy(rng.rand(n_pixels, n_rounds_use, n_channels_use, 1)).float()
    bled_codes = torch.from_numpy(rng.rand(n_pixels, n_rounds_use, n_channels_use, n_genes_added)).float()
    alpha = 2.0
    beta = 1.0
    pixel_colours_copy = pixel_colours.detach().clone()
    bled_codes_copy = bled_codes.detach().clone()

    solver = PixelScoreSolver()
    results = solver.get_next_gene_weights(pixel_colours, bled_codes, alpha, beta)
    assert type(results) is tuple
    assert len(results) == 3
    assert all(type(r) is torch.Tensor for r in results)
    residuals, epsilon_squared, weights = results
    assert residuals.ndim == 3
    assert residuals.shape == (n_pixels, n_rounds_use, n_channels_use)
    assert epsilon_squared.ndim == 3
    assert epsilon_squared.shape == (n_pixels, n_rounds_use, n_channels_use)
    assert (epsilon_squared >= 0).all()
    assert weights.ndim == 3
    assert weights.shape == (n_pixels, n_rounds_use, n_genes_added)
    # Check that the input tensors are left unchanged.
    assert torch.allclose(pixel_colours, pixel_colours_copy)
    assert torch.allclose(bled_codes, bled_codes_copy)

    # Check residuals and weights.
    for p in range(n_pixels):
        for r in range(n_rounds_use):
            # Has shape n_channels_use x 1.
            pixel_colour_r = pixel_colours[p, r]
            assert pixel_colour_r.ndim == 2
            # Has shape n_channels_use x n_genes_added.
            bled_codes_r = bled_codes[p, r]
            assert bled_codes_r.ndim == 2
            # Has shape n_genes_added.
            expected_weight_r = torch.linalg.lstsq(bled_codes_r, pixel_colour_r).solution[:, 0]
            assert expected_weight_r.shape == (n_genes_added,)
            assert weights[p, r].shape == expected_weight_r.shape
            assert torch.allclose(weights[p, r], expected_weight_r, atol=1e-4)

            expected_residual_r = pixel_colour_r[:, 0] - (expected_weight_r[np.newaxis] * bled_codes_r).sum(1)
            assert expected_residual_r.shape == (n_channels_use,)
            assert torch.allclose(residuals[p, r], expected_residual_r, atol=1e-4)


def test_get_gene_pixel_scores() -> None:
    import torch

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

    weights = np.zeros((n_pixels, n_genes_assigned, n_rounds_use), np.float32)
    weights = np.linalg.norm(weighted_bled_codes, axis=-1)

    bled_codes = np.zeros_like(weighted_bled_codes, np.float32)
    bled_codes = weighted_bled_codes.copy() / weights[:, :, :, np.newaxis]

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
    import torch

    n_batches = 2
    n_pixels = 3
    n_genes_assigned = 4
    n_rounds_use = 5
    n_channels_use = 5

    rng = np.random.RandomState(0)

    gene_weights = rng.rand(n_batches, n_pixels, n_rounds_use, n_genes_assigned).astype(np.float32)
    gene_weights = torch.from_numpy(gene_weights)
    bled_codes = rng.rand(n_batches, n_pixels, n_rounds_use, n_channels_use, n_genes_assigned).astype(np.float32)
    bled_codes = torch.from_numpy(bled_codes)
    alpha = 1.1
    beta = 2.3

    gene_weights_copy = gene_weights.detach().clone()
    bled_codes_copy = bled_codes.detach().clone()

    solver = PixelScoreSolver()
    epsilon_squared = solver._get_uncertainty_weights(gene_weights, bled_codes, alpha, beta)
    assert type(epsilon_squared) is torch.Tensor
    assert epsilon_squared.shape == (n_batches, n_pixels, n_rounds_use, n_channels_use)
    assert torch.allclose(gene_weights, gene_weights_copy)
    assert torch.allclose(bled_codes, bled_codes_copy)

    for b in range(n_batches):
        for p in range(n_pixels):
            # We require sigma squared for every round/channel pair for each epsilon squared computation.
            sigma_squared_values = torch.full((n_rounds_use, n_channels_use), torch.nan, dtype=torch.float32)
            for r in range(n_rounds_use):
                for c in range(n_channels_use):
                    sigma_squared = beta**2 + alpha * torch.square(gene_weights[b, p, r] * bled_codes[b, p, r, c]).sum()
                    sigma_squared_values[r, c] = sigma_squared
            sigma_squared_values = torch.reciprocal(sigma_squared_values)

            # Now compute epsilon squared for each round/channel pair and check the function's values are correct.
            for r in range(n_rounds_use):
                for c in range(n_channels_use):
                    epsilon_squared_expected = n_rounds_use * n_channels_use * sigma_squared_values[r, c]
                    epsilon_squared_expected /= sigma_squared_values.sum()
                    assert torch.isclose(epsilon_squared[b, p, r, c], epsilon_squared_expected)
