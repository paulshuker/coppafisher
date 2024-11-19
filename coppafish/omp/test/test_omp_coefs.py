import torch

from coppafish.omp import coefs


# def test_compute_omp_coefficients() -> None:
#     n_pixels = 7
#     n_genes = 2
#     n_rounds_use = 4
#     n_channels_use = 3
#     bled_codes = np.zeros((n_genes, n_rounds_use, n_channels_use), np.float32)
#     bled_codes[0, 0, 0] = 1
#     bled_codes[0, 1, 1] = 1
#     bled_codes[1, 0, 1] = 1
#     bled_codes[1, 1, 0] = 1
#     bled_codes[1, 2, 2] = 1
#     pixel_colours = np.zeros((n_pixels, n_rounds_use, n_channels_use), np.float16)
#     # Pixel 0 has no intensity, expecting zero coefficients.
#     pixel_colours[0] = 0
#     # Pixel 1 has only background, expecting zero coefficients.
#     pixel_colours[1, 1] = 2
#     # Pixel 2 has a one strong gene expression weak background.
#     pixel_colours[2] = 1.2 * bled_codes[1]
#     pixel_colours[2, 0] += 0.02
#     pixel_colours[2, 1] += 0.03
#     # Pixel 3 has a weak gene expression and strong background.
#     pixel_colours[3] = 0.5 * bled_codes[0]
#     pixel_colours[3, 0] += 2
#     # Pixel 4 has a weak gene expression below the normalisation_shift.
#     pixel_colours[4] = 0.005 * bled_codes[0]
#     # Pixel 5 has a weak and strong gene expression.
#     pixel_colours[5] = 0.1 * bled_codes[0] + 2.0 * bled_codes[1]
#     # Pixel 6 has both strong gene expressions.
#     pixel_colours[6] = 1.4 * bled_codes[0] + 2.0 * bled_codes[1]
#     background_codes = np.zeros((n_channels_use, n_rounds_use, n_channels_use), np.float32)
#     background_codes[0, 0] = 1
#     background_codes[1, 1] = 1
#     background_codes[2, 2] = 1
#     colour_norm_factor = np.ones((1, n_rounds_use, n_channels_use), np.float32)
#     colour_norm_factor[0, 3, 2] = 0.1
#     maximum_iterations = 4
#     dot_product_threshold = 0.5
#     normalisation_shift = 0.03
#
#     pixel_colours_previous = pixel_colours.copy()
#     bled_codes_previous = bled_codes.copy()
#     background_codes_previous = background_codes.copy()
#     colour_norm_factor_previous = colour_norm_factor.copy()
#     omp_solver = coefs.CoefficientSolverOMP()
#     coefficients = omp_solver.compute_omp_coefficients(
#         pixel_colours=pixel_colours,
#         bled_codes=bled_codes,
#         background_codes=background_codes,
#         colour_norm_factor=colour_norm_factor,
#         maximum_iterations=maximum_iterations,
#         dot_product_weight=1.0,
#         dot_product_threshold=dot_product_threshold,
#         normalisation_shift=normalisation_shift,
#     )
#     assert type(coefficients) is np.ndarray
#     assert np.allclose(pixel_colours_previous, pixel_colours)
#     assert np.allclose(bled_codes_previous, bled_codes)
#     assert np.allclose(background_codes_previous, background_codes)
#     assert np.allclose(colour_norm_factor_previous, colour_norm_factor)
#     assert coefficients.shape == (n_pixels, n_genes)
#     abs_tol = 0.01
#     assert np.allclose(coefficients[0], 0)
#     assert np.allclose(coefficients[1], 0)
#     assert np.allclose(coefficients[2, 0], 0)
#     # TODO: Create solid coefficient compute assertions for this unit test. The bled codes and pixel colours are L2
#     # normalised now during semi dot product scoring, so this needs updating.


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
        epsilon_squared=torch.ones_like(residual_colours, dtype=torch.float32),
        fail_gene_indices=fail_gene_indices,
        dot_product_threshold=dot_product_threshold,
        minimum_intensity=0.0,
    )
    omp_solver = coefs.CoefficientSolverOMP()
    best_genes, _ = omp_solver.get_next_gene_assignments(**kwargs)
    kwargs["return_all_scores"] = True
    other_result = omp_solver.get_next_gene_assignments(**kwargs)
    assert type(other_result) is tuple
    assert len(other_result) == 3
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
    n_pixels = 3
    n_genes_added = 2
    n_rounds_channels_use = 4
    pixel_colours = torch.zeros((n_pixels, n_rounds_channels_use, 1), dtype=torch.float32)
    pixel_colours[0, 0, 0] = 1.3
    pixel_colours[0, 1, 0] = 0.4
    pixel_colours[0, 2, 0] = 0.6
    # The second pixel will have a non-zero residual after genes are fitted.
    pixel_colours[1, 0, 0] = 1 * 0.4
    pixel_colours[1, 3, 0] = 4
    # The third pixel does not quite fit on the second gene.
    pixel_colours[2, 0, 0] = 1 * 0.74
    pixel_colours[2, 1, 0] = 0.4
    pixel_colours[2, 2, 0] = 0.8

    bled_codes = torch.zeros((n_pixels, n_rounds_channels_use, n_genes_added))
    bled_codes[:, 0, 0] = 1
    bled_codes[:, 1, 1] = 0.2
    bled_codes[:, 2, 1] = 0.3

    pixel_colours_previous = pixel_colours.detach().clone()
    bled_codes_previous = bled_codes.detach().clone()
    omp_solver = coefs.CoefficientSolverOMP()
    residuals, epsilon_squared = omp_solver.get_next_residual_colours(
        pixel_colours=pixel_colours,
        bled_codes=bled_codes,
        alpha=1.0,
        beta=120.0,
    )

    assert type(residuals) is torch.Tensor
    assert residuals.shape == (n_pixels, n_rounds_channels_use)
    abs_tol = 1e-6
    assert torch.allclose(residuals[0], torch.tensor(0).float(), atol=abs_tol)
    assert torch.isclose(residuals[1, 3], torch.tensor(4).float(), atol=abs_tol)
    assert torch.isclose(residuals[1], torch.tensor(0).float(), atol=abs_tol).sum() == (n_rounds_channels_use - 1)
    assert residuals[2, 1] < 0
    assert residuals[2, 2] > 0

    assert type(epsilon_squared) is torch.Tensor
    assert epsilon_squared.shape == (n_pixels, n_rounds_channels_use)
    assert (epsilon_squared <= 1.01).all()

    # Since tensors are mutable, check that the parameter tensors have not changed.
    assert torch.allclose(pixel_colours_previous, pixel_colours)
    assert torch.allclose(bled_codes_previous, bled_codes)
