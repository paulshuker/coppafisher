import math as maths

import numpy as np
import torch

from coppafisher.call_spots import dot_product


def test_dot_product():
    n_spots, n_rounds, n_channels_use, n_genes = 1, 3, 3, 2
    n_batches = 1
    spot_colours = np.zeros((n_spots, n_rounds, n_channels_use), np.float32)
    spot_colours[0, 0] = [1, 1, 0]
    spot_colours[0, 1] = [0, 0, 1]
    spot_colours[0, 2] = [0, 1, 1]
    spot_colours = spot_colours[np.newaxis]
    bled_codes = np.zeros((n_genes, n_rounds, n_channels_use))
    bled_codes[0, 0] = [np.sqrt(0.6), np.sqrt(0.4), 0]
    bled_codes[0, 1] = [0, np.sqrt(0.1), np.sqrt(0.9)]
    bled_codes[0, 2] = [0, np.sqrt(0.5), np.sqrt(0.5)]
    bled_codes[1, 0] = [0, -1, 0]
    bled_codes[1, 1] = [0, 0, 1]
    bled_codes[1, 2] = [1, 0, 0]
    bled_codes = bled_codes[np.newaxis, np.newaxis]
    spot_colours_copy = spot_colours.copy()
    bled_codes_copy = bled_codes.copy()
    scores = dot_product.dot_product_score(spot_colours=spot_colours, bled_codes=bled_codes)
    assert type(scores) is np.ndarray
    assert scores.shape == (n_batches, n_spots, n_genes)
    assert np.allclose(scores[0, 0, 0], 0.98120648)
    assert np.allclose(scores[0, 0, 1], 0.0976310729)
    assert np.allclose(spot_colours, spot_colours_copy)
    assert np.allclose(bled_codes, bled_codes_copy)

    # Check with torch Tensor inputs.
    spot_colours_torch = torch.from_numpy(spot_colours)
    bled_codes_torch = torch.from_numpy(bled_codes)
    spot_colours_torch_copy = spot_colours_torch.detach().clone()
    bled_codes_torch_copy = bled_codes_torch.detach().clone()
    scores_torch = dot_product.dot_product_score(spot_colours=spot_colours_torch, bled_codes=bled_codes_torch)
    assert type(scores_torch) is torch.Tensor
    assert scores_torch.shape == (n_batches, n_spots, n_genes)
    assert maths.isclose(scores_torch[0, 0, 0].item(), 0.98120648, abs_tol=5e-9)
    assert maths.isclose(scores_torch[0, 0, 1].item(), 0.09763107, abs_tol=5e-9)
    assert torch.allclose(spot_colours_torch, spot_colours_torch_copy)
    assert torch.allclose(bled_codes_torch, bled_codes_torch_copy)

    # Ensure the batching dimension is working.
    bled_codes = bled_codes.repeat(3, 0)
    scores_2 = dot_product.dot_product_score(spot_colours, bled_codes)
    assert scores_2.shape[:2] == (3, 1)
    assert scores_2.ndim == 3
    assert np.allclose(scores_2[0], scores_2[1])
    assert np.allclose(scores_2[0], scores_2[2])
    assert np.allclose(scores, scores_2[0, 0])
