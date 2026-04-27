import os

import numpy as np
import torch


def score(
    residual_spot_colours: torch.Tensor,
    assigned_bled_codes: torch.Tensor,
    bled_codes: torch.Tensor,
    gene_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Score each spot to each gene.

    Args:
        residual_spot_colours (`(n_spots x n_genes_assigned x n_rounds_use x n_channels_use) tensor[float]`): the spot
            colours for each gene assigned when all other gene assignment colours are subtracted.
        assigned_bled_codes (`(n_spots x n_genes_assigned x n_rounds_use x n_channels_use) tensor[float]`): normalised
            bled codes assigned to each spot. Each batch of bled_codes is scored on the spot colours. n_spots > 1 if the
            bled codes are different for each spot colour.
        bled_codes (`n_genes x n_rounds_use x n_channels_use) tensor[float]`): every gene bled code.
        gene_indices (`(n_spots x n_genes_assigned) tensor[float]`): gene indices for every spot.

    Returns:
        (`(n_spots x n_genes_assigned) tensor[float]`): score. score such that `score[d, c]` gives a "round dot product"
            between `spot_colours` vector `d` with `bled_codes` vector `c`.

    Notes:
        - If n_spots is 1 for assigned_bled_codes, then the bled codes will be repeated for all spots in spot_colours.
    """
    assert type(residual_spot_colours) is torch.Tensor
    assert type(assigned_bled_codes) is torch.Tensor
    assert residual_spot_colours.ndim == 3
    assert assigned_bled_codes.ndim == 4
    assert assigned_bled_codes.shape[0] == 1 or assigned_bled_codes.shape[0] == residual_spot_colours.shape[0]

    if assigned_bled_codes.shape[0] == 1:
        assigned_bled_codes = torch.repeat_interleave(assigned_bled_codes, residual_spot_colours.shape[0], 0)

    n_spots = residual_spot_colours.shape[0]
    n_genes_assigned = residual_spot_colours.shape[1]

    # Combine rounds/channels.
    residual_spot_colours = residual_spot_colours.reshape(
        (residual_spot_colours.shape[0], residual_spot_colours.shape[1], -1)
    )
    assigned_bled_codes = assigned_bled_codes.reshape((assigned_bled_codes.shape[0], assigned_bled_codes.shape[1], -1))

    # Becomes shape (n_spots * n_genes_assigned x 1 x n_rounds_use * n_channels_use).
    residual_spot_colours = residual_spot_colours.reshape((-1, 1, residual_spot_colours.shape[-1]))

    # Becomes shape (n_spots * n_genes_assigned x 1 x n_rounds_use * n_channels_use).
    assigned_bled_codes = assigned_bled_codes.reshape((-1, 1, assigned_bled_codes.shape[-1]))

    # Create one Spearman score for each spot's assigned gene.
    # Has shape (n_spots x n_genes_assigned).
    spears = spearman_score(residual_spot_colours, assigned_bled_codes).reshape((n_spots, n_genes_assigned))

    bled_codes = bled_codes.reshape((1, bled_codes.shape[0], -1))
    bled_codes = bled_codes.repeat_interleave(n_spots * n_genes_assigned, dim=0)
    # Produces a discriminality score for every spot colour on every bled code gene
    # Has shape (n_spots * n_genes_assigned x 1 x n_genes).
    spears_nobest = spearman_score(residual_spot_colours, bled_codes)
    # Change to shape (n_spots x n_genes_assigned x n_genes).
    spears_nobest = spears_nobest.reshape((n_spots, n_genes_assigned, bled_codes.shape[1]))

    # Exclude the assigned gene from spears_nobest and compute discriminality.
    discriminality = torch.full((n_spots, n_genes_assigned), torch.nan, dtype=residual_spot_colours.dtype)
    for i in range(n_genes_assigned):
        spears_nobest[np.arange(0, spears_nobest.shape[0]), i, gene_indices[:, i]] = np.nan
        discriminality[:, i] = (spears[:, i] - np.nanmean(spears_nobest[:, i], axis=1)) / np.nanstd(
            spears_nobest[:, i], axis=1
        )

    return discriminality

    # (n_genes x n_rounds_use x n_channels_use).
    # bc = nb.call_spots.bled_codes
    # (n_spots x n_rounds_use x n_channels_use).
    # col = coppafisher.omp.base.get_all_colours(nb.basic_info, nb.omp)[0]
    # Returns shape (n_spots x n_genes).
    # import spatiotemporal
    # spears = spatiotemporal.spearman(col.reshape(col.shape[0], -1), bc.reshape(bc.shape[0], -1))
    # spears_nobest = spears.copy()
    # spears_nobest[np.arange(0, spears.shape[0]), spot_data.gene_no] = np.nan
    # disc_score = (spears[np.arange(0, spears.shape[0]), spot_data.gene_no]-np.nanmean(spears_nobest, axis=1))/np.nanstd(spears_nobest, axis=1)


def spearman_score(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Score a batch of values using Spearman correlation scoring.

    Args:
        x (`(n_batch x k x n) tensor[float]`): first values.
        y (`(n_batch x m x n) tensor[float]`): second values.

    Returns:
        (`(n_batch x k x m) tensor[float]`): result. The Spearman correlation between x and y.
    """
    assert type(x) is torch.Tensor
    assert type(y) is torch.Tensor
    assert x.ndim == 3
    assert y.ndim == 3
    assert x.shape[0] == y.shape[0]
    assert x.shape[2] == y.shape[2]

    def _spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.ndim == 3
        assert b.ndim == 3

        # Allows for Tensor support in scipy functions.
        os.environ["SCIPY_ARRAY_API"] = "1"
        import scipy

        dtype = a.dtype
        a = scipy.stats.rankdata(a, axis=2)
        b = scipy.stats.rankdata(b, axis=2)
        result = (a - np.mean(a, axis=2, keepdims=True)) @ (b - np.mean(b, axis=2, keepdims=True)).transpose((0, 2, 1))
        result /= a.shape[2] * np.sqrt(np.var(a, axis=2)[:, :, None]) * np.sqrt(np.var(b, axis=2)[:, None, :])
        return torch.from_numpy(result).to(dtype)

    result = _spearman(x, y)
    result[torch.isnan(result)] = 0

    return result
