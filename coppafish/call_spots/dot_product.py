import numpy as np
import torch

from .. import log


def dot_product_score(
    spot_colours: np.ndarray | torch.Tensor,
    bled_codes: np.ndarray | torch.Tensor,
    intensity_threshold: float,
    dot_product_weight: float,
) -> np.ndarray | torch.Tensor:
    """
    Score each spot to each gene. The maximum score is the assigned gene for said spot. The scores range from 0 to 1.

    Args:
        spot_colours (`(n_spots x n_rounds x n_channels_use) ndarray[float]`): spot colours after call spots scaling
            has been applied. They must not be L2 normalised yet.
        bled_codes (`(n_genes x n_rounds x n_channels_use) ndarray[float]`): normalised bled codes.
        intensity_threshold (float): a pixel's score is set to zero if x < intensity_threshold, where x is
            abs(y).min(over rounds).max(over channels) is the the weakest round intensity, where y is spot_colours
            divided by its total L2 norm for each pixel. At zero, there is no intensity threshold. The threshold is
            important to cut out pixels/spots with missing round(s). There can be missing rounds due to different
            reasons: 1) Trying to score dim noise. 2) Registration mistakes. 3) Experimental issues.
        dot_product_weight (float): how much the dot product score is a pure dot product score. 1 is a simple dot
            product score, 0 gives each round an equal weighting on the score. A value between 0 and 1 is somewhere in
            between the two.

    Returns:
        (`(n_spots x n_genes) ndarray[float32] or tensor[float32]`): score. score such that `score[d, c]` gives a
            "semi-dot product" between `spot_colours` vector `d` with `bled_codes` vector `c`. Returns a tensor if
            spot_colours is a tensor.
    """
    assert type(spot_colours) in (np.ndarray, torch.Tensor)
    assert type(bled_codes) in (np.ndarray, torch.Tensor)
    assert type(intensity_threshold) is float
    assert intensity_threshold >= 0 and intensity_threshold <= 1
    if dot_product_weight < 0 or dot_product_weight > 1:
        raise ValueError(f"dot_product_weight must be between 0 and 1, got {dot_product_weight}")

    if type(spot_colours) is np.ndarray:
        spot_colours_torch = torch.from_numpy(spot_colours)
    else:
        spot_colours_torch = spot_colours.detach().clone()
    if type(bled_codes) is np.ndarray:
        bled_codes_torch = torch.from_numpy(bled_codes)
    else:
        bled_codes_torch = bled_codes.detach().clone()
    spot_colours_torch = spot_colours_torch.float()
    bled_codes_torch = bled_codes_torch.float()
    assert spot_colours_torch.ndim == bled_codes_torch.ndim == 3
    assert spot_colours_torch.shape[1:] == bled_codes_torch.shape[1:]

    worst_intensity = spot_colours_torch.abs().max(2).values.min(1).values
    is_dim = worst_intensity < intensity_threshold
    log.debug(f"Found {is_dim.sum()}/{is_dim.size} weak spots")
    del worst_intensity

    # L2 normalise the entire spot colour for each spot.
    spot_colours_torch /= torch.linalg.matrix_norm(spot_colours_torch, keepdim=True)
    bled_codes_torch /= torch.linalg.matrix_norm(bled_codes_torch, keepdim=True)

    # L2 norms of the spot colour and bled code for each round separately.
    spot_colours_norms = torch.linalg.vector_norm(spot_colours_torch, dim=2)
    bled_code_norms = torch.linalg.vector_norm(bled_codes_torch, dim=2)
    # F hat as defined in the docs, has shape (n_spots x n_rounds x n_channels).
    f_hat = spot_colours_torch.detach().clone()
    del spot_colours_torch
    f_hat /= spot_colours_norms[:, :, np.newaxis]
    # K hat as defined in the docs, has shape (n_genes x n_rounds x n_channels).
    k_hat = bled_codes_torch.detach().clone()
    del bled_codes_torch
    k_hat /= bled_code_norms[:, :, np.newaxis]

    # Has shape (n_spots x n_genes x n_rounds).
    scores = (spot_colours_norms[:, np.newaxis] * bled_code_norms[np.newaxis]) ** dot_product_weight
    scores *= (f_hat[:, np.newaxis] * k_hat[np.newaxis]).sum(3)
    del f_hat, k_hat
    # Has shape (n_spots x n_genes).
    scores = scores.sum(2)
    scores /= (spot_colours_norms ** (2 * dot_product_weight)).sum(1).sqrt()[:, np.newaxis]
    del spot_colours_norms
    scores /= (bled_code_norms ** (2 * dot_product_weight)).sum(1).sqrt()[np.newaxis]
    del bled_code_norms
    scores[is_dim] = 0
    scores = scores.clip(0, 1)

    if type(spot_colours) is np.ndarray:
        scores = scores.numpy()

    return scores


def gene_prob_score(spot_colours: np.ndarray, bled_codes: np.ndarray, kappa: float = 2) -> np.ndarray:
    """
    Probability model says that for each spot in a particular round, the normalised fluorescence vector follows a
    Von-Mises Fisher distribution with mean equal to the normalised fluorescence for each dye and concentration
    parameter kappa. Then invert this to get prob(dye | fluorescence) and multiply across rounds to get
    prob(gene | spot_colours).

    Args:
        spot_colours (`(n_spots x n_rounds x n_channels_use) ndarray`): spot colours.
        bled_codes (`(n_genes x n_rounds x n_channels_use) ndarray`): normalised bled codes.
        kappa (float, optional), scaling factor for dot product score. Default: 2.

    Returns:
        (`(n_spots x n_genes) ndarray[float]`): gene probabilities.
    """
    n_genes = bled_codes.shape[0]
    n_spots, n_rounds, n_channels_use = spot_colours.shape
    # First, normalise spot_colours so that for each spot s and round r, norm(spot_colours[s, r, :]) = 1
    spot_colours = spot_colours / np.linalg.norm(spot_colours, axis=2)[:, :, None]
    # Do the same for bled_codes
    bled_codes = bled_codes / np.linalg.norm(bled_codes, axis=2)[:, :, None]
    # Flip the sign of a single spot and round if spot_colours[s, r, c] < 0 for the greatest magnitude channel.
    spot_colours_reshaped = spot_colours.copy().reshape((-1, n_channels_use))
    negatives = np.take_along_axis(spot_colours_reshaped, np.argmax(spot_colours_reshaped, axis=1)[:, None], 1) < 0
    spot_colours[negatives.reshape((n_spots, n_rounds))] *= -1
    # At this point, reshape spot_colours to be [n_spots, n_rounds * n_channels_use] and bled_codes to be
    # [n_genes, n_rounds * n_channels_use]
    spot_colours = spot_colours.reshape((n_spots, -1))
    bled_codes = bled_codes.reshape((n_genes, -1))
    # Now we can compute the dot products of each spot with each gene, producing a matrix of shape [n_spots, n_genes]
    dot_product = spot_colours @ bled_codes.T
    probability = np.exp(kappa * dot_product)
    # Now normalise so that each row sums to 1
    probability = np.nan_to_num(probability / np.sum(probability, axis=1)[:, None])

    return probability
