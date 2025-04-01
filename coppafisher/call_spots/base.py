import numpy as np
import scipy


def bayes_mean(
    spot_colours: np.ndarray, prior_colours: np.ndarray, conc_param_parallel: float, conc_param_perp: float
) -> np.ndarray:
    """
    This function computes the posterior mean of the spot colours under a prior distribution with mean prior_colours
    and covariance matrix given by a diagonal matrix with diagonal entry conc_param_parallel for the direction parallel
    to prior_colours and conc_param_perp for the direction orthogonal to prior_colours.

    Args:
        spot_colours: np.ndarray [n_spots x n_channels_use]
            The spot colours for each spot.
        prior_colours: np.ndarray [n_channels_use]
            The prior mean colours.
        conc_param_parallel: np.ndarray [n_channels_use]
            The concentration parameter for the direction parallel to prior_colours.
        conc_param_perp: np.ndarray [n_channels_use]
            The concentration parameter for the direction orthogonal to prior_colours.
    """
    n_spots, data_sum = len(spot_colours), np.sum(spot_colours, axis=0)
    # deal with the case where there are no spots
    if n_spots == 0:
        return prior_colours

    prior_direction = prior_colours / np.linalg.norm(prior_colours)  # normalized prior direction
    sum_parallel = np.dot(data_sum, prior_direction) * prior_direction  # projection of data sum along prior direction
    sum_perp = data_sum - sum_parallel  # projection of data sum orthogonal to mean direction

    # now compute the weighted sum of the posterior mean for parallel and perpendicular directions
    posterior_parallel = (sum_parallel + conc_param_parallel * prior_direction) / (n_spots + conc_param_parallel)
    posterior_perp = sum_perp / (n_spots + conc_param_perp)
    return posterior_parallel + posterior_perp


def compute_bleed_matrix(
    spot_colours: np.ndarray, gene_no: np.ndarray, gene_codes: np.ndarray, n_dyes: int
) -> np.ndarray:
    """
    Function to compute the bleed matrix from the spot colours and the gene assignments.

    Args:
        spot_colours (`(n_spots x n_rounds x n_channels_use) ndarray`): the spot colours for each spot in each round and
            channel.
        gene_no (`(n_spots) ndarray`): the gene assignment for each spot.
        gene_codes (`(n_genes x n_rounds) ndarray`): the gene codes for each gene in each round.
        n_dyes (int): the number of dyes.

    Returns:
        (`(n_dyes x n_channels_use) ndarray`): bleed_matrix. The computed bleed matrix.
    """
    assert len(spot_colours) == len(gene_no), "Spot colours and gene_no must have the same length."

    _, n_rounds, n_channels_use = spot_colours.shape
    bleed_matrix = np.zeros((n_dyes, n_channels_use), np.float32)

    # Loop over all dyes, find the spots which are meant to be dye d in round r, and compute the SVD.
    for d in range(n_dyes):
        dye_d_colours = []
        for r in range(n_rounds):
            relevant_genes = np.where(gene_codes[:, r] == d)[0]
            relevant_gene_mask = np.isin(gene_no, relevant_genes)
            dye_d_colours.append(spot_colours[relevant_gene_mask, r, :])
        # All the good colours for dye d in shape (n_spots_found x n_channels_use).
        dye_d_colours = np.concatenate(dye_d_colours, axis=0)
        # Compute the SVD.
        _, _, v = scipy.sparse.linalg.svds(dye_d_colours, k=1)
        v = v[0]
        # Ensure the largest entry in v is positive.
        v *= np.sign(v[np.argmax(np.abs(v))])
        bleed_matrix[d] = v

    return bleed_matrix
