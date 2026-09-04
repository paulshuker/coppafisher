import numpy as np

from ..call_spots import dot_product


def preprocess_colours(
    colours: np.ndarray,
    colour_norm_factor: np.ndarray,
    background_dot_product_threshold: float,
    background_subtract_percentile: float,
) -> np.ndarray:
    """
    Given colours are pre-processed as described in OMP method documentation (Step 0).

    First, colours are multiplied by the colour normalisation factors calculated during call spots.

    Second, the colours are round-dot product scored against all background genes. If the highest scoring background
    gene has a score at least background_dot_product_threshold, then the background_subtract_percentile'th percentile is
    subtracted from said channel in all rounds. This is repeated n_channels_use times until up to all background genes
    have been subtracted from each colour. A background gene cannot be subtracted twice from the same colour.

    Args:
        colours (`(n_colours x n_rounds_use x n_channels_use) ndarray[float32]`): the colours to pre-process.
        colour_norm_factor (`(n_rounds_use x n_channels_use) ndarray[float32]`): the colour normalisation factors.
        background_dot_product_threshold (float): the background gene round-dot product threshold.
        background_subtract_percentile (float): the background gene subtraction percentile.

    Returns:
        (`(n_colours x n_rounds_use x n_channels_use) ndarray[float32]`): preprocessed_colours. The pre-processed
            colours.
    """
    assert type(colours) is np.ndarray
    assert colours.ndim == 3
    assert colours.dtype == np.float32
    assert type(colour_norm_factor) is np.ndarray
    assert colour_norm_factor.ndim == 2
    assert colour_norm_factor.dtype == np.float32
    assert type(background_dot_product_threshold) is float
    assert type(background_subtract_percentile) is float
    assert background_dot_product_threshold >= 0
    assert background_subtract_percentile >= 0
    assert background_subtract_percentile <= 100

    n_colours, n_rounds_use, n_channels_use = colours.shape

    preprocessed_colours = colours.copy()
    bg_genes = create_background_bled_codes(n_rounds_use, n_channels_use)
    preprocessed_colours *= colour_norm_factor[np.newaxis]
    # colours_background_gene_is_subtracted[i, c] is true for background gene channel index c has been subtracted from
    # colour index i. This is to avoid double subtracting a background's channel.
    colours_background_gene_is_subtracted = np.zeros((n_colours, n_channels_use), bool)
    colours_to_continue_subtraction = np.ones(n_colours, bool)

    for _ in range(n_channels_use):
        bg_scores = dot_product.dot_product_score(preprocessed_colours[np.newaxis], bg_genes[np.newaxis, np.newaxis])
        # Has shape (n_colours x n_channels_use).
        bg_scores = bg_scores[0]
        assert type(bg_scores) is np.ndarray
        highest_scoring_bg_genes = np.argmax(bg_scores, 1)
        highest_scoring_scores = bg_scores[range(n_colours), highest_scoring_bg_genes]
        # Continue with high scoring background genes that have not already been subtracted.
        colours_to_continue_subtraction[colours_to_continue_subtraction] = np.logical_and(
            highest_scoring_scores[colours_to_continue_subtraction] >= background_dot_product_threshold,
            ~(
                colours_background_gene_is_subtracted[
                    colours_to_continue_subtraction, highest_scoring_bg_genes[colours_to_continue_subtraction]
                ]
            ),
        )
        if not colours_to_continue_subtraction.any():
            break

        # Has shape (n_colours_continue, 1).
        percentiles = np.percentile(
            preprocessed_colours[
                colours_to_continue_subtraction,
                :,
                highest_scoring_bg_genes[colours_to_continue_subtraction],
            ],
            background_subtract_percentile,
            1,
            keepdims=True,
        )
        assert percentiles.shape == (colours_to_continue_subtraction.sum(), 1)
        preprocessed_colours[
            colours_to_continue_subtraction, :, highest_scoring_bg_genes[colours_to_continue_subtraction]
        ] -= percentiles
        colours_background_gene_is_subtracted[colours_to_continue_subtraction, highest_scoring_bg_genes] = True

    return preprocessed_colours


def create_background_bled_codes(n_rounds_use: int, n_channels_use: int) -> np.ndarray:
    """
    Create the background bled codes that are used during OMP pixel score computing.

    Args:
        n_rounds_use (int): the number of sequencing rounds.
        n_channels_use (int): the number of sequencing channels.

    Returns:
        (`(n_channels_use x n_rounds_use x n_channels_use) ndarray[float32]`): bg_bled_codes. bg_bled_codes[i] is
            the i'th background bled code.
    """
    bg_bled_codes = np.eye(n_channels_use, dtype=np.float32)[:, None, :].repeat(n_rounds_use, axis=1)
    # Normalise the codes the same way as gene bled codes.
    bg_bled_codes /= np.linalg.norm(bg_bled_codes, axis=(1, 2))

    return bg_bled_codes
