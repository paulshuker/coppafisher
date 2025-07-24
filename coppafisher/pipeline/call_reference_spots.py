import importlib.resources as importlib_resources
import itertools
import math as maths
import os
from typing import Tuple

import numpy as np
import zarr

from .. import log
from ..call_spots.base import bayes_mean, compute_bleed_matrix
from ..call_spots.dot_product import dot_product_score, gene_prob_score
from ..setup.config_section import ConfigSection
from ..setup.notebook_page import NotebookPage
from ..utils import intensity as utils_intensity
from ..utils import system

MAX_BUFFER_SIZE_BYTES = 2147483647


def call_reference_spots(
    config: ConfigSection, nbp_ref_spots: NotebookPage, nbp_file: NotebookPage, nbp_basic: NotebookPage
) -> Tuple[NotebookPage, NotebookPage]:
    """
    Function to do gene assignments to reference spots. In doing so we compute some important parameters for the
    downstream OMP analysis also.

    Args:
        config (ConfigSection): the `call_spots` config section.
        nbp_ref_spots (NotebookPage): `ref_spots` notebook page.
        nbp_file (NotebookPage): `file_names` notebook page.
        nbp_basic (NotebookPage): `basic_info` notebook page.

    Returns:
        nbp: NotebookPage
            The call spots notebook page.
        nbp_ref_spots: NotebookPage
            The reference spots notebook page.
    """
    # TODO: Run call spots on each tile separately as this is robust for huge, many tile datasets.

    log.info("Call spots started")
    nbp = NotebookPage("call_spots", {config.name: config.to_dict()})

    # assign config values that have not been provided
    if config["target_values"] is None:
        if len(nbp_basic.use_channels) == 7:
            config["target_values"] = [1, 1, 0.9, 0.7, 0.8, 1, 1]
        elif len(nbp_basic.use_channels) == 9:
            config["target_values"] = [1, 0.8, 0.2, 0.9, 0.6, 0.8, 0.3, 0.7, 1]
        else:
            raise ValueError("The target values should be provided in the config.")
    if config["d_max"] is None:
        if len(nbp_basic.use_channels) == 7:
            config["d_max"] = [0, 1, 3, 2, 4, 5, 6]
        elif len(nbp_basic.use_channels) == 9:
            config["d_max"] = [0, 1, 1, 3, 2, 4, 5, 5, 6]
        else:
            raise ValueError("The d_max values should be provided in the config.")

    gene_names, gene_codes = np.genfromtxt(nbp_file.code_book, dtype=(str, str), encoding="utf-8-sig").transpose()
    gene_codes = np.array([[int(i) for i in gene_codes[j]] for j in range(len(gene_codes))], np.int32)
    if config["kappa"] is None:
        n_genes = len(gene_names)
        config["kappa"] = 2 if n_genes <= 100 else 3

    # check shapes
    assert (
        len(config["target_values"]) == len(config["d_max"]) == len(nbp_basic.use_channels)
    ), "The target values, d_max and use_channels should have the same length."

    # load in frequently used variables
    spot_colours = nbp_ref_spots.colours[:].astype(np.float32)
    spot_tile = nbp_ref_spots.tile[:]
    n_tiles, n_rounds, n_channels_use = nbp_basic.n_tiles, nbp_basic.n_rounds, len(nbp_basic.use_channels)
    n_dyes, n_spots, n_genes = len(nbp_basic.dye_names), len(spot_colours), len(gene_names)
    use_tiles, use_channels = (nbp_basic.use_tiles, nbp_basic.use_channels)

    if nbp_file.initial_bleed_matrix is not None:
        raw_bleed_matrix = np.load(nbp_file.initial_bleed_matrix)
    else:
        raw_bleed_path = importlib_resources.files("coppafisher.setup").joinpath("dye_info_raw.npy")
        raw_bleed_matrix = np.load(raw_bleed_path)[:, use_channels].astype(float)
    raw_bleed_matrix = raw_bleed_matrix.astype(np.float32)
    raw_bleed_matrix = raw_bleed_matrix / np.linalg.norm(raw_bleed_matrix, axis=1)[:, None]

    # 1. Normalise spot colours and remove background as constant offset across different rounds of the same channel
    colour_norm_factor_initial = np.zeros((n_tiles, n_rounds, n_channels_use), np.float32)
    for t in use_tiles:
        # Dividing by zero can happen when bad_trc is set. This warning is ignored. Infinities are set to ones.
        with np.errstate(divide="ignore", invalid="ignore"):
            colour_norm_factor_initial[t] = 1 / (np.percentile(spot_colours[spot_tile == t], 95, axis=0))
        colour_norm_factor_initial[colour_norm_factor_initial == np.inf] = 1
        spot_colours[spot_tile == t] *= colour_norm_factor_initial[t]

    if config["background_subtract"]:
        # Remove background as constant offset across different rounds of the same channel
        spot_colours -= np.percentile(spot_colours, 25, axis=1, keepdims=True)

    # Compute the spot intensities
    spot_intensities = utils_intensity.compute_intensity(spot_colours).numpy().astype(np.float16)
    intensity_threshold = config["gene_intensity_threshold"]

    # 2. Compute gene probabilities for each spot
    bled_codes = raw_bleed_matrix[gene_codes]
    gene_prob_initial = gene_prob_score(spot_colours, bled_codes, kappa=config["kappa"])

    # 3. Use spots with score above threshold to work out global dye codes
    prob_mode_initial, prob_score_initial = np.argmax(gene_prob_initial, axis=1), np.max(gene_prob_initial, axis=1)
    pixel_chunk_size: int = gene_prob_initial.shape[0]
    while pixel_chunk_size * 4 > MAX_BUFFER_SIZE_BYTES:
        pixel_chunk_size = maths.ceil(pixel_chunk_size / 2)
    kwargs = dict(zarr_version=2, overwrite=True)
    gene_prob_initial = zarr.array(
        gene_prob_initial,
        store=os.path.join(nbp_file.output_dir, "gene_prob_init.zarray"),
        chunks=(pixel_chunk_size, 1),
        **kwargs,
    )
    prob_threshold = min(config["gene_prob_threshold"], np.percentile(prob_score_initial, 90))
    good = (prob_score_initial > prob_threshold) & (spot_intensities > intensity_threshold)
    bleed_matrix_initial = compute_bleed_matrix(spot_colours[good], prob_mode_initial[good], gene_codes, n_dyes)

    # 4. Compute the free_bled_codes
    free_bled_codes_tile_indep = np.zeros((n_genes, n_rounds, n_channels_use), np.float32)
    free_bled_codes = np.zeros((n_genes, n_tiles, n_rounds, n_channels_use), np.float32)

    for g in range(n_genes):
        good_g = (prob_mode_initial == g) & good
        for r in range(n_rounds):
            free_bled_codes_tile_indep[g, r] = bayes_mean(
                spot_colours=spot_colours[good_g, r],
                prior_colours=bleed_matrix_initial[gene_codes[g, r]],
                conc_param_parallel=config["concentration_parameter_parallel"],
                conc_param_perp=config["concentration_parameter_perpendicular"],
            )
            for t in use_tiles:
                good_gt = (prob_mode_initial == g) & (spot_tile == t) & good
                free_bled_codes[g, t, r] = bayes_mean(
                    spot_colours=spot_colours[good_gt, r],
                    prior_colours=bleed_matrix_initial[gene_codes[g, r]],
                    conc_param_parallel=config["concentration_parameter_parallel"],
                    conc_param_perp=config["concentration_parameter_perpendicular"],
                )
    # normalise the free bled codes
    free_bled_codes_tile_indep /= np.linalg.norm(free_bled_codes_tile_indep, axis=(1, 2))[:, None, None]
    free_bled_codes[:, use_tiles] /= np.linalg.norm(free_bled_codes[:, use_tiles], axis=(2, 3))[:, :, None, None]

    # 5. compute the scale factor V_rc maximising the similarity between the tile independent codes and the target
    # values. Then rename the product V_rc * free_bled_codes to bled_codes
    rc_scale = np.ones((n_rounds, n_channels_use), np.float32)
    for r, c in np.ndindex(n_rounds, n_channels_use):
        rc_genes = np.where(gene_codes[:, r] == config["d_max"][c])[0]
        n_spots_per_gene = np.array(
            [np.sum((prob_mode_initial == g) & (prob_score_initial > prob_threshold)) for g in rc_genes]
        )
        if np.sum(n_spots_per_gene) == 0:
            continue
        rc_scale[r, c] = np.sum(
            np.sqrt(n_spots_per_gene) * free_bled_codes_tile_indep[rc_genes, r, c] * config["target_values"][c]
        ) / np.sum(np.sqrt(n_spots_per_gene) * free_bled_codes_tile_indep[rc_genes, r, c] ** 2)
    bled_codes = free_bled_codes_tile_indep * rc_scale[None, :, :]
    # normalise the constrained bled codes
    bled_codes /= np.linalg.norm(bled_codes, axis=(1, 2), keepdims=True)

    # 6. Compute the scale factor Q_trc maximising the similarity between the tile independent codes and the
    # constrained bled codes.
    tile_scale = np.ones((n_tiles, n_rounds, n_channels_use), np.float32)
    for t, r, c in itertools.product(use_tiles, range(n_rounds), range(n_channels_use)):
        relevant_genes = np.where(gene_codes[:, r] == config["d_max"][c])[0]
        n_spots_per_gene = np.array(
            [
                np.sum((prob_mode_initial == g) & (prob_score_initial > prob_threshold) & (spot_tile == t))
                for g in relevant_genes
            ]
        )
        if n_spots_per_gene.sum() == 0:
            log.warn(f"No relevant spots found to calculate tile scale factor Q for {t=}, {r=}, {c=}")
            continue
        tile_scale[t, r, c] = np.sum(
            np.sqrt(n_spots_per_gene) * bled_codes[relevant_genes, r, c] * free_bled_codes[relevant_genes, t, r, c]
        ) / np.sum(np.sqrt(n_spots_per_gene) * free_bled_codes[relevant_genes, t, r, c] ** 2)

    # 7. Update the normalised spots and the bleed matrix, then do a second round of gene assignments with the new bled
    # codes.
    colour_norm_factor = colour_norm_factor_initial * tile_scale
    spot_colours *= tile_scale[spot_tile, :, :]  # update the spot colours
    gene_prob = gene_prob_score(spot_colours=spot_colours, bled_codes=bled_codes, kappa=config["kappa"])  # update probs
    prob_mode, prob_score = np.argmax(gene_prob, axis=1), np.max(gene_prob, axis=1)
    gene_prob = zarr.array(
        gene_prob, store=os.path.join(nbp_file.output_dir, "gene_prob.zarray"), chunks=(pixel_chunk_size, 1), **kwargs
    )
    # Computing all dot product scores at once can take too much memory.
    gene_dot_products = np.zeros((n_spots, n_genes), np.float16)
    n_max_score_pixels = 8.7e-2 * system.get_available_memory() * 1e9 / (n_genes * n_rounds * n_channels_use)
    n_max_score_pixels = int(max(1, n_max_score_pixels))
    n_batches = maths.ceil(n_spots / n_max_score_pixels)
    log.debug(f"{n_max_score_pixels=}")
    for batch_i in range(n_batches):
        index_min = batch_i * n_max_score_pixels
        index_max = min(spot_colours.shape[0], (batch_i + 1) * n_max_score_pixels)
        batch_scores = dot_product_score(
            spot_colours=spot_colours[np.newaxis, index_min:index_max], bled_codes=bled_codes[np.newaxis, np.newaxis]
        )[0]
        gene_dot_products[index_min:index_max] = batch_scores
        del batch_scores
    dp_gene, dp_score = np.argmax(gene_dot_products, axis=1).astype(np.int16), np.max(gene_dot_products, axis=1)
    dp_gene = zarr.array(
        dp_gene, store=os.path.join(nbp_file.output_dir, "dp_mode.zarray"), chunks=pixel_chunk_size, **kwargs
    )
    dp_score = zarr.array(
        dp_score, store=os.path.join(nbp_file.output_dir, "dp_score.zarray"), chunks=pixel_chunk_size, **kwargs
    )
    # Update bleed matrix.
    good = (prob_score > prob_threshold) & (spot_intensities > intensity_threshold)
    bleed_matrix = compute_bleed_matrix(spot_colours[good], prob_mode[good], gene_codes, n_dyes)
    intensity = utils_intensity.compute_intensity(
        nbp_ref_spots.colours[:].astype(np.float32) * colour_norm_factor[spot_tile]
    )
    intensity = intensity.numpy()

    # 8. Save the results.
    nbp.intensity = zarr.array(
        intensity, store=os.path.join(nbp_file.output_dir, "intensity.zarray"), chunks=pixel_chunk_size, **kwargs
    )
    nbp.dot_product_gene_no, nbp.dot_product_gene_score = dp_gene, dp_score
    nbp.gene_probabilities_initial = gene_prob_initial
    nbp.gene_probabilities = gene_prob
    nbp.gene_names, nbp.gene_codes = gene_names, gene_codes
    nbp.initial_scale, nbp.rc_scale, nbp.tile_scale = colour_norm_factor_initial, rc_scale, tile_scale
    nbp.colour_norm_factor = colour_norm_factor
    nbp.free_bled_codes, nbp.free_bled_codes_tile_independent = free_bled_codes, free_bled_codes_tile_indep
    nbp.bled_codes = bled_codes
    nbp.bleed_matrix_raw, nbp.bleed_matrix_initial, nbp.bleed_matrix = (
        raw_bleed_matrix,
        bleed_matrix_initial,
        bleed_matrix,
    )
    log.info("Call spots complete")

    return nbp
