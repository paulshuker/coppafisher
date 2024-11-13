from typing import Tuple

import numpy as np
import torch

from .. import log
from ..call_spots import dot_product


class CoefficientSolverOMP:
    NO_GENE_ASSIGNMENT: int = -32_768

    def __init__(self) -> None:
        """
        Initialise the CoefficientSolverOMP object. Used to compute OMP coefficients.
        """
        pass

    def solve(
        self,
        pixel_colours: np.ndarray[np.float32],
        bled_codes: np.ndarray[np.float32],
        background_codes: np.ndarray[np.float32],
        maximum_iterations: int,
        dot_product_threshold: float,
        minimum_intensity: float,
        return_all_scores: bool = False,
        return_all_weights: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute OMP coefficients for all pixel colours from the same tile. At each iteration of OMP, the next best gene
        assignment is found from the residual spot colours. A pixel is stopped iterating on if gene assignment fails.
        See function `get_next_gene_assignments` below for details on the stopping criteria and gene scoring. Pixels
        that do not fail have their new gene score added to the final coefficients. Pixels that are gene assigned are
        then fitted with the additional gene to find updated coefficients. See function `get_next_gene_coefficients`
        for details on the coefficient computation.

        Args:
            pixel_colours (`(n_pixels x n_rounds_use x n_channels_use) ndarray[float]`): pixel intensity in each
                sequencing round and channel.
            bled_codes (`(n_genes x n_rounds_use x n_channels_use) ndarray[float32]`): every gene bled code.
            background_codes (`(n_channels_use x n_rounds_use x n_channels_use) tensor[float]`): the background bled
                codes. These are simply uniform brightness in one channel for all rounds. background_codes[0] is the
                first code, background_codes[1] is the second code, etc.
            maximum_iterations (int): the maximum number of gene assignments allowed for one pixel.
            dot_product_threshold (float): a gene must have a dot product score above this value on the residual spot
                colour to be assigned the gene. If more than one gene is above this threshold, the top score is used.
            minimum_intensity (float): a pixel's residual intensity must be above minimum_intensity to pass gene
                assignment.
            return_all_scores (bool, optional): return all gene round dot product scores on each iteration. Default:
                false.
            return_all_weights (bool, optional): return all gene bled code weights for every gene that was assigned.
                Default: false.

        Returns:
            Tuple (tensor if only one tensor is returned) containing the following:
                - (`(n_pixels x n_genes) ndarray[float32]`): coefficients. Each gene's final coefficient for every
                    pixel.
                - (`((n_iterations + 1) x n_pixels x n_genes_all) ndarray[float32]`): dp_scores. The dot product
                    score for every gene on each iteration. This even includes the iteration that did not assign any new
                    genes so you can see what the final gene scores were before stopping. Only returned if
                    return_dp_scores is true.
                - (`(n_pixels x n_genes) ndarray[float32]`): gene_weights. The gene weights given to each gene on all
                    pixels on their final iteration. For genes that were not assigned on a pixel, nan is placed. Only
                    returned if return_all_weights is true.

        Notes:
            - All computations are run with 32-bit float precision.
        """
        n_pixels, n_rounds_use, n_channels_use = pixel_colours.shape
        n_rounds_channels_use = n_rounds_use * n_channels_use
        n_genes = bled_codes.shape[0]
        assert type(pixel_colours) is np.ndarray
        assert type(bled_codes) is np.ndarray
        assert type(background_codes) is np.ndarray
        assert type(maximum_iterations) is int
        assert type(dot_product_threshold) is float
        assert type(minimum_intensity) is float
        assert type(return_all_scores) is bool
        assert type(return_all_weights) is bool
        assert maximum_iterations > 0
        assert dot_product_threshold >= 0
        assert minimum_intensity >= 0
        assert pixel_colours.ndim == 3
        assert bled_codes.ndim == 3
        assert background_codes.ndim == 3
        assert pixel_colours.size > 0, "pixel_colours cannot be empty"
        assert bled_codes.size > 0, "bled_codes cannot be empty"
        assert background_codes.size > 0, "background_codes cannot be empty"
        assert bled_codes.shape == (n_genes, n_rounds_use, n_channels_use)
        assert background_codes.shape == (n_channels_use, n_rounds_use, n_channels_use)

        dp_scores = []
        bled_codes_torch = torch.tensor(bled_codes, dtype=torch.float32)
        background_codes_torch = torch.tensor(background_codes, dtype=torch.float32)
        all_bled_codes = torch.concat((bled_codes_torch, background_codes_torch), dim=0)

        coefficients = torch.zeros((n_pixels, n_genes), dtype=torch.float32)
        colours = torch.from_numpy(pixel_colours.astype(np.float32))
        # Remember the residual colour between iterations.
        residual_colours = colours.detach().clone()
        # Remember what pixels still need iterating on.
        pixels_to_continue = torch.ones(n_pixels, dtype=bool)
        # Remember the gene selections made for each pixel. NO_GENE_ASSIGNMENT for no gene selection made.
        genes_selected = torch.full((n_pixels, maximum_iterations), self.NO_GENE_ASSIGNMENT, dtype=torch.int32)
        bg_gene_indices = torch.linspace(n_genes, n_genes + n_channels_use - 1, n_channels_use, dtype=torch.int32)
        bg_gene_indices = bg_gene_indices[np.newaxis].repeat_interleave(n_pixels, dim=0)
        if return_all_weights:
            # Remember the gene weightings given to each pixel.
            all_weights = torch.full_like(coefficients, torch.nan, dtype=torch.float32)

        for iteration in range(maximum_iterations):
            log.debug(f"Iteration: {iteration}")
            # Find the next best gene and coefficients for pixels that have not reached a stopping criteria yet.
            log.debug(f"Finding next gene assignments")
            fail_gene_indices = torch.cat((genes_selected[:, :iteration], bg_gene_indices), 1)
            fail_gene_indices = fail_gene_indices[pixels_to_continue]
            gene_assigment_results = self.get_next_gene_assignments(
                residual_colours,
                all_bled_codes,
                fail_gene_indices,
                dot_product_threshold,
                minimum_intensity,
                return_all_scores=return_all_scores,
            )
            del fail_gene_indices
            genes_selected[pixels_to_continue, iteration] = gene_assigment_results[0]
            if return_all_scores:
                dp_score = torch.zeros((n_pixels, n_genes + n_channels_use), dtype=torch.float32)
                dp_score[pixels_to_continue] = gene_assigment_results[2].cpu()
                dp_scores.append(dp_score)

            # Update what pixels to continue iterating on.
            pixels_to_continue = genes_selected[:, iteration] != self.NO_GENE_ASSIGNMENT
            if pixels_to_continue.sum() == 0:
                break

            # Update coefficients on the pixels that passed.
            log.debug(f"Assigning new coefficients")
            coefficients[pixels_to_continue, genes_selected[pixels_to_continue, iteration]] = gene_assigment_results[1][
                ~torch.isnan(gene_assigment_results[1])
            ]
            del gene_assigment_results

            # On the pixels still being iterated on, update the gene weights and hence the residual colours for the
            # next iteration.
            log.debug(f"Computing residual colours")
            latest_gene_selections = genes_selected[pixels_to_continue, : iteration + 1]
            bled_codes_to_continue = bled_codes_torch[latest_gene_selections]
            # Flatten rounds/channels.
            bled_codes_to_continue = bled_codes_to_continue.reshape((-1, iteration + 1, n_rounds_channels_use))
            # Change to shape (n_pixels_continue, n_rounds_channels_use, iteration + 1).
            bled_codes_to_continue = bled_codes_to_continue.swapaxes(1, 2)
            residual_colours = self.get_next_residual_colours(
                colours[pixels_to_continue].reshape((-1, n_rounds_channels_use))[:, :, np.newaxis],
                bled_codes_to_continue,
                return_weights=return_all_weights,
            )
            if return_all_weights:
                all_weights[pixels_to_continue][:, latest_gene_selections] = residual_colours[1]
                residual_colours = residual_colours[0]
            del latest_gene_selections, bled_codes_to_continue
            residual_colours = residual_colours.reshape((-1, n_rounds_use, n_channels_use))

        result = (coefficients.cpu().numpy(),)
        if return_all_scores:
            result += (np.array([score.cpu().numpy() for score in dp_scores]),)
        if return_all_weights:
            result += (all_weights.cpu().numpy(),)
        if len(result) == 1:
            result = result[0]
        return result

    def create_background_bled_codes(self, n_rounds_use: int, n_channels_use: int) -> np.ndarray:
        """
        Create the background bled codes that are used during OMP coefficient computing.

        Args:
            n_rounds_use (int): the number of sequencing rounds.
            n_channels_use (int): the number of sequencing channels.
        """
        bg_bled_codes = np.eye(n_channels_use)[:, None, :].repeat(n_rounds_use, axis=1)
        # Normalise the codes the same way as gene bled codes.
        bg_bled_codes /= np.linalg.norm(bg_bled_codes, axis=(1, 2))
        return bg_bled_codes

    def get_next_gene_assignments(
        self,
        residual_colours: torch.Tensor,
        all_bled_codes: torch.Tensor,
        fail_gene_indices: torch.Tensor,
        dot_product_threshold: float,
        minimum_intensity: float,
        return_all_scores: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the next best gene assignment for each residual colour. Each gene is scored to each pixel using a dot
        product scoring where each round has an equal contribution. A pixel fails gene assignment if one or more of the
        conditions is met:

        - The top gene dot product score is below the dot_product_threshold.
        - The next best gene is in the fail_gene_indices list.
        - The intensity of the colour is below the minimum intensity.

        The reason for each of these conditions is:

        - to cut out unconfident gene reads.
        - to not doubly assign a gene and avoid assigning background genes.
        - to cut out dim colour.

        respectively.

        Args:
            residual_colours (`(n_pixels x n_rounds_use x n_channels_use) tensor[float32]`): residual pixel colour.
            all_bled_codes (`(n_genes_all x n_rounds_use x n_channels_use) tensor[float32]`): gene bled codes and
                background genes appended.
            fail_gene_indices (`(n_pixels x n_genes_fail) tensor[int32]`): if the next gene assignment for a pixel is
                included on the list of fail gene indices, consider gene assignment a fail.
            dot_product_threshold (float): a gene can only be assigned if the dot product score is above this threshold.
            minimum_intensity (float): a colour's intensity must be above minimum_intensity to pass gene assignment.
                The intensity is defined as min_r (max_c abs(residual_colour)).
            return_all_scores (bool, optional): return the dot product scores for every gene. Default: false.

        Returns:
            Tuple containing:
                - `(n_pixels) tensor[int32]`: next_best_genes. The next best gene assignment for each pixel. A value of
                    -32_768 is placed for pixels that failed to find a next best gene.
                - `(n_pixels) tensor[float32]`: next_best_gene_scores. The dot product score given for the next best
                    gene assignment. These will become the next, updated coefficients. If a pixel did not get a new
                    gene assignment, then the score is torch.nan.
                - `(n_pixels x n_genes_all) tensor[float32]`: all_gene_scores. Every genes' round dot product score.
                    This includes genes that are in fail_gene_indices. Only returned if return_scores is set to true.
        """
        assert type(residual_colours) is torch.Tensor
        assert type(all_bled_codes) is torch.Tensor
        assert type(fail_gene_indices) is torch.Tensor
        assert type(dot_product_threshold) is float
        assert type(minimum_intensity) is float
        assert residual_colours.ndim == 3
        assert all_bled_codes.ndim == 3
        assert fail_gene_indices.ndim == 2
        assert residual_colours.shape[0] > 0, "Require at least one pixel"
        assert residual_colours.shape[1] > 0, "Require at least one round/channel"
        assert residual_colours.shape[1:] == all_bled_codes.shape[1:]
        assert all_bled_codes.shape[0] > 0, "Require at least one bled code"
        assert fail_gene_indices.shape[0] == residual_colours.shape[0]
        assert (fail_gene_indices >= 0).all() and (fail_gene_indices < all_bled_codes.shape[0]).all()
        assert dot_product_threshold >= 0
        assert minimum_intensity >= 0

        intensity_is_low = residual_colours.clone().abs().max(2).values.min(1).values < minimum_intensity

        all_gene_scores = dot_product.dot_product_score(residual_colours, all_bled_codes)

        next_best_gene_scores, next_best_genes = torch.max(all_gene_scores, dim=1)
        next_best_genes = next_best_genes.int()

        # A pixel only passes if the highest scoring gene is above the dot product threshold.
        pixels_passed = (all_gene_scores > dot_product_threshold).any(1)
        log.debug(f"Pixels passed min score: {pixels_passed.sum()} out of {pixels_passed.shape}")

        # A best gene in the fail_gene_indices means assignment failed.
        in_fail_gene_indices = (fail_gene_indices == next_best_genes[:, np.newaxis]).any(1)
        log.debug(f"Pixels in failing gene index: {in_fail_gene_indices.sum()} out of {in_fail_gene_indices.shape}")
        pixels_passed = pixels_passed & (~in_fail_gene_indices)

        # An intensity below the minimum_intensity means assignment failed.
        log.debug(f"Pixels too dim: {intensity_is_low.sum()} out of {intensity_is_low.shape}")
        pixels_passed = pixels_passed & (~intensity_is_low)

        log.debug(f"So total pixels passed: {pixels_passed.sum()} out of {pixels_passed.shape}")
        next_best_genes[~pixels_passed] = self.NO_GENE_ASSIGNMENT
        next_best_gene_scores[~pixels_passed] = torch.nan

        output = (next_best_genes, next_best_gene_scores)

        if return_all_scores:
            output += (all_gene_scores,)

        return output

    def get_next_residual_colours(
        self, pixel_colours: torch.Tensor, bled_codes: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gene weights for each given pixel colour by least squares with the gene bled codes. These weighted bled
        codes are then subtracted off the pixel colour to get the minimised residual colour for each pixel.

        Args:
            pixel_colours (`(n_pixels x n_rounds_channels_use x 1) tensor[float32]`): each pixel's colour.
            bled_codes (`(n_pixels x n_rounds_channels_use x n_genes_added) tensor[float32]`): the bled code for each
                added gene for each pixel.
            return_weights (bool, optional): return the weightings given to each assigned gene.

        Returns:
            Tuple (unravelled if the tuple is of length 1) containing:
                - (`(n_pixels x n_rounds_channels_use) tensor[float32]`) residuals: the residual colour after
                    subtracting the assigned, weighted gene bled codes.
                - (`(n_pixels x n_genes_added) tensor[float32]`): gene_weights. The weight given to every gene bled
                    code.
        """
        assert type(pixel_colours) is torch.Tensor
        assert type(bled_codes) is torch.Tensor
        assert type(return_weights) is bool
        assert pixel_colours.ndim == 3
        assert bled_codes.ndim == 3
        assert pixel_colours.shape[0] == bled_codes.shape[0]
        assert pixel_colours.shape[1] == bled_codes.shape[1]
        assert pixel_colours.shape[2] == 1
        assert bled_codes.shape[0] > 0, "Require at least one pixel to run on"
        assert bled_codes.shape[1] > 0, "Require at least one round and channel"
        assert bled_codes.shape[2] > 0, "Require at least one gene assigned"

        # Compute least squares for gene weights.
        # First parameter A has shape (n_pixels x n_rounds_channels_use x n_genes_added)
        # Second parameter B has shape (n_pixels x n_rounds_channels_use x 1)
        # So, the resulting weights has shape (n_pixels x n_genes_added x 1)
        # The least squares is minimising || A @ coefficients - B || ^ 2
        weights = torch.linalg.lstsq(bled_codes, pixel_colours, rcond=-1, driver="gels")[0]
        # Squeeze shape to (n_pixels x n_genes_added).
        weights: torch.Tensor = weights[..., 0]

        # From the new weights, find the residual spot colours.
        pixel_residuals = pixel_colours[..., 0] - (weights[:, np.newaxis] * bled_codes).sum(2)

        result = pixel_residuals

        if return_weights:
            result = (result, weights)

        return result
