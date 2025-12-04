import os
import warnings
from typing import List, Literal

import numpy as np
import pandas as pd
import scipy

from ..omp import base as omp_base
from ..setup.notebook_page import NotebookPage


# Each instance of this class holds data on a specific gene calling method.
class MethodData:
    """
    Spot results for a particular gene calling method.

    The way that data is saved to the notebook is different for each gene calling method. Therefore, we create this
    class as a common interface for gathering and filtering spot results. It is used by multiple post-pipeline functions
    like export_to_pciseq and the Viewer.
    """

    MAX_CHECK_COUNT = 10_000
    _ATTRIBUTE_NAMES = ("tile", "local_yxz", "yxz", "gene_no", "score", "colours", "intensity", "indices")

    method: str
    tile: np.ndarray
    local_yxz: np.ndarray
    yxz: np.ndarray
    gene_no: np.ndarray
    score: np.ndarray
    colours: np.ndarray
    intensity: np.ndarray
    # We keep track of the spots' indices relative to the notebook since we will cut out spots that are part of
    # invisible genes to improve performance.
    indices: np.ndarray

    def __init__(
        self,
        method: str,
        nbp_basic: NotebookPage,
        nbp_stitch: NotebookPage,
        nbp_ref_spots: NotebookPage,
        nbp_call_spots: NotebookPage,
        nbp_omp: NotebookPage | None,
        show_tiles: List[int] | None = None,
        spot_scoring: Literal["discriminality"] | None = None,
    ) -> None:
        """
        Gather all spot data for a particular gene calling method that is stored in self.

        Args:
            method (str): gene calling method to gather. Can be 'prob_init', 'prob', 'anchor', or 'omp'.
            nbp_basic (NotebookPage): `basic_info` notebook page.
            nbp_stitch (NotebookPage): `stitch` notebook page.
            nbp_ref_spots (NotebookPage): `ref_spots` notebook page.
            nbp_call_spots (NotebookPage): `call_spots` notebook page.
            nbp_omp (NotebookPage or none): `omp` notebook page.
            show_tiles (list of int, optional): tile indices to gather. Default: all tiles.
            spot_scoring (str or none, optional): what scoring method to display. If none, then the default scoring from
                the notebook is used. If set to "discriminality", then the spots are scored based on the formula
                `c / stdev(v)` where `c` is the spearman correlation of the spot's colour versus the assigned bled code,
                `stdev` is the standard deviation of values, and `v` is the spearman correlation of the spot's colour
                against every other gene bled code. Default: none.
        """
        assert type(method) is str
        assert method in ["prob_init", "prob", "anchor", "omp"], f"Unknown method {method}"
        assert type(nbp_ref_spots) is NotebookPage
        assert type(nbp_call_spots) is NotebookPage
        assert type(nbp_omp) is NotebookPage or nbp_omp is None
        assert type(show_tiles) is list or show_tiles is None
        assert type(spot_scoring) is str or spot_scoring is None
        if spot_scoring is not None:
            assert spot_scoring in ["discriminality"], f"Unknown spot scoring {spot_scoring}"
        self.spot_scoring = spot_scoring

        self.method = method

        if method == "prob_init":
            self.score = nbp_call_spots.gene_probabilities_initial[:].max(1)
            self.gene_no = np.argmax(nbp_call_spots.gene_probabilities_initial[:], 1).astype(np.int16)
        elif method == "prob":
            self.score = nbp_call_spots.gene_probabilities[:].max(1)
        elif method == "anchor":
            self.score = nbp_call_spots.dot_product_gene_score[:]

        if method in ("prob", "anchor"):
            self.gene_no = np.argmax(nbp_call_spots.gene_probabilities[:], 1).astype(np.int16)

        if method in ("prob_init", "prob", "anchor"):
            self.tile = nbp_ref_spots.tile[:]
            self.local_yxz = nbp_ref_spots.local_yxz[:].astype(np.int16)
            self.yxz = self.local_yxz.astype(np.float32) + nbp_stitch.tile_origin[self.tile]
            self.colours = nbp_ref_spots.colours[:].astype(np.float32)
            self.intensity = nbp_call_spots.intensity[:]
        elif method == "omp":
            if nbp_omp is None:
                raise ValueError(f"{method} requires the omp notebook page")
            self.local_yxz, self.tile = omp_base.get_all_local_yxz(nbp_basic, nbp_omp)
            self.yxz = self.local_yxz.astype(np.float32) + nbp_stitch.tile_origin[self.tile]
            self.gene_no = omp_base.get_all_gene_no(nbp_basic, nbp_omp)[0].astype(np.int16)
            self.score = omp_base.get_all_scores(nbp_basic, nbp_omp)[0]
            self.colours = omp_base.get_all_colours(nbp_basic, nbp_omp)[0].astype(np.float32)
            self.intensity = omp_base.get_all_intensities(nbp_basic, nbp_call_spots, nbp_omp)

        self.yxz -= np.floor(nbp_stitch.tile_origin[nbp_basic.use_tiles].min(0))[np.newaxis]
        self.indices = np.linspace(0, self.score.size - 1, self.score.size, dtype=np.uint32)

        if self.spot_scoring == "discriminality":
            n_spots = self.gene_no.size
            n_genes = nbp_call_spots.bled_codes.shape[0]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=scipy.stats.ConstantInputWarning)
                self.score = [
                    scipy.stats.spearmanr(
                        self.colours.reshape((n_spots, -1))[s],
                        nbp_call_spots.bled_codes[self.gene_no].reshape((n_spots, -1))[s],
                        axis=1,
                        nan_policy="raise",
                    ).statistic
                    for s in range(self.colours.shape[0])
                ]
            self.score = np.array(self.score, np.float32)
            self.score[np.isnan(self.score)] = 0

            # Each score is divided by the standard deviation of all other gene bled code spearman correlations.
            keep_bled_codes = [[i for i in range(n_genes) if i != self.gene_no[s]] for s in range(n_spots)]
            # Has shape (n_spots * (n_genes - 1), n_rounds_use * n_channels_use).
            other_bled_codes = nbp_call_spots.bled_codes[keep_bled_codes].reshape((n_spots, (n_genes - 1), -1))
            colours = self.colours.copy().reshape((n_spots, 1, -1))
            colours = np.repeat(colours, n_genes - 1, axis=1)
            correlations = []
            for s in range(n_spots):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=scipy.stats.ConstantInputWarning)
                    correlations.append(
                        [
                            scipy.stats.spearmanr(colours[s, g], other_bled_codes[s, g], nan_policy="raise").statistic
                            for g in range(n_genes - 1)
                        ]
                    )
            correlations = np.array(correlations, np.float32)

            self.score[(~np.isnan(correlations)).all(1)] /= np.std(
                correlations[(~np.isnan(correlations)).all(1)], axis=1
            )
            self.score[np.isnan(correlations).any(1)] = 0

        # Sanity check spot data.
        self._check_variables()
        if show_tiles is not None:
            self.remove_data_at((self.tile[np.newaxis] != np.array(show_tiles)[:, np.newaxis]).all(0))
            self._check_variables()

    def remove_data_at(self, remove: np.ndarray[bool]) -> None:
        """
        Delete a subset of the spot data in self.

        Args:
            remove (`(n_spots) ndarray[bool]`): removes ith spot if remove[i] is true.
        """
        assert type(remove) is np.ndarray
        assert remove.ndim == 1
        assert remove.size == self.tile.size

        keep_sum = (~remove).sum().item()
        for var_name in self._ATTRIBUTE_NAMES:
            self.__setattr__(var_name, self.__getattribute__(var_name)[~remove])
            assert self.__getattribute__(var_name).shape[0] == keep_sum
        self._check_variables()

    def save_csv(self, file_path: str, gene_names: np.ndarray[str], keep: np.ndarray[bool] | None = None) -> None:
        """
        Save a .csv file containing gene spot information.

        The csv contains:

        - Gene: Name of gene each spot was assigned to.
        - y: y coordinate of each spot in stitched coordinate system.
        - x: x coordinate of each spot in stitched coordinate system.
        - z_stack: z coordinate of each spot in stitched coordinate system (in units of z-pixels).
        - score: the spot's score.
        - intensity: the spot's intensity.

        Args:
            file_path (str): the csv file path.
            gene_names (`(n_genes) ndarray[str]`): gene_names[i] is the gene name for gene i.
            keep (`(n_spots) ndarray[bool]`, optional): keep[i] is true if i'th spot is kept. Default: keep all spots.

        Raises:
            (SystemError): if csv file already exists.
        """
        if os.path.isfile(file_path):
            raise SystemError(f"File at {file_path} already exists")

        gene = gene_names[self.gene_no]
        yxz = self.yxz
        score = self.score
        intensity = self.intensity

        if keep is not None:
            assert type(keep) is np.ndarray
            assert keep.shape == (self.tile.size,)

            gene = gene[keep]
            yxz = yxz[keep]
            score = score[keep]
            intensity = intensity[keep]

        df_to_export = pd.DataFrame()
        df_to_export["Gene"] = gene
        df_to_export["y"] = yxz[:, 0]
        df_to_export["x"] = yxz[:, 1]
        df_to_export["z_stack"] = yxz[:, 2]
        df_to_export["score"] = score
        df_to_export["intensity"] = intensity
        df_to_export.to_csv(file_path, mode="w", index=False)

    def _check_variables(self) -> None:
        assert all([type(self.__getattribute__(var_name)) is np.ndarray] for var_name in self._ATTRIBUTE_NAMES)
        assert self.tile.ndim == 1
        assert self.tile.shape[0] >= 0
        assert self.local_yxz.ndim == 2
        assert self.local_yxz.shape[0] >= 0
        assert self.local_yxz.shape[1] == 3
        assert self.gene_no.ndim == 1
        assert self.gene_no.shape[0] >= 0
        assert self.score.ndim == 1
        assert self.score.shape[0] >= 0
        assert self.intensity.ndim == 1
        assert self.intensity.shape[0] >= 0
        assert self.indices.ndim == 1
        assert self.indices.shape[0] >= 0
        if self.tile.size > np.iinfo(self.indices.dtype).max:
            raise ValueError(f"Too many spots in {self.method} to index with uint32")
        assert (
            self.tile.size
            == self.local_yxz.shape[0]
            == self.yxz.shape[0]
            == self.gene_no.size
            == self.score.size
            == self.colours.shape[0]
            == self.intensity.size
            == self.indices.size
        )

        # Check values on a subset.
        assert (self.tile[: self.MAX_CHECK_COUNT] >= 0).all()
        assert (self.gene_no[: self.MAX_CHECK_COUNT] >= 0).all()
        if self.spot_scoring != "discriminality":
            assert (self.score[: self.MAX_CHECK_COUNT] >= 0).all()
        assert (self.intensity[: self.MAX_CHECK_COUNT] >= 0).all()
        assert (self.indices[: self.MAX_CHECK_COUNT] >= 0).all()
