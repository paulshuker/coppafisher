import os
import textwrap
import webbrowser
from typing import Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.transforms import ScaledTranslation
from tqdm import tqdm
from typing_extensions import Self

from .. import log
from ..omp import base as omp_base
from ..setup.notebook import Notebook
from ..setup.notebook_page import NotebookPage


class BuildPDF:
    # Plot settings
    A4_SIZE_INCHES = (11.693, 8.268)
    LARGE_FONTSIZE = 25
    NORMAL_FONTSIZE = 18
    SMALL_FONTSIZE = 15
    SMALLER_FONTSIZE = 10
    TINY_FONTSIZE = 4
    INFO_FONTDICT = {"fontsize": NORMAL_FONTSIZE, "verticalalignment": "center"}
    N_GENES_SHOW = 40
    GENE_PROB_THRESHOLD = 0.7
    DEFAULT_REF_SCORE_THRESHOLD = 0.3
    DEFAULT_OMP_SCORE = 0.3
    HEATMAP_BIN_SIZE = 12  # In pixel count
    HEATMAP_PROB_SCORE_THRESH = 0.5
    HEATMAP_ANCHOR_SCORE_THRESH = 0.5
    HEATMAP_OMP_SCORE_THRESH = 0.2

    def __init__(
        self,
        nb: Union[Notebook, str],
        nbp_file: NotebookPage,
        output_dir: Optional[str] = None,
        auto_open: bool = False,
    ) -> None:
        """
        Build a diagnostic PDF of coppafisher results for each relevant section. A section pdf is not re-generated if the
        file was found inside output_dir.

        Args:
            - nb (Notebook or str): notebook or file path to notebook.
            - nbp_file (NotebookPage): `file_names` notebook page.
            - output_dir (str, optional): directory to save pdfs. Default: `nb.basic_info.file_names`.
            - auto_open (bool, optional): automatically open the output directory after creation. Default: false.
        """
        log.debug("Creating diagnostic PDF started")
        pbar = tqdm(desc="Creating Diagnostic PDFs", total=10, unit="section")
        pbar.set_postfix_str("Loading notebook")
        if type(nb) is str:
            nb = Notebook(nb)
        pbar.update()
        if output_dir is None:
            output_dir = nbp_file.output_dir
        output_dir = os.path.abspath(output_dir)
        assert os.path.isdir(output_dir), f"output_dir {output_dir} is not a valid directory"

        # Light default theme
        plt.style.use("default")
        self.tile_shape = (nb.basic_info.tile_sz, nb.basic_info.tile_sz, len(nb.basic_info.use_z))
        self.use_channels_anchor = [
            c for c in [nb.basic_info.dapi_channel, nb.basic_info.anchor_channel] if c is not None
        ]
        self.use_channels_anchor.sort()
        self.use_channels_plus_dapi = list(nb.basic_info.use_channels)
        if nb.basic_info.dapi_channel is not None:
            self.use_channels_plus_dapi += [nb.basic_info.dapi_channel]
        self.use_channels_plus_dapi.sort()
        self.use_channels_all = list(self.use_channels_plus_dapi)
        if nb.basic_info.anchor_channel is not None:
            self.use_channels_all += [nb.basic_info.anchor_channel]
        self.use_channels_all = list(set(self.use_channels_all))
        self.use_channels_all.sort()
        self.use_rounds_all = list(nb.basic_info.use_rounds) + nb.basic_info.use_anchor * [nb.basic_info.anchor_round]
        self.use_rounds_all.sort()

        if not os.path.isfile(os.path.join(output_dir, "_basic_info.pdf") and nb.has_page("basic_info")):
            with PdfPages(os.path.join(output_dir, "_basic_info.pdf")) as pdf:
                mpl.rcParams.update(mpl.rcParamsDefault)
                # Build a pdf with data from scale, extract, filter, find_spots, register, OMP
                pbar.set_postfix_str("basic info")
                text_intro_info = self.get_basic_info(nb.basic_info, nbp_file)
                fig, axes = self.create_empty_page(1, 1)
                self.empty_plot_ticks(axes)
                axes[0, 0].set_title(text_intro_info, fontdict=self.INFO_FONTDICT, y=0.5)
                pdf.savefig(fig)
                plt.close(fig)
        pbar.update()

        if not os.path.isfile(os.path.join(output_dir, "_extract.pdf")) and nb.has_page("extract"):
            with PdfPages(os.path.join(output_dir, "_extract.pdf")) as pdf:
                # Extract section
                pbar.set_postfix_str("extract")
                if nb.has_page("extract"):
                    fig, axes = self.create_empty_page(1, 1)
                    text_extract_info = ""
                    text_extract_info += self.get_extract_text_info(nb.extract)
                    axes[0, 0].set_title(text_extract_info, fontdict=self.INFO_FONTDICT, y=0.5)
                    extract_image_dtype = np.uint16
                    self.empty_plot_ticks(axes[0, 0])
                    pdf.savefig(fig)
                    plt.close(fig)
                    del fig, axes
                    file_path = os.path.join(nbp_file.extract_dir, "hist_counts_values.npz")
                    extract_pixel_unique_values, extract_pixel_unique_counts = None, None
                    if os.path.isfile(file_path):
                        results = np.load(file_path)
                        extract_pixel_unique_counts, extract_pixel_unique_values = results["arr_0"], results["arr_1"]
                    if extract_pixel_unique_values is not None:
                        pixel_min, pixel_max = np.iinfo(extract_image_dtype).min, np.iinfo(extract_image_dtype).max
                        # Histograms of pixel value histograms
                        figs = self.create_pixel_value_hists(
                            nb,
                            "Extract",
                            extract_pixel_unique_values,
                            extract_pixel_unique_counts,
                            pixel_min,
                            pixel_max,
                            bin_size=2**10,
                        )
                        for fig in figs:
                            pdf.savefig(fig)
                            plt.close(fig)
                        del figs
                    del extract_pixel_unique_values, extract_pixel_unique_counts
        pbar.update()

        if not os.path.isfile(os.path.join(output_dir, "_filter.pdf")) and nb.has_page("filter"):
            with PdfPages(os.path.join(output_dir, "_filter.pdf")) as pdf:
                # Filter section
                pbar.set_postfix_str("filter")
                if nb.has_page("filter") or nb.has_page("extract"):
                    fig, axes = self.create_empty_page(1, 1)
                    text_filter_info = ""
                    if nb.has_page("filter"):
                        # Versions >=0.5.0
                        text_filter_info += self.get_filter_info(nb.filter, nb.filter_debug)
                    else:
                        text_filter_info += self.get_filter_info(nb.extract)
                    axes[0, 0].set_title(text_filter_info, fontdict=self.INFO_FONTDICT, y=0.5)
                    self.empty_plot_ticks(axes[0, 0])
                    pdf.savefig(fig)
                    plt.close(fig)
        pbar.update()

        if not os.path.isfile(os.path.join(output_dir, "_find_spots.pdf")) and nb.has_page("find_spots"):
            with PdfPages(os.path.join(output_dir, "_find_spots.pdf")) as pdf:
                pbar.set_postfix_str("find spots")
                fig, axes = self.create_empty_page(1, 1)
                text_find_spots_info = ""
                text_find_spots_info += self.get_find_spots_info(nb.find_spots)
                axes[0, 0].set_title(text_find_spots_info, fontdict=self.INFO_FONTDICT, y=0.5)
                self.empty_plot_ticks(axes[0, 0])
                pdf.savefig(fig)
                plt.close(fig)

                minimum_spot_count = nb.find_spots.spot_no[nb.find_spots.spot_no != 0].min()
                maximum_spot_count = nb.find_spots.spot_no.max()
                for t in nb.basic_info.use_tiles:
                    fig, axes = self.create_empty_page(1, 1)
                    fig.suptitle(f"Find spot counts, tile {t}")
                    ax: plt.Axes = axes[0, 0]
                    channels_to_index = {c: i for i, c in enumerate(self.use_channels_all)}
                    X = np.zeros(
                        (nb.basic_info.n_rounds + nb.basic_info.n_extra_rounds, len(channels_to_index)),
                        dtype=np.int32,
                    )
                    ticks_channels = np.arange(X.shape[1])
                    ticks_channels_labels = ["" for _ in range(ticks_channels.size)]
                    ticks_rounds = np.arange(X.shape[0])
                    ticks_rounds_labels = ["" for _ in range(ticks_rounds.size)]
                    for r in self.use_rounds_all:
                        if nb.basic_info.use_anchor and r == nb.basic_info.anchor_round:
                            use_channels = [
                                c for c in [nb.basic_info.dapi_channel, nb.basic_info.anchor_channel] if c is not None
                            ]
                        else:
                            use_channels = list(nb.basic_info.use_channels)
                        for c in use_channels:
                            X[r, channels_to_index[c]] = nb.find_spots.spot_no[t, r, c]
                            ticks_channels_labels[channels_to_index[c]] = f"{c}"
                            if nb.basic_info.dapi_channel is not None and c == nb.basic_info.dapi_channel:
                                ticks_channels_labels[channels_to_index[c]] = "dapi"
                            if nb.basic_info.anchor_channel is not None and c == nb.basic_info.anchor_channel:
                                ticks_channels_labels[channels_to_index[c]] = "anchor"
                            ticks_rounds_labels[r] = f"{r if r != nb.basic_info.anchor_round else 'anchor'}"
                    im = ax.imshow(X, cmap="viridis", norm="log", vmin=minimum_spot_count, vmax=maximum_spot_count)
                    ax.set_xlabel("Channels")
                    ax.set_xticks(ticks_channels)
                    ax.set_xticklabels(ticks_channels_labels)
                    ax.set_yticks(ticks_rounds)
                    ax.set_yticklabels(ticks_rounds_labels)
                    ax.set_ylabel("Rounds")
                    # Create colour bar
                    cbar = ax.figure.colorbar(im, ax=ax)
                    cbar.ax.set_ylabel("Spot count", rotation=-90, va="bottom")
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)

                # Plot a z plane for each tile anchor round/channel with the detected spots shown as points.
                cmap = mpl.cm.PiYG
                for t in nb.basic_info.use_tiles:
                    fig, axes = self.create_empty_page(1, 1)
                    ax: plt.Axes = axes[0, 0]
                    anchor_r, anchor_c = nb.basic_info.anchor_round, nb.basic_info.anchor_channel
                    threshold = nb.find_spots.auto_thresh[t, anchor_r, anchor_c].item()
                    z = nb.basic_info.use_z[len(nb.basic_info.use_z) // 2]
                    fig.suptitle(f"Tile {t} anchor round/channel {z=}")
                    t_anchor_plane = nb.filter.images[t, anchor_r, anchor_c, :, :, z]
                    norm = mpl.colors.Normalize(vmin=-t_anchor_plane.max(), vmax=t_anchor_plane.max())
                    im = ax.imshow(t_anchor_plane, cmap=cmap, norm=norm)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    cbar = ax.figure.colorbar(im, ax=ax, label="Image intensity")
                    # Save the figure without spot detections first, then with after.
                    pdf.savefig(fig)
                    fig.suptitle(
                        f"Tile {t} anchor round/channel {z=} with spot detections. "
                        + f"Threshold: {'{:.2f}'.format(threshold)}"
                    )
                    spots_yxz = nb.find_spots.spot_yxz[f"t{t}r{anchor_r}c{anchor_c}"][:]
                    spots_yxz = spots_yxz[spots_yxz[:, 2] == z]
                    if spots_yxz.size > 0:
                        ax.scatter(spots_yxz[:, 1], spots_yxz[:, 0], marker="x", linewidths=1.0)
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
        pbar.update()

        pbar.set_postfix_str("register")
        pbar.update()

        pbar.set_postfix_str("call spots")
        ref_spots_filepath = os.path.join(output_dir, "_call_spots.pdf")
        if not os.path.isfile(ref_spots_filepath) and nb.has_page("ref_spots") and nb.has_page("call_spots"):
            with PdfPages(ref_spots_filepath) as pdf:
                for t in nb.basic_info.use_tiles:
                    keep = nb.ref_spots.tile[:] == t
                    fig = self.create_positions_histograms(
                        nb.call_spots.dot_product_gene_score[:][keep],
                        nb.ref_spots.local_yxz[:][keep],
                        self.tile_shape,
                        self.DEFAULT_REF_SCORE_THRESHOLD,
                        title=f"Spot position histograms for {t=}, scores "
                        + r"$\geq$"
                        + str(self.DEFAULT_REF_SCORE_THRESHOLD),
                    )
                    pdf.savefig(fig)
                    plt.close(fig)
                # Create a page for every gene
                gene_probabilities = nb.call_spots.gene_probabilities[:]
                scores = nb.ref_spots.colours[:] * nb.call_spots.colour_norm_factor[nb.ref_spots.tile[:]]
                n_genes = len(nb.call_spots.gene_names)
                gene_names = nb.call_spots.gene_names
                spot_colours_rnorm = scores / np.linalg.norm(scores, axis=2)[:, :, None]
                signs = np.sign(np.sum(spot_colours_rnorm, axis=(1, 2)))
                spot_colours_rnorm *= signs[:, None, None]
                n_rounds = spot_colours_rnorm.shape[1]
                for g in range(n_genes):
                    g_spots = np.argsort(-gene_probabilities[:, g])
                    # Sorted probabilities, with greatest score at index 0
                    g_probs = gene_probabilities[g_spots, g]
                    # Bled codes are of shape (rounds, channels, )
                    g_bled_code = nb.call_spots.bled_codes[g]
                    g_bled_code = g_bled_code / np.linalg.norm(g_bled_code, axis=1)[:, None]
                    g_r_dot_products = np.abs(np.sum(spot_colours_rnorm * g_bled_code[None, :, :], axis=2))
                    thresh_spots = np.argmax(gene_probabilities, axis=1) == g
                    thresh_spots = thresh_spots * (np.max(gene_probabilities) > self.GENE_PROB_THRESHOLD)
                    colours_mean = np.mean(scores[thresh_spots], axis=0)
                    fig, axes = self.create_empty_page(2, 2, gridspec_kw={"width_ratios": [2, 1]})
                    self.empty_plot_ticks(axes[1, 1])
                    fig.suptitle(f"{gene_names[g]}", size=self.NORMAL_FONTSIZE)
                    im = axes[0, 0].imshow(
                        g_r_dot_products[g_spots[: self.N_GENES_SHOW], :].T, vmin=0, vmax=1, aspect="auto"
                    )
                    axes[0, 0].set_yticks(range(n_rounds))
                    axes[0, 0].set_ylim([n_rounds - 0.5, -0.5])
                    axes[0, 0].set_ylabel("round")
                    axes[0, 0].set_title("dye code match")
                    self.empty_plot_ticks(axes[1, 0], show_bottom_frame=True, show_left_frame=True)
                    max_shown = g_probs[: self.N_GENES_SHOW].size
                    axes[1, 0].plot(np.arange(max_shown), g_probs[: self.N_GENES_SHOW])
                    axes[1, 0].plot(np.arange(max_shown), g_r_dot_products[g_spots[: self.N_GENES_SHOW]].mean(1))
                    axes[1, 0].legend(("probability score", "mean match"), loc="lower right")
                    for ax in [axes[0, 0], axes[1, 0]]:
                        ax.set_xlim([0, self.N_GENES_SHOW - 1])
                        ax.set_xticks([0, self.N_GENES_SHOW - 1], labels=["1", self.N_GENES_SHOW])
                    axes[1, 0].set_xlabel("spot number (ranked by probability)")
                    axes[1, 0].set_ylim([0, 1])
                    axes[1, 0].set_yticks([0, 0.25, 0.5, 0.75, 1])
                    axes[1, 0].grid(True)
                    axes[0, 0].autoscale(enable=True, axis="x", tight=True)
                    axes[0, 1].imshow(g_bled_code, vmin=0, vmax=1)
                    axes[0, 1].set_title("bled code GE")
                    axes[0, 1].set_xlabel("channels")
                    axes[0, 1].set_xticks(
                        range(len(nb.basic_info.use_channels)), labels=(str(c) for c in nb.basic_info.use_channels)
                    )
                    axes[1, 1].imshow(colours_mean, vmin=0, vmax=1)
                    axes[1, 1].set_title(f"mean spot colour\nspots with prob > {self.GENE_PROB_THRESHOLD}")
                    axes[1, 1].set_ylabel("rounds")
                    axes[1, 1].set_yticks(
                        range(len(nb.basic_info.use_rounds)), labels=(str(r) for r in nb.basic_info.use_rounds)
                    )
                    axes[1, 1].set_xlabel("channels")
                    axes[1, 1].set_xticks(
                        range(len(nb.basic_info.use_channels)), labels=(str(c) for c in nb.basic_info.use_channels)
                    )
                    cbar = fig.colorbar(im, ax=axes[0, 1], orientation="vertical")
                    cbar.ax.set_ylabel("Score", rotation=-90, va="bottom")
                    cbar = fig.colorbar(im, ax=axes[1, 1], orientation="vertical")
                    cbar.ax.set_ylabel("Score", rotation=-90, va="bottom")
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
        pbar.update()

        pbar.set_postfix_str("omp")
        omp_filepath = os.path.join(output_dir, "_omp.pdf")
        if nb.has_page("omp") and not os.path.isfile(omp_filepath):
            with PdfPages(omp_filepath) as pdf:
                fig, axes = self.create_empty_page(1, 1)
                info = self.get_omp_text_info(nb.omp)
                axes[0, 0].set_title(info, fontdict=self.INFO_FONTDICT, y=0.5)
                self.empty_plot_ticks(axes[0, 0])
                pdf.savefig(fig)
                plt.close(fig)

                fig = self.create_omp_score_distribution_fig(nb.basic_info, nb.omp)
                pdf.savefig(fig)
                plt.close(fig)

                fig = self.create_omp_spot_shape_fig(nb.omp)
                pdf.savefig(fig)
                plt.close(fig)

                for i in range(10):
                    fig = self.create_omp_gene_counts_fig(
                        nb.basic_info, nbp_file, nb.call_spots, nb.omp, score_threshold=i * 0.1
                    )
                    pdf.savefig(fig)
                    plt.close(fig)

                for t in nb.basic_info.use_tiles:
                    fig = self.create_positions_histograms(
                        nb.omp.results[f"tile_{t}/scores"][:],
                        nb.omp.results[f"tile_{t}/local_yxz"][:],
                        self.tile_shape,
                        self.DEFAULT_OMP_SCORE,
                        title=f"Spot position histograms for {t=}, scores " + r"$\geq$" + str(self.DEFAULT_OMP_SCORE),
                    )
                    pdf.savefig(fig)
                    plt.close(fig)
            plt.close(fig)
        pbar.update()

        pbar.set_postfix_str("call spots heatmaps")
        anchor_heatmap_path = os.path.join(output_dir, "_heat_maps_anchor.pdf")
        prob_heatmap_path = os.path.join(output_dir, "_heat_maps_prob.pdf")
        file_missing = not os.path.isfile(anchor_heatmap_path) or not os.path.isfile(prob_heatmap_path)
        if nb.has_page("call_spots") and file_missing:
            gene_names = nb.call_spots.gene_names
            local_yxzs = nb.ref_spots.local_yxz[:].astype(np.float32)
            tile_numbers = nb.ref_spots.tile[:]
            global_yxzs = local_yxzs + nb.stitch.tile_origin[tile_numbers]

            # Anchor heat maps.
            gene_numbers = nb.call_spots.dot_product_gene_no[:]
            scores = nb.call_spots.dot_product_gene_score[:]
            with PdfPages(anchor_heatmap_path) as pdf:
                self.create_spatial_heatmaps(
                    pdf, global_yxzs, gene_numbers, scores, gene_names, self.HEATMAP_ANCHOR_SCORE_THRESH
                )

            # Probability heat maps.
            gene_numbers = np.argmax(nb.call_spots.gene_probabilities[:], axis=1)
            scores = nb.call_spots.gene_probabilities[:].max(1)
            with PdfPages(prob_heatmap_path) as pdf:
                self.create_spatial_heatmaps(
                    pdf, global_yxzs, gene_numbers, scores, gene_names, self.HEATMAP_PROB_SCORE_THRESH
                )
        pbar.update()

        pbar.set_postfix_str("omp heatmaps")
        omp_heatmap_path = os.path.join(output_dir, "_heat_maps_omp.pdf")
        if nb.has_page("omp") and not os.path.isfile(omp_heatmap_path):
            gene_names = nb.call_spots.gene_names
            with PdfPages(omp_heatmap_path) as pdf:
                local_yxzs, tile_numbers = omp_base.get_all_local_yxz(nb.basic_info, nb.omp)
                local_yxzs = local_yxzs.astype(np.float32)
                # Convert local positions to global positions using stitch tile origins.
                global_yxzs = local_yxzs + nb.stitch.tile_origin[tile_numbers].astype(np.float32)
                scores, _ = omp_base.get_all_scores(nb.basic_info, nb.omp)
                gene_numbers, _ = omp_base.get_all_gene_no(nb.basic_info, nb.omp)
                self.create_spatial_heatmaps(
                    pdf, global_yxzs, gene_numbers, scores, gene_names, self.HEATMAP_OMP_SCORE_THRESH
                )
        pbar.update()
        pbar.close()

        log.debug("Creating diagnostic PDF complete")
        if auto_open:
            webbrowser.open_new_tab(rf"{output_dir}")

    def create_empty_page(
        self,
        nrows: int,
        ncols: int,
        hide_frames: bool = True,
        size: Tuple[float, float] = A4_SIZE_INCHES,
        share_x: bool = False,
        share_y: bool = False,
        gridspec_kw: dict = None,
    ) -> Tuple[plt.figure, np.ndarray]:
        if gridspec_kw is None:
            gridspec_kw = {}

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, squeeze=False, sharex=share_x, sharey=share_y, gridspec_kw=gridspec_kw
        )
        fig.set_size_inches(size)
        for ax in axes.ravel():
            ax.set_frame_on(not hide_frames)
        fig.tight_layout()
        return fig, axes

    def empty_plot_ticks(
        self,
        axes: Union[np.ndarray, plt.Axes],
        show_top_frame: bool = False,
        show_bottom_frame: bool = False,
        show_left_frame: bool = False,
        show_right_frame: bool = False,
    ) -> None:
        def _apply_to(axis: plt.Axes):
            axis.set_frame_on(True)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.spines["top"].set_visible(show_top_frame)
            axis.spines["bottom"].set_visible(show_bottom_frame)
            axis.spines["left"].set_visible(show_left_frame)
            axis.spines["right"].set_visible(show_right_frame)

        if isinstance(axes, np.ndarray):
            for ax in axes.ravel():
                _apply_to(ax)
        else:
            ax = axes
            _apply_to(ax)

    def get_version_from_page(self, page: NotebookPage) -> str:
        output = ""
        try:
            output += f"version: {page.software_version}\n"
            output += f"version hash: {page.revision_hash}\n\n"
        except AttributeError:
            pass
        return output

    def get_time_taken_from_page(self, page: NotebookPage) -> str:
        try:
            time_taken = page.time_taken
            time_taken = "time taken: {0} hour(s) and {1} minute(s)\n".format(
                int(time_taken // 60**2), int((time_taken // 60) % 60)
            )
        except AttributeError:
            time_taken = ""
        return time_taken

    def get_basic_info(self, basic_info_page: NotebookPage, file_names_page: NotebookPage) -> str:
        try:
            output = f"Coppafisher {basic_info_page.software_version} Diagnostics"
        except AttributeError:
            output = "Coppafisher <0.5.0 Diagnostics"
        output += "\n \n"
        use_tiles = basic_info_page.use_tiles
        output += "\n".join(textwrap.wrap(f"{len(use_tiles)} tiles: {use_tiles}", 88)) + "\n"
        output += (
            "...\n".join(
                textwrap.wrap(
                    "3D tile dimensions: "
                    + f"{basic_info_page.nz}x{basic_info_page.tile_sz}x{basic_info_page.tile_sz}",
                    85,
                )
            )
            + "\n"
        )
        output += f"sequencing rounds: {basic_info_page.use_rounds}\n"
        if basic_info_page.use_anchor:
            output += (
                f"anchor round: {basic_info_page.anchor_round}\nanchor channel: {basic_info_page.anchor_channel}\n"
            )
        output += f"channels used: {basic_info_page.use_channels}\n"
        if basic_info_page.dapi_channel is not None:
            output += f"dapi channel: {basic_info_page.dapi_channel}\n"
        try:
            output += f"version hash: {basic_info_page.revision_hash}\n"
        except AttributeError:
            pass
        input_dir = f"input directory: {file_names_page.input_dir}"
        output_dir = f"output directory: {file_names_page.output_dir}"
        wrapped_input = "...\n  ".join(textwrap.wrap(input_dir, 85))
        wrapped_output = "...\n  ".join(textwrap.wrap(output_dir, 85))
        output += f"{wrapped_input}\n"
        output += f"{wrapped_output}\n"
        return output

    def get_extract_text_info(self, extract_page: NotebookPage) -> str:
        output = "Extract\n \n"
        output += self.get_version_from_page(extract_page)
        return output

    def get_filter_info(self, filter_page: NotebookPage, filter_debug_page: Optional[NotebookPage] = None) -> str:
        output = "Filter\n \n"
        output += self.get_version_from_page(filter_page)
        if filter_debug_page is not None:
            time_taken = self.get_time_taken_from_page(filter_debug_page)
            output += time_taken
        else:
            output += "no dapi filtering"
        return output

    def create_pixel_value_hists(
        self,
        nb: Notebook,
        section_name: str,
        pixel_unique_values: np.ndarray,
        pixel_unique_counts: np.ndarray,
        pixel_min: int,
        pixel_max: int,
        bin_size: int,
        log_count: bool = True,
        auto_thresh_values: np.ndarray = None,
    ) -> list:
        assert bin_size >= 1
        assert (pixel_max - pixel_min + 1) % bin_size == 0

        figures = []
        use_channels = list(nb.basic_info.use_channels)
        if nb.basic_info.dapi_channel is not None:
            use_channels += [nb.basic_info.dapi_channel]
            use_channels.sort()
        use_channels_all = list(set(use_channels + self.use_channels_anchor))
        use_channels_all.sort()
        first_channel = use_channels[0]
        use_rounds_all = list(nb.basic_info.use_rounds)
        if nb.basic_info.use_anchor:
            use_rounds_all += [nb.basic_info.anchor_round]
        use_rounds_all = list(set(use_rounds_all))
        use_rounds_all.sort()
        final_round = use_rounds_all[-1]
        greatest_possible_y = nb.basic_info.tile_sz * nb.basic_info.tile_sz * len(nb.basic_info.use_z)
        if log_count:
            greatest_possible_y = np.log2(greatest_possible_y)
        for t in nb.basic_info.use_tiles:
            fig, axes = self.create_empty_page(
                nrows=len(use_rounds_all),
                ncols=len(set(use_channels + self.use_channels_anchor)),
                size=(self.A4_SIZE_INCHES[0] * 2, self.A4_SIZE_INCHES[1] * 2),
                share_x=True,
                share_y=True,
            )
            fig.set_layout_engine("constrained")
            fig.suptitle(
                f"{section_name} {' log of ' if log_count else ''} pixel values, {t=}",
                fontsize=self.SMALL_FONTSIZE,
            )
            for i, r in enumerate(use_rounds_all):
                if r == nb.basic_info.anchor_round:
                    use_channels_r = self.use_channels_anchor
                else:
                    use_channels_r = list(nb.basic_info.use_channels)
                    if nb.basic_info.dapi_channel is not None:
                        use_channels_r += [nb.basic_info.dapi_channel]
                for j, c in enumerate(use_channels_all):
                    ax: plt.Axes = axes[i, j]
                    self.empty_plot_ticks(ax, show_left_frame=c == first_channel, show_bottom_frame=True)
                    if c not in use_channels_r:
                        self.empty_plot_ticks(ax)
                        continue
                    hist_x = []
                    hist_loc = np.arange(pixel_max - pixel_min + 1, step=bin_size, dtype=int) + bin_size // 2
                    k = 0
                    for pixel_value in range(pixel_max + 1):
                        if pixel_value == pixel_unique_values[k]:
                            count = pixel_unique_counts[k, t, r, c]
                            hist_x.append(count)
                            k += 1
                        else:
                            hist_x.append(0)
                    if bin_size > 1:
                        new_hist_x = [0 for _ in range(len(hist_x) // bin_size)]
                        for k in range(len(hist_x) // bin_size):
                            for l in range(bin_size):
                                new_hist_x[k] += hist_x[k * bin_size + l]
                        hist_x = new_hist_x
                    if log_count:
                        for k, count in enumerate(hist_x):
                            if count == 0:
                                continue
                            hist_x[k] = np.log2(count)
                    if np.sum(hist_x) <= 0:
                        log.warn(f"The {section_name.lower()} image for {t=}, {r=}, {c=} looks to be all zeroes!")
                        continue
                    ax.bar(x=hist_loc, height=hist_x, color="red", width=bin_size)
                    ax.set_xlim(pixel_min, pixel_max)
                    # Vertical line at the auto thresh value, i.e. the detecting spots threshold
                    if auto_thresh_values is not None:
                        ax.vlines(
                            auto_thresh_values[t, r, c],
                            0,
                            greatest_possible_y,
                            linestyles="dotted",
                        )
                    # Axis labelling and ticks
                    if c == first_channel:
                        round_label = str(r)
                        if nb.basic_info.use_anchor and r == nb.basic_info.anchor_round:
                            round_label = "anchor"
                        round_label += "\n" + r"$\log_2$ count" if log_count else "count"
                        ax.set_ylabel(
                            f"round {round_label}",
                            fontdict={"fontsize": self.SMALL_FONTSIZE},
                        )
                    if r == final_round:
                        ax.set_xlabel(
                            f"channel {c if c != nb.basic_info.dapi_channel else 'dapi'}",
                            fontdict={"fontsize": self.SMALL_FONTSIZE},
                        )
                        ax.set_xticks([pixel_min, pixel_max])
            figures.append(fig)
            ax.set_ylim([0, greatest_possible_y])
        return figures

    def get_find_spots_info(self, find_spots_page: NotebookPage) -> str:
        output = "Find Spots\n \n"
        output += self.get_version_from_page(find_spots_page)
        time_taken = self.get_time_taken_from_page(find_spots_page)
        output += time_taken
        return output

    def get_omp_text_info(self, omp_page: NotebookPage) -> str:
        output = "OMP\n \n"
        output += self.get_version_from_page(omp_page)
        output += f"computed spot shape size: {omp_page.mean_spot.shape}\n"
        return output

    def create_omp_score_distribution_fig(
        self, basic_info_page: NotebookPage, omp_page: NotebookPage
    ) -> mpl.figure.Figure:
        fig, axes = self.create_empty_page(2, 2, gridspec_kw={"width_ratios": [21, 1], "height_ratios": [2, 1]})
        fig.suptitle("OMP spots")
        # Plot a bar graph of the spot count found by OMP for each z plane and tile. The colour of the bar
        # represents the mean score of the spots in that z plane and tile
        median_scores = np.zeros(len(basic_info_page.use_tiles) * len(basic_info_page.use_z))
        spot_counts = np.zeros_like(median_scores, dtype=int)
        bar_x = np.arange(0, median_scores.size, dtype=float) + 0.5
        ticks = []
        labels = []
        i = 0
        for t in basic_info_page.use_tiles:
            t_scores: np.ndarray = omp_page.results[f"tile_{t}/scores"][:]
            t_z_local: np.ndarray = omp_page.results[f"tile_{t}/local_yxz"][:, 2]
            for z in basic_info_page.use_z:
                spot_counts[i] = (t_z_local == z).sum()
                t_z_scores = t_scores[t_z_local == z]
                if t_z_scores.size > 0:
                    median_scores[i] = np.median(t_z_scores)
                if z == basic_info_page.use_z[len(basic_info_page.use_z) // 2]:
                    labels.append(f"Tile {t}")
                    ticks.append(bar_x[i])
                if z == basic_info_page.use_z[0] or z == basic_info_page.use_z[-1]:
                    labels.append("")
                    ticks.append(bar_x[i])
                i += 1
        del i
        # Create a colour map for the bars to be coloured based on the median spot score
        max_median_score = median_scores.max()
        cmap = mpl.cm.plasma
        norm = mpl.colors.Normalize(vmin=0, vmax=max_median_score)
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=axes[0, 1],
            orientation="vertical",
            label="Median score",
        )
        bar_colours = [cmap(norm(median_scores[i])) for i in range(median_scores.size)]
        axes[0, 0].set_title("Counts")
        axes[0, 0].bar(bar_x, spot_counts, width=1, color=bar_colours, edgecolor="black", linewidth=0.5)
        axes[0, 0].set_xticks(ticks, labels=labels)
        axes[0, 0].set_ylabel("Spot count")
        axes[0, 0].spines["left"].set_visible(True)
        axes[0, 0].spines["bottom"].set_visible(True)

        # Create a histogram showing the distribution of OMP spot scores
        all_scores = np.zeros(0, dtype=np.float16)
        for t in basic_info_page.use_tiles:
            all_scores = np.append(all_scores, omp_page.results[f"tile_{t}/scores"][:], 0)
        axes[1, 0].set_title("Score distribution")
        axes[1, 0].hist(all_scores, bins=200, color="red", edgecolor="black", linewidth=0.25)
        axes[1, 0].set_xlabel("Spot score")
        axes[1, 0].set_xlim(
            [
                omp_base.get_all_scores(basic_info_page, omp_page)[0].min(),
                omp_base.get_all_scores(basic_info_page, omp_page)[0].max(),
            ]
        )
        axes[1, 0].set_ylabel("Spot count")
        axes[1, 0].spines["left"].set_visible(True)
        axes[1, 0].spines["bottom"].set_visible(True)

        axes[1, 1].axis("off")
        fig.tight_layout()
        return fig

    def create_omp_spot_shape_fig(self, omp_page: NotebookPage) -> mpl.figure.Figure:
        fig, axes = self.create_empty_page(1, 4, hide_frames=False, gridspec_kw={"width_ratios": [5, 5, 5, 1]})
        if omp_page.mean_spot.shape[2] < 3:
            return fig

        cmap = mpl.cm.coolwarm
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        [
            fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=axes[row, -1],
                orientation="vertical",
                label="Pixel intensity",
            )
            for row in range(axes.shape[0])
        ]
        z_offsets = [-1, 0, +1]
        mean_spot_shape = omp_page.mean_spot
        if mean_spot_shape is not None:
            mid_z = mean_spot_shape.shape[2] // 2
            max_z = mean_spot_shape.shape[2] - 1
            for column, z_offset in enumerate(z_offsets):
                z = min([max([0, z_offset + mid_z]), max_z])
                title = "central z"
                if z_offset != 0:
                    title += f" {'+ ' if z_offset > 0 else '- '}{int(np.abs(z_offset))}"
                else:
                    title = "Mean spot shape\n" + title
                axes[0, column].set_title(title)
                axes[0, column].imshow(mean_spot_shape[:, :, z], cmap=cmap, norm=norm, aspect="equal")
                self.empty_plot_ticks(
                    axes[0, column],
                    show_top_frame=True,
                    show_bottom_frame=True,
                    show_left_frame=True,
                    show_right_frame=True,
                )

        fig.tight_layout()
        return fig

    def create_omp_gene_counts_fig(
        self,
        basic_info_page: NotebookPage,
        file_page: NotebookPage,
        call_spots_page: NotebookPage,
        omp_page: NotebookPage,
        score_threshold: float = 0,
    ) -> mpl.figure.Figure:
        """Creates a gene count bar chart. Each bar is coloured based on the median spot score of the gene."""
        fig, axes = self.create_empty_page(1, 2, gridspec_kw={"width_ratios": [24, 1]})
        ax: plt.Axes = axes[0, 0]
        labels = []
        gene_counts = []
        median_scores = []
        n_genes = call_spots_page.gene_probabilities.shape[1]
        if os.path.isfile(file_page.code_book):
            gene_names, _ = np.genfromtxt(file_page.code_book, dtype=(str, str)).transpose()
        else:
            gene_names = [f"gene_{g}" for g in range(n_genes)]

        all_scores = omp_base.get_all_scores(basic_info_page, omp_page)[0]
        all_gene_no = omp_base.get_all_gene_no(basic_info_page, omp_page)[0]
        all_gene_no = all_gene_no[all_scores >= score_threshold]
        all_scores = all_scores[all_scores >= score_threshold]
        unique_genes, counts = np.unique(all_gene_no, return_counts=True)
        for g, gene_name in enumerate(gene_names):
            if np.isin(g, unique_genes):
                gene_counts.append(int(counts[unique_genes == g].item()))
                scores_g = all_scores[all_gene_no == g]
                median_scores.append(float(np.median(scores_g)))
            else:
                gene_counts.append(0)
                median_scores.append(0)
            labels.append(gene_name)
        bar_x = np.arange(n_genes) + 0.5
        if score_threshold > 0:
            ax.set_title(r"Gene counts for scores $\geq$ " + str(round(score_threshold, 3)))
        else:
            ax.set_title("Gene counts")
        # Create a colour map for the bars to be coloured based on the median scores
        cmap = mpl.cm.viridis
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=axes[0, 1],
            orientation="vertical",
            label="Median score",
        )
        bar_colours = [cmap(norm(median_scores[i])) for i in range(n_genes)]
        ax.bar(bar_x, gene_counts, color=bar_colours, linewidth=0.9, edgecolor="black")
        ax.set_xticks(bar_x, labels, rotation=70, ha="right")
        # Apply a 5pt x offset to all x tick labels, makes gene labels better aligned with ticks
        dx, dy = 5, 0
        offset = ScaledTranslation(dx / fig.dpi, dy / fig.dpi, fig.dpi_scale_trans)
        for label in ax.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)

        fig.tight_layout()
        return fig

    def create_positions_histograms(
        self,
        scores: np.ndarray,
        local_yxz: np.ndarray,
        tile_shape: tuple[int, int, int],
        score_threshold: float = 0,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Histograms of positions x, y, and z.

        Args:
            - scores (`(n_spots) ndarray[float]`): scores of each position.
            - local_yxz (`(n_spots x 3) ndarray[int]`): local y, x, and z positions.
            - tile_shape (tuple of 3 ints): the tile's shape in y, x, and z directions.
            - score_threshold (float): score threshold. Default: 0.
            - title (str, optional): plot title.

        Returns:
            figure: positions histograms plot.
        """
        assert scores.size == local_yxz.shape[0]
        assert type(tile_shape) is tuple and len(tile_shape) == 3

        fig, axes = self.create_empty_page(3, 1, share_y=False)
        positions = {1: "x", 0: "y", 2: "z"}
        colours = ["green", "blue", "red"]
        for index, position in positions.items():
            ax: plt.Axes = axes[index, 0]
            ax.set_title(position.upper())
            ax.hist(
                local_yxz[scores >= score_threshold, index],
                color=colours[index],
                bins=tile_shape[index],
                range=(0, tile_shape[index] - 1),
                edgecolor="black",
                linewidth=0 if index < 2 else 0.4,
            )
            ax.set_ylabel("Count")
        if title is None:
            fig.suptitle(r"Position histogram for scores $\geq$" + str(score_threshold))
        else:
            fig.suptitle(title)
        fig.tight_layout()
        return fig

    def create_spatial_heatmaps(
        self: Self,
        pdf: PdfPages,
        global_yxzs: np.ndarray[np.float32],
        gene_numbers: np.ndarray[np.int32],
        scores: np.ndarray[np.float32],
        gene_names: np.ndarray[str],
        score_threshold: float,
    ) -> None:
        """
        Save a spatial heat map of each spot location for each gene along y and x directions for all z positions. This
        is saved to the given PDF file.

        Args:
            (PdfPages): pdf pages to save every figure to.
            (`(n_spots x 3) ndarray[float32]`) global_yxzs: the y, x, and z position for every spot on a global picture.
            (`(n_spots) ndarray[int32]`) gene_numbers: gene number for each spot.
            (`(n_spots) ndarray[float32]`) scores: every spot score.
            (`(n_spots) ndarray[str]`) gene_names: every gene name.
            (float) score_threshold: gene score threshold.
        """
        assert global_yxzs.shape[0] == gene_numbers.shape[0] == scores.shape[0]
        assert np.logical_and(gene_numbers >= 0, gene_numbers < gene_names.size).all()

        global_minimums_yxz = global_yxzs.min(0)
        global_maximums_yxz = global_yxzs.max(0)
        bin_counts = np.ceil(global_maximums_yxz / self.HEATMAP_BIN_SIZE)[:2].astype(int).tolist()
        hist_range = ((0, bin_counts[0] * self.HEATMAP_BIN_SIZE), (0, bin_counts[1] * self.HEATMAP_BIN_SIZE))
        cmap = mpl.cm.viridis

        for g, gene_name in enumerate(gene_names):
            spot_passes = scores >= score_threshold
            spot_gene_g = gene_numbers == g
            subset_yxzs = global_yxzs[spot_passes & spot_gene_g]
            image = np.histogram2d(subset_yxzs[:, 0], subset_yxzs[:, 1], bins=bin_counts, range=hist_range)[0]
            if (image > 0).sum() == 0:
                continue
            image[image <= 0] = np.nan
            image[image > 0] = np.log10(image[image > 0])
            max_count = max(np.nanmax(image).astype(int).item(), 1)
            norm = mpl.colors.Normalize(vmin=0, vmax=max_count)
            fig, axes = self.create_empty_page(1, 1)
            im = axes[0, 0].imshow(image, cmap=cmap, norm=norm)
            axes[0, 0].set_xlabel("")
            axes[0, 0].set_ylabel("")
            axes[0, 0].set_xticks([])
            axes[0, 0].set_yticks([])
            axes[0, 0].set_xlim(
                -0.5 + global_minimums_yxz[1] / self.HEATMAP_BIN_SIZE, global_maximums_yxz[1] / self.HEATMAP_BIN_SIZE
            )
            axes[0, 0].set_ylim(
                -0.5 + global_minimums_yxz[0] / self.HEATMAP_BIN_SIZE, global_maximums_yxz[0] / self.HEATMAP_BIN_SIZE
            )
            fig.colorbar(im, ax=axes[0, 0], label=r"$\log_{10}$(Spot count)", ticks=[n for n in range(max_count + 1)])
            fig.suptitle(f"Gene {g}: {gene_name}, score >= {score_threshold}")
            fig.tight_layout()

            pdf.savefig(fig)
            plt.close(fig)
