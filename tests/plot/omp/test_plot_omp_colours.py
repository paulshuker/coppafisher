import os
import tempfile

import numpy as np
import zarr

from coppafisher.plot.omp import colours
from coppafisher.setup.notebook_page import NotebookPage


def test_ViewOMPColourSum() -> None:
    tmp_dir = tempfile.TemporaryDirectory("coppafisher", delete=False)
    rng = np.random.RandomState(0)

    n_genes = 5

    nbp_basic = NotebookPage("basic_info")

    nbp_basic.use_rounds = (0, 1, 2)
    n_rounds_use = len(nbp_basic.use_rounds)
    nbp_basic.use_channels = (0, 1, 2, 3)
    n_channels_use = len(nbp_basic.use_channels)
    nbp_basic.use_tiles = (1, 2)

    nbp_call_spots = NotebookPage("call_spots")
    nbp_call_spots.gene_names = np.full(n_genes, "gene_name")
    colour_norm_factor = rng.rand(max(nbp_basic.use_tiles) + 1, max(nbp_basic.use_rounds) + 1, n_channels_use)
    colour_norm_factor = colour_norm_factor.astype(np.float32)
    nbp_call_spots.colour_norm_factor = colour_norm_factor
    bled_codes = rng.rand(n_genes, max(nbp_basic.use_rounds) + 1, n_channels_use)
    bled_codes /= np.linalg.norm(bled_codes, axis=(1, 2), keepdims=True)
    nbp_call_spots.bled_codes = bled_codes.astype(np.float32)

    nbp_omp = NotebookPage(
        "omp",
        {
            "omp": {
                "alpha": 1.0,
                "beta": 3.0,
                "max_genes": 3,
                "dot_product_threshold": 0.1,
                "background_subtract_percentile": 1.0,
            }
        },
    )
    omp_results_store = zarr.ZipStore(os.path.join(tmp_dir.name, "results.zip"))
    omp_results = zarr.group(omp_results_store)
    for t in nbp_basic.use_tiles:
        g = omp_results.create_group(f"tile_{t}")
        g.attrs.update({"minimum_intensity": 0.0})
    nbp_omp.results = omp_results

    plot = colours.ViewOMPColourSum(
        nbp_basic,
        nbp_call_spots,
        nbp_omp,
        "omp",
        np.zeros(3, int),
        1,
        rng.rand(n_rounds_use, n_channels_use),
        show=False,
    )
    plot.close()
    omp_results_store.close()
    tmp_dir.cleanup()
