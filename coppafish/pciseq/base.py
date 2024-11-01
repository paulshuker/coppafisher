import os
from typing import Optional

import numpy as np
import pandas as pd

from ..omp import base as omp_base
from ..call_spots import qual_check
from ..setup.notebook import Notebook
from ..setup.notebook_page import NotebookPage


def get_thresholds_page(nb: Notebook) -> NotebookPage:
    """
    Makes notebook page from thresholds section of config file.

    Args:
        nb: Notebook containing all experiment information.

    Returns:
        thresholds NotebookPage.
    """
    config = nb.get_config()["thresholds"]
    if config["intensity"] is None:
        config["intensity"] = nb.call_spots.gene_efficiency_intensity_thresh
    nbp = NotebookPage("thresholds")
    nbp.intensity = config["intensity"]
    nbp.score_ref = config["score_ref"]
    nbp.score_omp = config["score_omp"]
    nbp.score_omp_multiplier = config["score_omp_multiplier"]
    return nbp


def export_to_pciseq(
    nb: Notebook,
    method: str,
    intensity_thresh: Optional[float] = None,
    score_thresh: Optional[float] = None,
):
    """
    This saves .csv files containing plotting information for pciseq-

    - y - y coordinate of each spot in stitched coordinate system.
    - x - x coordinate of each spot in stitched coordinate system.
    - z_stack - z coordinate of each spot in stitched coordinate system (in units of z-pixels).
    - Gene - Name of gene each spot was assigned to.

    Only spots which pass `quality_threshold` are saved.
    This depends on parameters given in `config['thresholds']`.

    One .csv file is saved for each method: *omp*, *anchor*, and *prob*. The .csv file is saved into the directory
    where the notebook is located.

    Args:
        - nb (Notebook): Notebook for the experiment containing at least the *ref_spots* page.
        - method (str): 'omp', 'anchor', or 'prob'.
        - intensity_thresh (float, optional): only include spots with colour intensity > intensity_thresh. Default: no
            threshold.
        - score_thresh (float, optional): only include spots with score > score_thresh. Default: no threshold.

    """
    if intensity_thresh is None:
        intensity_thresh = -1.0
        print(f"Using no intensity threshold")
    if score_thresh is None:
        score_thresh = -1.0
        print(f"Using no score threshold")
    intensity_thresh = float(intensity_thresh)
    score_thresh = float(score_thresh)
    assert type(intensity_thresh) is float, "Floating point intensity_thresh required"
    assert type(score_thresh) is float, "Floating point score_thresh required"
    if method.lower() != "omp" and method.lower() != "anchor" and method.lower() != "prob":
        raise ValueError(f"method must be 'omp', 'anchor' or 'prob' but {method} given.")
    file_path = os.path.join(os.path.dirname(nb.directory), f"pciseq_{method.lower()}.csv")
    page_name = "omp" if method.lower() == "omp" else "ref_spots"
    if not nb.has_page(page_name):
        raise ValueError(f"Notebook does not contain {page_name} page.")
    if os.path.isfile(file_path):
        raise FileExistsError(f"File already exists: {file_path}")
    qual_ok = qual_check.quality_threshold(nb, method, intensity_thresh, score_thresh)

    # get coordinates in stitched image
    if page_name == "omp":
        global_spot_yxz, spot_tile_no = omp_base.get_all_local_yxz(nb.basic_info, nb.omp)
        global_spot_yxz = global_spot_yxz.astype(np.float32) + nb.stitch.tile_origin[spot_tile_no]
    else:
        global_spot_yxz = (
            nb.__getattribute__(page_name).local_yxz + nb.stitch.tile_origin[nb.__getattribute__(page_name).tile]
        )
    if method.lower() == "omp":
        spot_gene, _ = omp_base.get_all_gene_no(nb.basic_info, nb.omp)
    elif method.lower() == "ref" or method.lower() == "anchor":
        spot_gene = nb.call_spots.gene_names[nb.call_spots.dot_product_gene_no]
    elif method.lower() == "prob":
        spot_gene = nb.call_spots.gene_names[np.argmax(nb.call_spots.gene_probabilities, axis=1)]
    spot_gene = spot_gene[qual_ok]
    global_spot_yxz = global_spot_yxz[qual_ok]
    df_to_export = pd.DataFrame(data=global_spot_yxz, index=spot_gene, columns=["y", "x", "z_stack"])
    df_to_export["Gene"] = df_to_export.index
    df_to_export.to_csv(file_path, index=False)
    print(f"pciSeq file saved for method {method} at " + file_path)
