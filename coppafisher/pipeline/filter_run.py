import math as maths
import os
from typing import Tuple

import joblib
import numpy as np
import psutil
import skimage
import tqdm
import zarr

from .. import log
from ..setup.config_section import ConfigSection
from ..setup.notebook_page import NotebookPage
from ..utils import indexing, system, zarray

FILTER_DTYPE = np.float16


def run_filter(
    config: ConfigSection, nbp_file: NotebookPage, nbp_basic: NotebookPage
) -> Tuple[NotebookPage, NotebookPage]:
    """
    Read in extracted raw images, filter them, then re-save in a different location.

    Args:
        config (ConfigSection): config section obtained from 'filter' section of config file.
        nbp_file (NotebookPage): 'file_names' notebook page.
        nbp_basic (NotebookPage): 'basic_info' notebook page.

    Returns tuple containing:
        - NotebookPage: 'filter' notebook page.
        - NotebookPage: 'filter_debug' notebook page.

    Notes:
        - See `'filter'` and `'filter_debug'` sections of `notebook_page.py` file for description of variables.
    """
    filter_config = {config.name: config.to_dict()}
    nbp = NotebookPage("filter", filter_config)
    nbp_debug = NotebookPage("filter_debug", filter_config)

    log.debug("Filter started")

    indices = indexing.create(
        nbp_basic,
        include_anchor_round=True,
        include_anchor_channel=True,
        include_dapi_seq=True,
        include_dapi_anchor=True,
        include_bad_trc=False,
    )

    max_ind = np.array(indices).max(0).tolist()
    shape = (max_ind[0] + 1, max_ind[1] + 1, max_ind[2] + 1, nbp_basic.tile_sz, nbp_basic.tile_sz, len(nbp_basic.use_z))
    chunks = (1, 1, 1, min(576, nbp_basic.tile_sz), min(576, nbp_basic.tile_sz), 5)
    images = zarr.open_array(
        os.path.join(nbp_file.output_dir, "filter_images.zarr"),
        "a",
        shape=shape,
        chunks=chunks,
        fill_value=np.nan,
        zarr_version=2,
        dtype=FILTER_DTYPE,
    )
    if "completed_indices" not in images.attrs:
        images.attrs["completed_indices"] = []
    # Bad trc images are filled with zeros.
    for t, r, c in nbp_basic.bad_trc:
        images[t, r, c] = 0
        images.attrs["completed_indices"] = images.attrs["completed_indices"] + [[t, r, c]]

    if not os.path.isfile(nbp_file.psf):
        raise FileNotFoundError(f"Could not find the PSF at location {nbp_file.psf}")

    # ZYX -> YXZ.
    psf = np.load(nbp_file.psf)["arr_0"].astype(np.float32).swapaxes(0, 2).swapaxes(0, 1)
    if np.max(psf.shape[:2]) < psf.shape[2]:
        log.warn(f"The given PSF has a strange shape of yxz = {psf.shape}")
    # Normalise psf so the min is 0 and the max is 1.
    psf = psf - psf.min()
    psf = psf / psf.max()
    # The PSF is tapered at every edge using a Hanning window.
    psf = (
        psf
        * np.hanning(psf.shape[0]).reshape(-1, 1, 1)
        * np.hanning(psf.shape[1]).reshape(1, -1, 1)
        * np.hanning(psf.shape[2]).reshape(1, 1, -1)
    )
    nbp_debug.psf = psf

    batch_size: int | None = config["num_cores"]
    if batch_size is None:
        batch_size = max(1, maths.floor(system.get_available_memory() / 27))
    batch_size = min(batch_size, config["max_cores"])
    batch_count: int = maths.ceil(len(indices) / batch_size)

    current_process = psutil.Process()
    subproc_before = set([p.pid for p in current_process.children(recursive=True)])

    for batch_i in tqdm.trange(batch_count, desc="Filtering extract images", unit="batch"):
        index_min, index_max = batch_i * batch_size, min((batch_i + 1) * batch_size, len(indices))
        batch_images: list[np.np.ndarray] = []
        batch_trcs: list[tuple[int, int, int]] = []

        for t, r, c in indices[index_min:index_max]:
            if [t, r, c] in images.attrs["completed_indices"]:
                # Already saved filtered images are not re-filtered.
                continue

            file_path_raw = nbp_file.tile_unfiltered[t][r][c]
            raw_image_exists = zarray.image_exists(file_path_raw)
            if not raw_image_exists:
                raise FileNotFoundError(f"Raw, extracted file at\n\t{file_path_raw}\nnot found")

            image = zarr.open_array(file_path_raw, mode="r")[:]
            image = image.astype(np.float64)
            batch_images.append(image)
            batch_trcs.append((t, r, c))
            del image

        assert len(batch_images) == len(batch_trcs)

        if len(batch_images) == 0:
            continue

        filtered_images = joblib.Parallel(n_jobs=len(batch_images), return_as="list", timeout=60 * 20)(
            joblib.delayed(skimage.restoration.wiener)(batch_images.pop(0), psf, config["wiener_constant"], clip=False)
            for _ in range(len(batch_images))
        )

        for filtered_image, (t, r, c) in zip(filtered_images, batch_trcs):
            # All images are deconvolved, including the DAPI.
            filtered_image = filtered_image.astype(FILTER_DTYPE)
            images[t, r, c] = filtered_image
            images.attrs["completed_indices"] = images.attrs["completed_indices"] + [[t, r, c]]
            del filtered_image

    # Following the joblib leak issue at https://github.com/joblib/joblib/issues/945, any remaining process after the
    # use of joblib are explicitly killed.
    subproc_after = set([p.pid for p in current_process.children(recursive=True)])
    for subproc in subproc_after - subproc_before:
        log.debug("Killing process with pid {}".format(subproc))
        psutil.Process(subproc).terminate()

    nbp.images = images
    log.debug("Filter complete")

    return nbp, nbp_debug
