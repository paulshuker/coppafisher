import tempfile
from os import path

import numpy as np
import torch

from ..utils import julia


def detect_spots(
    image: np.ndarray | torch.Tensor,
    remove_duplicates: bool,
    intensity_thresh: float,
    radius_xy: int,
    radius_z: int,
) -> tuple[np.ndarray, np.ndarray]:
    assert type(image) is np.ndarray or type(image) is torch.Tensor
    assert remove_duplicates
    assert type(intensity_thresh) is float
    assert image.ndim == 3
    assert radius_xy > 0
    assert radius_z > 0

    temp_dir = tempfile.TemporaryDirectory("coppafisher")

    image_filepath = path.join(temp_dir.name, "image.npy")
    np.save(image_filepath, np.array(image))

    script_filepath = path.join(path.dirname(__file__), "detect_io.jl")
    julia.run_julia(script_filepath, (image_filepath, str(intensity_thresh), str(radius_xy), str(radius_z)))

    maxima_yxz_filepath = path.join(temp_dir.name, "maxima_yxz.npy")
    maxima_intensities_filepath = path.join(temp_dir.name, "maxima_intensity.npy")

    maxima_yxz = np.load(maxima_yxz_filepath)
    maxima_intensity = np.load(maxima_intensities_filepath)

    temp_dir.cleanup()

    return maxima_yxz, maxima_intensity
