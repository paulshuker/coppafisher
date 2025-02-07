import numpy as np
import torch


def compute_intensity(colours: np.ndarray[np.floating] | torch.Tensor) -> torch.Tensor:
    """
    Compute the intensity of each given pixel colour.

    Intensity is defined as `min_over_rounds(max_over_channels(C))` where `C` is a single pixel's colour in every
    round/channel.

    Args:
        colours (`(n_pixels x n_rounds_use x n_channels_use) ndarray[float-like] or tensor[float-like]`): pixel colours
            to compute the intensity of.

    Returns:
        (`(n_pixels) tensor[colours.dtype]`): intensities. The colour intensities.
    """
    assert type(colours) in (np.ndarray, torch.Tensor)
    assert colours.ndim == 3

    if type(colours) is np.ndarray:
        intensities = torch.from_numpy(colours)
    elif type(colours) is torch.Tensor:
        intensities = colours.clone()
    return intensities.abs().max(2).values.min(1).values
