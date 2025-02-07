import numpy as np
import torch

from .. import intensity


def test_compute_intensity() -> None:
    rng = np.random.RandomState(0)

    n_pixels = 4
    n_rounds_use = 6
    n_channels_use = 9
    colours = rng.rand(n_pixels, n_rounds_use, n_channels_use).astype(np.float32)

    intensities = intensity.compute_intensity(colours)

    assert type(intensities) is torch.Tensor
    assert intensities.dtype == torch.float32
    assert intensities.shape == (n_pixels,)
    assert np.allclose(intensities.numpy(), np.abs(colours).max(2).min(1))
    assert torch.allclose(intensities, intensity.compute_intensity(torch.from_numpy(colours)))
