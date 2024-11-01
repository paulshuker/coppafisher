import numpy as np
import torch


def score_coefficient_image(
    coefficient_image: torch.Tensor,
    mean_spot: torch.Tensor,
    force_cpu: bool = True,
) -> torch.Tensor:
    """
    Computes OMP score(s) for the coefficient image. This is a weighted average of spot_shape_mean at spot_shape's == 1
    in a local area with their corresponding coefficients. This is a convolution over the entire image. Scores can
    range anywhere from 0 to infinity.

    Args:
        - coefficient_image (`(n_batches x im_y x im_x x im_z) tensor[float32]`): OMP coefficients in 3D shape. Any
            non-computed or out of bounds coefficients will be zero.
        - mean_spot (`(size_y x size_x x size_z) tensor[float32]`): OMP mean spot shape. This can range from -1 and 1.
        - force_cpu (bool): use the CPU only, never the GPU. Default: true.

    Returns:
        (`(n_batches x im_y x im_x x im_z) tensor[float32]`) score_image: OMP score for every coefficient image pixel,
            on every given batch.
    """
    assert type(coefficient_image) is torch.Tensor
    assert type(mean_spot) is torch.Tensor
    assert coefficient_image.dim() == 4
    assert coefficient_image.shape[0] < 2_000, "More than 2,000 batches given"
    assert (mean_spot >= 0).all()

    run_on = torch.device("cpu")
    if not force_cpu and torch.cuda.is_available():
        run_on = torch.device("cuda")

    coef_image = coefficient_image.detach().clone().to(device=run_on)
    mean_spot = mean_spot.to(device=run_on)

    spot_shape_kernel = mean_spot.detach().clone()
    spot_shape_kernel /= spot_shape_kernel.sum()

    spot_shape_kernel = spot_shape_kernel[np.newaxis, np.newaxis]
    coef_image = coef_image[:, np.newaxis]
    scores = torch.nn.functional.conv3d(coef_image, spot_shape_kernel, padding="same", bias=None)[:, 0]
    scores = scores.cpu().to(dtype=coefficient_image.dtype)

    return scores