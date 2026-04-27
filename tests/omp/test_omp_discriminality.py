import numpy as np
import torch

from coppafisher.omp import discriminality


def test_score() -> None:
    pass


def test_spearman_score() -> None:
    rng = np.random.RandomState(0)
    xs = rng.rand(3, 4, 5).astype(np.float32)
    ys = rng.rand(3, 2, 5).astype(np.float32)
    results = discriminality.spearman_score(torch.from_numpy(xs), torch.from_numpy(ys))
    expected_results = np.array(
        [
            [[-0.3, 0.3], [0.0, 0.6], [-0.2, 0.7], [0.8, -0.7]],
            [[-0.2, -0.1], [0.9, -0.3], [-0.6, 0.8], [0.0, 0.1]],
            [[-0.6, -0.3], [0.3, 0.4], [-1.0, -0.9], [0.3, 0.1]],
        ],
        np.float32,
    )

    assert type(results) is torch.Tensor
    assert results.dtype == torch.float32
    assert results.shape == (3, 4, 2)
    assert np.allclose(results.numpy(), expected_results)
