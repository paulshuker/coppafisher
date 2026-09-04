import numpy as np

from coppafisher.omp import preprocessing


def test_preprocess_colours() -> None:
    rng = np.random.RandomState(0)
    n_colours = 11
    n_rounds_use = 3
    n_channels_use = 4

    colours = rng.rand(n_colours, n_rounds_use, n_channels_use).astype(np.float32)
    colours_copy = colours.copy()
    colour_norm_factor = 1 + rng.rand(n_rounds_use, n_channels_use).astype(np.float32)
    background_dot_product_threshold = 0.1
    background_subtract_percentile = 25.0

    preprocessed_colours = preprocessing.preprocess_colours(
        colours, colour_norm_factor, background_dot_product_threshold, background_subtract_percentile
    )
    assert np.allclose(colours, colours_copy)
    assert preprocessed_colours.ndim == 3
    assert preprocessed_colours.shape == colours.shape
    assert preprocessed_colours.dtype == np.float32

    # Check that a colour is subtracted correctly.
    colours[6] = 2
    colours[6, :, 0] = 1
    colours[6, :, 1] = 5
    colours[6, :, 3] = 0.3
    preprocessed_colours = preprocessing.preprocess_colours(
        colours, np.ones_like(colour_norm_factor), background_dot_product_threshold, background_subtract_percentile
    )
    assert preprocessed_colours.ndim == 3
    assert preprocessed_colours.shape == colours.shape
    assert preprocessed_colours.dtype == np.float32
    assert np.allclose(preprocessed_colours[6], 0)

    background_dot_product_threshold = 1.0
    preprocessed_colours = preprocessing.preprocess_colours(
        colours, colour_norm_factor, background_dot_product_threshold, background_subtract_percentile
    )
    assert np.allclose(preprocessed_colours, colours * colour_norm_factor[np.newaxis])
