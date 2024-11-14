import numpy as np

from coppafish.call_spots import dot_product


def test_dot_product():
    n_spots, n_rounds, n_channels_use, n_genes = 1, 3, 3, 1
    spot_colours = np.zeros((n_spots, n_rounds, n_channels_use), np.float32)
    spot_colours[0, 0, 0] = 1
    spot_colours[0, 0, 1] = 1
    spot_colours[0, 1, 1] = 1
    spot_colours[0, 2, 0] = 1
    bled_codes = np.zeros((n_genes, n_rounds, n_channels_use))
    bled_codes[0, 0, 0] = np.sqrt(0.1)
    bled_codes[0, 0, 1] = np.sqrt(0.9)
    bled_codes[0, 1, 1] = 1
    bled_codes[0, 2, 0] = 1
    scores = dot_product.dot_product_score(spot_colours=spot_colours, bled_codes=bled_codes)
    assert type(scores) is np.ndarray
    assert scores.shape == (n_spots, n_genes)
    assert np.allclose(scores, 1.088, atol=5e-4)


def test_gene_prob_score():
    # Test that the gene probabilities are different when kappa is varied
    rng = np.random.RandomState(0)
    n_spots = 11
    n_rounds = 3
    n_channels_use = 4
    n_genes = 5
    # Colours range from -1 to 1
    spot_colours = (rng.rand(n_spots, n_rounds, n_channels_use) - 0.5) * 2
    bled_codes = rng.rand(n_genes, n_rounds, n_channels_use)
    kappa_option = 1
    spot_colours_clone = spot_colours.copy()
    bled_codes_clone = bled_codes.copy()
    probabilities_1 = dot_product.gene_prob_score(spot_colours, bled_codes)
    assert np.allclose(spot_colours, spot_colours_clone), "Function changed spot_colours input"
    assert np.allclose(bled_codes, bled_codes_clone), "Function changed bled_codes input"
    probabilities_2 = dot_product.gene_prob_score(spot_colours, bled_codes, kappa_option)
    assert np.allclose(spot_colours, spot_colours_clone), "Function changed spot_colours input"
    assert np.allclose(bled_codes, bled_codes_clone), "Function changed bled_codes input"
    assert isinstance(probabilities_1, np.ndarray), "Expected ndarray as output"
    assert isinstance(probabilities_2, np.ndarray), "Expected ndarray as output"
    assert probabilities_1.shape == probabilities_2.shape == (n_spots, n_genes), "Expected shape (n_spots, n_genes)"
    assert not np.allclose(probabilities_1, probabilities_2), "Gene probabilities should change as kappa varies"
