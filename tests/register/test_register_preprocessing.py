import os

import numpy as np

from coppafisher.register import preprocessing as reg_pre


def test_custom_shift():
    # Set up data.
    data_filepath = os.path.join(os.path.dirname(__file__), "astronaut_data.npz")
    im = np.load(data_filepath)["arr_0"]
    shift = np.array([10, 20]).astype(int)
    im_new = reg_pre.custom_shift(im, shift)
    # check that the shape is correct
    assert im_new.shape == im.shape
    # check that the values are correct
    assert np.allclose(im_new[10:, 20:], im[:-10, :-20])
    assert np.allclose(im_new[:10, :20], 0)
