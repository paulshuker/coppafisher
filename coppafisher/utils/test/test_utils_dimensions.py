import numpy as np
import pytest

from coppafisher.utils.dimensions import DimensionReducer


def test_DimensionReducer() -> None:
    rng = np.random.RandomState(0)
    a = rng.randint(100, size=(5, 1, 2, 1, 6)).astype(np.int16)
    c = rng.randint(100, size=(5, 1, 2, 6)).astype(np.int16)

    reducer = DimensionReducer()
    with pytest.raises(TypeError):
        reducer.reduce(0)
    b = reducer.reduce(a)
    assert type(b) is np.ndarray
    assert b.shape == tuple([d for d in a.shape if d > 1])
    assert b.dtype == np.int16
    assert (a.reshape(b.shape) == b).all()
    with pytest.raises(ValueError):
        reducer.reduce(b)
    with pytest.raises(ValueError):
        reducer.reduce(c)
    a_undone = reducer.undo(b)
    assert type(a_undone) is np.ndarray
    assert a_undone.shape == a.shape
    assert a_undone.dtype == np.int16
    assert (a_undone == a).all()
