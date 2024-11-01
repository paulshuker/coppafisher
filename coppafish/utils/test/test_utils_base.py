import itertools

from coppafish.utils import base


def test_deep_convert():
    assert base.deep_convert(list()) == tuple()
    assert base.deep_convert([[0, 2, 3], [1, 4], []]) == ((0, 2, 3), (1, 4), tuple())
    assert base.deep_convert(([0, 2, 3], [1, 4], [])) == ((0, 2, 3), (1, 4), tuple())
    assert base.deep_convert([[0, 2, [0]], [1, 4], []]) == ((0, 2, (0,)), (1, 4), tuple())
    assert base.deep_convert([[0, 2, (0,)], [1, 4], []]) == ((0, 2, (0,)), (1, 4), tuple())
    assert base.deep_convert([["fg" * 400], ["a" * 300]]) == (("fg" * 400,), ("a" * 300,))
    assert base.deep_convert(((0, 1), (0,)), list) == [[0, 1], [0]]
    assert base.deep_convert([], list) == []
    assert base.deep_convert([], tuple) == tuple()


def test_reed_solomon_codes():
    n_channels_try = [3, 4]
    n_rounds_try = [3, 4, 5, 6]
    for n_channels, n_rounds in itertools.product(n_channels_try, n_rounds_try):
        codes = base.reed_solomon_codes(4, n_rounds, n_channels)
        assert len(codes) == len(set(codes)), "All reed solomon codes must be unique"
