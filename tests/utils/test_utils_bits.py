from coppafisher.utils import bits


def test_get_bit_positions() -> None:
    assert bits.get_bit_positions(0) == []
    assert bits.get_bit_positions(1) == [0]
    assert bits.get_bit_positions(int("1111", 2)) == [0, 1, 2, 3]
    assert bits.get_bit_positions(int("1001", 2)) == [0, 3]
    assert bits.get_bit_positions(int("1000", 2)) == [3]
    assert bits.get_bit_positions(int("1100", 2)) == [2, 3]
