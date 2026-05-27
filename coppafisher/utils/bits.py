from typing import List


def get_bit_positions(n: int) -> List[int]:
    """
    Get the indices of every 1 bit in an integer in sorted order.

    Args:
        n (int): integer.

    Returns:
        (list of int): every index where n contains a 1 bit in sorted order.
    """
    positions = []
    for i in range(n.bit_length()):
        if n >> i & 1:
            positions.append(i)
    return positions
