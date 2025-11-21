from typing import Tuple

import numpy as np


class DimensionReducer:
    """
    Reduces the dimension of numpy array(s) and allows you to undo the reduction.

    This is useful in some cases when z plane count is 1 and has to be removed when a function is called. If the array
    has no dimensions to reduce, then it is left unchanged.
    """

    _shape: None | Tuple[int, ...]
    _reduced_shape: None | Tuple[int, ...]

    def __init__(self) -> None:
        """
        Create a dimension reducer. It will work to reduce arrays of all the same shape.
        """
        self._shape = None
        self._reduced_shape = None

    def reduce(self, array: np.ndarray) -> np.ndarray:
        """
        Reduce any dimensions that have a count of 1.

        Args:
            array (`ndarray`): the array.

        Returns:
            (`ndarray`): array_reduced. The reduced array.
        """
        if type(array) is not np.ndarray:
            raise TypeError("Expected type ndarray")
        if self._shape is None:
            self._shape = array.shape
            self._reduced_shape = tuple([d for d in self._shape if d > 1])
        if array.shape != self._shape:
            raise ValueError(f"array must have shape {self._shape}, got {array.shape} instead")

        return array.reshape(self._reduced_shape)

    def undo(self, array: np.ndarray) -> np.ndarray:
        """
        Add back the dimensions with a count of 1.

        Args:
            array (`ndarray`): the array.

        Returns:
            (`ndarray`): array_extended. The array with additional dimensions added back.
        """
        if type(array) is not np.ndarray:
            raise TypeError("Expected type ndarray")
        if self._shape is None:
            raise ValueError("Reduce must be called first")
        if array.shape != self._reduced_shape:
            raise ValueError(f"array must have shape {self._reduced_shape}, got {array.shape} instead")

        return array.reshape(self._shape)
