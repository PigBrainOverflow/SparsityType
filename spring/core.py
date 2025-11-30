from __future__ import annotations
from typing import Any
import numpy as np
import scipy.sparse as sp


class NDArray:
    THRESHOLD_DENSITY = 0.4  # Threshold density to switch between dense and sparse
    # DENSE: type = np.ndarray
    # SPARSE: type = sp.spmatrix
    _data: np.ndarray | sp.spmatrix | None
    _nz_ub: int | None  # Upper bound on number of nonzeros, or None if unknown
    _nz_lb: int | None  # Lower bound on number of nonzeros, or None if unknown

    @property
    def shape(self) -> tuple[int, ...]:
        if self._data is None:
            raise ValueError("Data is not initialized.")
        return self._data.shape

    @property
    def size(self) -> int:
        if isinstance(self._data, np.ndarray):
            return self._data.size
        elif isinstance(self._data, sp.spmatrix):
            return self._data.shape[0] * self._data.shape[1]
        raise ValueError("Data is not initialized.")

    @property
    def dtype(self) -> Any:
        if self._data is None:
            raise ValueError("Data is not initialized.")
        dtype = getattr(self._data, "dtype", None)
        if dtype is None:
            raise ValueError("Data type information is not available.")
        return dtype

    def count_nz(self):
        if self._nz_lb is not None and self._nz_ub is not None and self._nz_lb == self._nz_ub:
            return
        if isinstance(self._data, np.ndarray):
            self._nz_lb = self._nz_ub = int(np.count_nonzero(self._data))
        elif isinstance(self._data, sp.spmatrix):
            self._nz_lb = self._nz_ub = int(self._data.getnnz())
        else:
            raise ValueError("Unsupported data type for counting nonzeros.")

    def __init__(self):
        self._data = None
        self._nz_ub = None
        self._nz_lb = None

    @classmethod
    def from_dense(cls, data: np.ndarray) -> NDArray:
        array = cls()
        array._data = data
        return array

    @classmethod
    def from_sparse(cls, data: sp.spmatrix) -> NDArray:
        array = cls()
        array._data = data
        return array

    @classmethod
    def from_obj(cls, data: Any) -> NDArray:
        array = cls()
        array._data = np.array(data)    # Default to dense representation
        return array

    def __add__(self, other: NDArray) -> NDArray:
        if not isinstance(other, NDArray):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError("Shapes of the arrays must be the same for addition.")
        # check self's density
