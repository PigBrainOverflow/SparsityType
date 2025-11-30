from __future__ import annotations
from typing import Any, cast
import numpy as np
import scipy.sparse as sp


class NDArray:
    THRESHOLD_DENSITY = 0.4  # Threshold density to switch between dense and sparse
    DEBUG = True
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
        array._data = np.array(data)    # default to dense representation
        return array

    def _adapt(self):
        if self._data is None:
            return
        if isinstance(self._data, np.ndarray):  # dense
            if self._nz_ub is None or self._nz_ub > self.size * self.THRESHOLD_DENSITY:
                # cannot determine whether it is too sparse
                self.count_nz()
            if self._nz_ub is not None and self._nz_ub <= self.size * self.THRESHOLD_DENSITY:
                self._sparsify()
        elif isinstance(self._data, sp.spmatrix):  # sparse
            if self._nz_lb is None or self._nz_lb < self.size * self.THRESHOLD_DENSITY:
                # cannot determine whether it is too dense
                self.count_nz()
            if self._nz_lb is not None and self._nz_lb > self.size * self.THRESHOLD_DENSITY:
                self._densify()

    def _densify(self):
        if isinstance(self._data, sp.spmatrix):
            self._data = self._data.todense()
            if self.DEBUG:
                print("Densified")

    def _sparsify(self):
        if isinstance(self._data, np.ndarray) and self._data.ndim == 2: # only 2D arrays
            self._data = sp.csr_matrix(self._data)
            if self.DEBUG:
                print("Sparsified")

    def __repr__(self) -> str:
        return f"NDArray(shape={self.shape}, dtype={self.dtype}, nz_lb={self._nz_lb}, nz_ub={self._nz_ub}), sparse={isinstance(self._data, sp.spmatrix)}" + f"\ndata:\n{self._data}"

    """
    Arithmetic Operations
    """
    def _add_scalar(self, scalar: Any) -> NDArray:
        self._adapt()
        if self._data is None:
            raise ValueError("Data is not initialized.")
        res_data = self._data + scalar
        if isinstance(res_data, np.ndarray):
            return NDArray.from_dense(res_data)
        elif isinstance(res_data, sp.spmatrix):
            return NDArray.from_sparse(res_data)
        raise ValueError("Unsupported data type after addition.")

    def _add_array(self, other: NDArray) -> NDArray:
        self._adapt()
        other._adapt()
        if self._data is None or other._data is None:
            raise ValueError("Data is not initialized.")
        if self.shape != other.shape:
            raise ValueError("Shapes do not match for addition.")
        res_data = cast(np.ndarray, self._data) + cast(np.ndarray, other._data) # type: ignore
        if isinstance(res_data, np.ndarray):
            res = NDArray.from_dense(res_data)
        elif isinstance(res_data, sp.spmatrix):
            res = NDArray.from_sparse(res_data)
        # estimate nonzero bounds
        res._nz_ub = self._nz_ub + other._nz_ub if self._nz_ub is not None and other._nz_ub is not None else None
        res._nz_lb = abs(self._nz_lb - other._nz_lb) if self._nz_lb is not None and other._nz_lb is not None else None
        return res

    def __add__(self, other: Any) -> NDArray:
        if isinstance(other, NDArray):
            return self._add_array(other)
        return self._add_scalar(other)