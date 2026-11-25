# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Host adapter for the Toy Model.

The toy host stores every variable as a NumPy array: `(NCOLUMNS, NLEVELS)` for
`GridShape.CELL`, `(NCOLUMNS, NLEVELS + 1)` for `GridShape.INTERFACE`, and `(NCOLUMNS,)`
for `GridShape.SURFACE_CELL`, `GridShape.MODEL_TOP_CELL` and `GridShape.COLUMN`. Level
index 0 is the model top, and interface `k` is the upper interface of grid cell `k`.

Note the vertical array dimension is reverse indexed from the standard views QUACS expects.

This is intended to demonstrate that host applications can implement QUACS using their
native state data structures, and that parameterizations are agnostic to the host's
data layout.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from quacs import VariableSpec
from quacs.array_host import ArrayHostAdapter
from quacs.shapes import GridShape

NCOLUMNS = 4
NLEVELS = 3

# The toy host's storage type for each QUACS data type. Precision is the host's choice.
_STORAGE_TYPES: dict[str, np.dtype] = {
    "float": np.dtype(np.float64),
    "int": np.dtype(np.int32),
    "bool": np.dtype(np.bool_),
}


class ToyHostAdapter(ArrayHostAdapter):
    """Toy Host Model adapter.

    With `compiled=True`, parameterization functions are compiled with Numba. Otherwise
    they are called once per grid cell, column, or interface from a Python loop.
    """

    def __init__(
        self,
        ncolumns: int = NCOLUMNS,
        nlevels: int = NLEVELS,
        compiled: bool = True,
    ) -> None:
        super().__init__(ncolumns, nlevels, compiled=compiled)
        self._arrays: dict[str, np.ndarray] = {
            "t_host": np.full(self.view_shape(GridShape.CELL), 290.0),
            "surface_wind_speed": np.full(self.view_shape(GridShape.SURFACE_CELL), 3.0),
            "cos_solar_zenith_angle": np.linspace(0.25, 1.0, ncolumns),
            "time_step": np.array(600.0),
        }
        self._shapes: dict[str, GridShape] = {
            "t_host": GridShape.CELL,
            "surface_wind_speed": GridShape.SURFACE_CELL,
            "cos_solar_zenith_angle": GridShape.COLUMN,
            "time_step": GridShape.DOMAIN,
        }

    def grid_shapes(self) -> Mapping[str, GridShape]:
        return self._shapes

    def view(self, host_name: str) -> np.ndarray:
        if self._shapes[host_name].is_level_resolved:
            return self._arrays[host_name][:, ::-1]
        return self._arrays[host_name]

    def create_variable(self, spec: VariableSpec) -> None:
        self._arrays[spec.name] = np.zeros(
            self.view_shape(spec.shape), dtype=_STORAGE_TYPES[spec.dtype]
        )
        self._shapes[spec.name] = spec.shape

    def array(self, host_name: str) -> np.ndarray:
        """Host-side access to a variable, for the toy model's own code and diagnostics."""
        return self._arrays[host_name]
