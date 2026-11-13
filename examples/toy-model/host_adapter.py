# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Host adapter for the Toy Model.

The toy host stores every variable as a NumPy array: `(NCOLUMNS, NLEVELS)` for
`GridShape.CELL`, and `(NCOLUMNS,)` for `GridShape.SURFACE_CELL` and
`GridShape.MODEL_TOP_CELL`. Level index 0 is the model top. None of this is visible to
QUACS or to parameterizations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from quacs import VariableSpec
from quacs.exceptions import OverwriteVariableError, UnknownVariableError, VariableNotFoundError
from quacs.shapes import GridShape
from quacs.variables import Permissions

NCOLUMNS = 4
NLEVELS = 3

_MODEL_TOP_LEVEL = 0
_SURFACE_LEVEL = NLEVELS - 1

_SIZES: dict[GridShape, tuple[int, ...]] = {
    GridShape.CELL: (NCOLUMNS, NLEVELS),
    GridShape.SURFACE_CELL: (NCOLUMNS,),
    GridShape.MODEL_TOP_CELL: (NCOLUMNS,),
}


def _size(shape: GridShape) -> tuple[int, ...]:
    if shape not in _SIZES:
        raise NotImplementedError(f"The toy host does not support {shape.name} variables.")
    return _SIZES[shape]


class _ToyColumn:
    """One column of the toy model, with one of its grid cells selected."""

    def __init__(self) -> None:
        self.index = 0
        self.level = _SURFACE_LEVEL

    def step_toward_surface(self, count: int = 1) -> None:
        self._select(self.level + count)

    def step_toward_model_top(self, count: int = 1) -> None:
        self._select(self.level - count)

    def select_surface_cell(self) -> None:
        self.level = _SURFACE_LEVEL

    def select_model_top(self) -> None:
        self.level = _MODEL_TOP_LEVEL

    def at_surface(self, buffer: int = 0) -> bool:
        return _SURFACE_LEVEL - self.level <= buffer

    def at_model_top(self, buffer: int = 0) -> bool:
        return self.level - _MODEL_TOP_LEVEL <= buffer

    def _select(self, level: int) -> None:
        if not 0 <= level < NLEVELS:
            raise IndexError("Cannot step beyond the surface or the model top.")
        self.level = level


class _ToyVariable:
    """One toy model variable at the selected grid cell of a column."""

    def __init__(self, array: np.ndarray, column: _ToyColumn, level_resolved: bool) -> None:
        self._array = array
        self._column = column
        self._level_resolved = level_resolved

    def get(self) -> float:
        return float(self._array[self._index()])

    def set(self, value: float) -> None:
        self._array[self._index()] = value

    def _index(self) -> tuple[int, ...]:
        if self._level_resolved:
            return (self._column.index, self._column.level)
        return (self._column.index,)


class ToyHostAdapter:
    """Toy Host Model adapter.

    Parameterization functions are called once per grid cell or once per column.
    """

    def __init__(self) -> None:
        self._arrays: dict[str, np.ndarray] = {
            "t_host": np.full(_size(GridShape.CELL), 290.0),
            "surface_wind_speed": np.full(_size(GridShape.SURFACE_CELL), 3.0),
        }
        self._shapes: dict[str, GridShape] = {
            "t_host": GridShape.CELL,
            "surface_wind_speed": GridShape.SURFACE_CELL,
        }

    def canonical_name_map(self) -> dict[str, str | None]:
        return {"temperature": "t_host"}

    def for_each_cell(self, func: Callable[..., None], *host_names: str) -> None:
        column = _ToyColumn()
        variables = self._variables(host_names, column)
        level_resolved = {self._shapes[name].is_level_resolved for name in host_names}
        if len(level_resolved) != 1:
            raise ValueError(
                "The toy host requires the variables in a for_each_cell call to be either "
                f"all per grid cell or all per column: {', '.join(host_names)}"
            )
        levels = range(NLEVELS) if level_resolved.pop() else range(1)
        for index in range(NCOLUMNS):
            for level in levels:
                column.index = index
                column.level = level
                func(*variables)

    def for_each_column(
        self,
        func: Callable[..., None],
        *host_names: str,
        start: Literal["surface", "model_top"] = "surface",
    ) -> None:
        column = _ToyColumn()
        variables = self._variables(host_names, column)
        start_level = _SURFACE_LEVEL if start == "surface" else _MODEL_TOP_LEVEL
        for index in range(NCOLUMNS):
            column.index = index
            column.level = start_level
            func(column, *variables)

    def allocate_registered_variables(self, specs: list[VariableSpec]) -> None:
        """Create variables owned by parameterizations and check the rest exist.

        This stands in for a real host's field registration step. It is not part of the
        `HostAdapter` protocol.
        """
        for spec in specs:
            if spec.permissions is Permissions.OWN:
                if spec.name in self._arrays:
                    raise OverwriteVariableError(
                        spec.name, ["toy host", spec.requested_by or "unknown"]
                    )
                self._arrays[spec.name] = np.zeros(_size(spec.shape))
                self._shapes[spec.name] = spec.shape
        for spec in specs:
            if spec.name not in self._arrays:
                raise VariableNotFoundError(spec.name, spec.requested_by or "unknown")
            if spec.shape is not self._shapes[spec.name]:
                raise ValueError(
                    f"Variable '{spec.name}' is {self._shapes[spec.name].name} in the toy "
                    f"host but was requested as {spec.shape.name}."
                )

    def array(self, host_name: str) -> np.ndarray:
        """Host-side access to a variable, for the toy model's own code and diagnostics."""
        return self._arrays[host_name]

    def _variables(self, host_names: tuple[str, ...], column: _ToyColumn) -> list[_ToyVariable]:
        for name in host_names:
            if name not in self._arrays:
                raise UnknownVariableError(name, list(self._arrays))
        return [
            _ToyVariable(self._arrays[name], column, self._shapes[name].is_level_resolved)
            for name in host_names
        ]
