# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Host adapter for the Toy Model.

The toy host stores every variable as a NumPy array: `(NCOLUMNS, NLEVELS)` for
`GridShape.CELL`, `(NCOLUMNS, NLEVELS + 1)` for `GridShape.INTERFACE`, and `(NCOLUMNS,)`
for `GridShape.SURFACE_CELL`, `GridShape.MODEL_TOP_CELL` and `GridShape.COLUMN`. Level
index 0 is the model top, and interface `k` is the upper interface of grid cell `k`.

This is intended to demonstrate that host applications can implement QUACS using their
native state data structures, and that parameterizations are agnostic to the host's
data layout.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from quacs import VariableSpec
from quacs.accessors import Adjacent
from quacs.exceptions import (
    OverwriteVariableError,
    ReadOnlyVariableError,
    UnknownVariableError,
    VariableNotFoundError,
)
from quacs.shapes import GridShape
from quacs.variables import Permissions

NCOLUMNS = 4
NLEVELS = 3

_MODEL_TOP_LEVEL = 0
_SURFACE_LEVEL = NLEVELS - 1

_SIZES: dict[GridShape, tuple[int, ...]] = {
    GridShape.CELL: (NCOLUMNS, NLEVELS),
    GridShape.INTERFACE: (NCOLUMNS, NLEVELS + 1),
    GridShape.SURFACE_CELL: (NCOLUMNS,),
    GridShape.MODEL_TOP_CELL: (NCOLUMNS,),
    GridShape.COLUMN: (NCOLUMNS,),
}

# How a variable with one value per grid cell or per interface must be named in each
# operation. Every other variable is named without a selector.
_SELECTORS: dict[tuple[str, GridShape], tuple[str | None, ...]] = {
    ("cell", GridShape.CELL): (None,),
    ("cell", GridShape.INTERFACE): ("upper", "lower"),
    ("column", GridShape.CELL): (None,),
    ("column", GridShape.INTERFACE): ("upper", "lower"),
    ("interface", GridShape.CELL): ("above", "below"),
    ("interface", GridShape.INTERFACE): (None,),
}

# Level index of a variable relative to the selected grid cell or interface.
_LEVEL_OFFSETS: dict[str | None, int] = {None: 0, "upper": 0, "lower": 1, "above": -1, "below": 0}

_READ_ONLY_REASONS = {
    "cell": (
        "this function is applied to every grid cell, where only variables with one value "
        "per grid cell can be set. Use for_each_column to set it."
    ),
    "interface": (
        "this function is applied to every interface, where only variables with one value "
        "per interface can be set. Use for_each_column to set it."
    ),
}

# The toy host's storage type for each QUACS data type. Precision is the host's choice.
_STORAGE_TYPES: dict[str, np.dtype] = {
    "float": np.dtype(np.float64),
    "int": np.dtype(np.int32),
    "bool": np.dtype(np.bool_),
}


def _size(shape: GridShape) -> tuple[int, ...]:
    if shape not in _SIZES:
        raise NotImplementedError(f"The toy host does not support {shape.name} variables.")
    return _SIZES[shape]


def _name(variable: str | Adjacent) -> str:
    return variable.name if isinstance(variable, Adjacent) else variable


def _matches(value: object, kind: str) -> bool:
    """Whether a Python value has the type stored by a NumPy array of the given kind."""
    if kind == "b":
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    if kind == "i":
        return isinstance(value, int)
    return isinstance(value, (int, float))


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
    """One toy model variable at a location relative to the selected grid cell or interface."""

    def __init__(
        self,
        name: str,
        array: np.ndarray,
        position: _ToyColumn,
        level_offset: int | None,
        read_only_reason: str | None,
    ) -> None:
        self._name = name
        self._array = array
        self._position = position
        self._level_offset = level_offset
        self._read_only_reason = read_only_reason

    def get(self) -> float | int | bool:
        value: float | int | bool = self._array[self._index()].item()
        return value

    def set(self, value: float | int | bool) -> None:  # noqa: PYI041
        if self._read_only_reason is not None:
            raise ReadOnlyVariableError(self._name, self._read_only_reason)
        if not _matches(value, self._array.dtype.kind):
            raise TypeError(
                f"Variable '{self._name}' holds {self._array.dtype} values and cannot be "
                f"set to {value!r}."
            )
        self._array[self._index()] = value

    def exists(self) -> bool:
        """Whether there is a grid cell or interface at the variable's location."""
        if self._level_offset is None:
            return True
        return 0 <= self._position.level + self._level_offset < self._array.shape[1]

    def _index(self) -> tuple[int, ...]:
        if self._level_offset is None:
            return (self._position.index,)
        return (self._position.index, self._position.level + self._level_offset)


class ToyHostAdapter:
    """Toy Host Model adapter.

    Parameterization functions are called once per grid cell, column, or interface.
    """

    def __init__(self) -> None:
        self._arrays: dict[str, np.ndarray] = {
            "t_host": np.full(_size(GridShape.CELL), 290.0),
            "surface_wind_speed": np.full(_size(GridShape.SURFACE_CELL), 3.0),
            "cos_solar_zenith_angle": np.linspace(0.25, 1.0, NCOLUMNS),
        }
        self._shapes: dict[str, GridShape] = {
            "t_host": GridShape.CELL,
            "surface_wind_speed": GridShape.SURFACE_CELL,
            "cos_solar_zenith_angle": GridShape.COLUMN,
        }

    def canonical_name_map(self) -> dict[str, str | None]:
        return {"temperature": "t_host"}

    def for_each_cell(self, func: Callable[..., None], *variables: str | Adjacent) -> None:
        self._check_known(variables)
        position = _ToyColumn()
        accessors = self._accessors(variables, position, "cell", GridShape.CELL)
        for index in range(NCOLUMNS):
            for level in range(NLEVELS):
                position.index = index
                position.level = level
                func(*accessors)

    def for_each_column(
        self,
        func: Callable[..., None],
        *variables: str | Adjacent,
        start: Literal["surface", "model_top"] = "surface",
    ) -> None:
        self._check_known(variables)
        column = _ToyColumn()
        accessors = self._accessors(variables, column, "column", None)
        start_level = _SURFACE_LEVEL if start == "surface" else _MODEL_TOP_LEVEL
        for index in range(NCOLUMNS):
            column.index = index
            column.level = start_level
            func(column, *accessors)

    def for_each_interface(self, func: Callable[..., None], *variables: str | Adjacent) -> None:
        self._check_known(variables)
        # Here the position's level is an interface index, not a grid cell index.
        position = _ToyColumn()
        accessors = self._accessors(variables, position, "interface", GridShape.INTERFACE)
        for index in range(NCOLUMNS):
            for interface in range(NLEVELS + 1):
                position.index = index
                position.level = interface
                func(*(accessor if accessor.exists() else None for accessor in accessors))

    def create_variables(self, specs: list[VariableSpec]) -> None:
        """Create every variable in `specs` that the toy host does not already have.
        
        For use with the QUACS profiler.
        """
        for spec in specs:
            if spec.name not in self._arrays:
                self._arrays[spec.name] = np.zeros(
                    _size(spec.shape), dtype=_STORAGE_TYPES[spec.dtype]
                )
                self._shapes[spec.name] = spec.shape

    def grid_cell_count(self) -> int:
        return NCOLUMNS * NLEVELS

    def domain_mean(self, host_name: str) -> float:
        return float(self._arrays[host_name].mean())

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
                self._arrays[spec.name] = np.zeros(
                    _size(spec.shape), dtype=_STORAGE_TYPES[spec.dtype]
                )
                self._shapes[spec.name] = spec.shape
        for spec in specs:
            if spec.name not in self._arrays:
                raise VariableNotFoundError(spec.name, spec.requested_by or "unknown")
            if spec.shape is not self._shapes[spec.name]:
                raise ValueError(
                    f"Variable '{spec.name}' is {self._shapes[spec.name].name} in the toy "
                    f"host but was requested as {spec.shape.name}."
                )
            if self._arrays[spec.name].dtype != _STORAGE_TYPES[spec.dtype]:
                raise ValueError(
                    f"Variable '{spec.name}' is {self._arrays[spec.name].dtype} in the toy "
                    f"host but was requested as {spec.dtype}."
                )

    def array(self, host_name: str) -> np.ndarray:
        """Host-side access to a variable, for the toy model's own code and diagnostics."""
        return self._arrays[host_name]

    def _check_known(self, variables: tuple[str | Adjacent, ...]) -> None:
        for variable in variables:
            if _name(variable) not in self._arrays:
                raise UnknownVariableError(_name(variable), list(self._arrays))

    def _accessors(
        self,
        variables: tuple[str | Adjacent, ...],
        position: _ToyColumn,
        operation: str,
        settable: GridShape | None,
    ) -> list[_ToyVariable]:
        """Build one accessor per variable. If `settable` is given, only variables of that
        shape can be set."""
        accessors = []
        for variable in variables:
            name = _name(variable)
            side = variable.side if isinstance(variable, Adjacent) else None
            shape = self._shapes[name]
            allowed = _SELECTORS.get((operation, shape), (None,))
            if side not in allowed:
                expected = " or ".join(f"{s}()" for s in allowed if s) or "no selector"
                raise ValueError(
                    f"In for_each_{operation}, the {shape.name} variable '{name}' must be "
                    f"named with {expected}."
                )
            level_offset = _LEVEL_OFFSETS[side] if shape.is_level_resolved else None
            read_only = settable is not None and shape is not settable
            accessors.append(
                _ToyVariable(
                    name,
                    self._arrays[name],
                    position,
                    level_offset,
                    _READ_ONLY_REASONS[operation] if read_only else None,
                )
            )
        return accessors
