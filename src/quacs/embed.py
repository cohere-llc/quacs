# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Runs QUACS inside a host application that is not written in Python.

The host application calls the C functions declared in `embed/quacs_embed.h`. Those
functions start Python and pass every call on to an `EmbeddedRuntime`, which holds any
information about the simulation needed by QUACS.

The simulation is described by a Python configuration file that defines:

- `parameterizations`: a list of parameterization objects
- `hooks`: a `dict` that holds the names of the parameterizations to run for each hook label,
           in order.

The configuration can also specify to exclude Numba compilation with `compiled = False`.

The host application creates and owns every variable. It provides QUACS access to variable
memory by calling `bind_variable`. Host state data are operated on in-place.
"""

from __future__ import annotations

import runpy
import sys
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np

from quacs.array_host import ArrayHostAdapter
from quacs.conformance import CellCheck, Check, ConformanceReport, tag_columns, verify
from quacs.exceptions import HostConfigError, UnknownVariableError
from quacs.hooks import HookDispatcher
from quacs.parameterization import Parameterization
from quacs.registry import ParameterizationRegistry
from quacs.shapes import GridShape
from quacs.state import QuacsState
from quacs.variables import Permissions, VariableSpec


class Storage(IntEnum):
    """How the host application stores the values of a variable.
    
    The numbers match `quacs_storage` in `quacs_embed.h`.
    """

    FLOAT32 = 1
    FLOAT64 = 2
    INT32 = 3
    INT64 = 4
    BOOL = 5


_STORAGE_TYPES: dict[Storage, np.dtype[Any]] = {
    Storage.FLOAT32: np.dtype(np.float32),
    Storage.FLOAT64: np.dtype(np.float64),
    Storage.INT32: np.dtype(np.int32),
    Storage.INT64: np.dtype(np.int64),
    Storage.BOOL: np.dtype(np.bool_),
}

# Variable type identifiers. The `int` values match `quacs_shape`,
# `quacs_dtype`, and `quacs_permissions` in `quacs_embed.h`.
_SHAPE_CODES: dict[GridShape, int] = {
    GridShape.CELL: 1,
    GridShape.INTERFACE: 2,
    GridShape.SURFACE_CELL: 3,
    GridShape.MODEL_TOP_CELL: 4,
    GridShape.COLUMN: 5,
    GridShape.DOMAIN: 6,
}
_DTYPE_CODES: dict[str, int] = {"float": 1, "int": 2, "bool": 3}
_PERMISSION_CODES: dict[Permissions, int] = {
    Permissions.READ_ONLY: 1,
    Permissions.UPDATE: 2,
    Permissions.OWN: 3,
}


class _Memory:
    """Host memory description for use in creating Numpy wrappers."""

    def __init__(
        self, address: int, dtype: np.dtype[Any], shape: tuple[int, ...], strides: tuple[int, ...]
    ) -> None:
        self.__array_interface__ = {
            "version": 3,
            "data": (address, False),
            "typestr": dtype.str,
            "shape": shape,
            "strides": strides,
        }


class BoundHostAdapter(ArrayHostAdapter):
    """A host adapter for variables that stay in the memory of the host application."""

    def __init__(
        self, ncolumns: int, nlevels: int, name_map: dict[str, str | None], compiled: bool = True
    ) -> None:
        super().__init__(ncolumns, nlevels, compiled=compiled)
        self._name_map = name_map
        self._arrays: dict[str, np.ndarray] = {}
        self._shapes: dict[str, GridShape] = {}

    def canonical_name_map(self) -> dict[str, str | None]:
        return self._name_map

    def grid_shapes(self) -> dict[str, GridShape]:
        """Returns the shape of every variable that has been bound, by host name."""
        return self._shapes

    def view(self, host_name: str) -> np.ndarray:
        return self._arrays[host_name]

    def create_variable(self, spec: VariableSpec) -> None:
        raise HostConfigError(
            f"Variable '{spec.name}' must be created by the host application and bound."
        )

    def bind(self, spec: VariableSpec, array: np.ndarray) -> None:
        """Uses `array` as the variable described by `spec`, replacing and previously
        associated array.
        
        Raises `ValueError` if the array does not have the variable's shape or type. The
        variable is then left without an associated array.
        """
        self._arrays[spec.name] = array
        self._shapes[spec.name] = spec.shape
        try:
            self._check(spec)
        except ValueError:
            del self._arrays[spec.name], self._shapes[spec.name]
            raise


class EmbeddedRuntime:
    """Everything QUACS knows about the current simulation in a host application.
    
    The host application uses instances of this class as follows, in order:

    1. Create an instance of `EmbeddedRuntime` with the configuration defining the
       parameterizations, hooks and which parameterizations to run in order, and
       optional QUACS settings including the `compiled` flag;
    2. `map_name` to associate each host variable with a known QUACS name;
    3. `variables` to find out which variables to create, and which to look up;
    4. `set_grid` once the size of the grid is known;
    5. `bind_variable` for every variable used by any QUACS parameterization in
       configuration. Called once during initialization, and each time arrays
       change location in memory.
    6. `run_hook` for each QUACS hook embedded in the host application.
    """

    def __init__(self, config_path: str) -> None:
        self._parameterizations, hooks, self._compiled = _load_config(config_path)
        self._registry = ParameterizationRegistry()
        for parameterization in self._parameterizations:
            self._registry.register(parameterization)
        self._dispatcher = HookDispatcher(self._registry, hooks)
        self._name_map: dict[str, str | None] = {}
        self._variables: dict[str, VariableSpec] | None = None
        self._host: BoundHostAdapter | None = None
        self._started = False

    def map_name(self, name: str, host_name: str) -> None:
        """Associates a host application's name for a variable with the QUACS variable name."""
        if self._variables is not None:
            raise HostConfigError("Names must be mapped before the variables are listed.")
        self._name_map[name] = host_name

    def variables(self) -> list[VariableSpec]:
        """Returns the variables used by the configured parameterizations, one for each
        host name.
        
        The host application must create each variable that has `OWN` permissions, and
        must already have every other one. After this, no more names can be mapped.
        """
        if self._variables is None:
            self._variables = _merge(self._registry.collect_variable_specs(self._name_map))
        return list(self._variables.values())

    def variable_info(self, index: int) -> tuple[str, str, str, int, int, int, bool]:
        """Describes a variable in the form the C interface uses."""
        spec = self.variables()[index]
        return(
            spec.name,
            spec.units,
            spec.description,
            _SHAPE_CODES[spec.shape],
            _DTYPE_CODES[spec.dtype],
            _PERMISSION_CODES[spec.permissions],
            spec.transported,
        )

    def set_grid(self, ncolumns: int, nlevels: int) -> None:
        """Sets the number of columns and number of grid cells per column.
        
        This cannot change after being set.
        """
        if self._host is not None:
            raise HostConfigError("The grid has already been set.")
        if ncolumns < 1 or nlevels < 1:
            raise ValueError(f"A grid of {ncolumns} columns and {nlevels} levels has no cells.")
        self.variables()
        self._host = BoundHostAdapter(ncolumns, nlevels, self._name_map, self._compiled)

    def bind_variable(
        self,
        name: str,
        address: int,
        storage: int,
        column_stride: int = 0,
        level_stride: int = 0,
    ) -> None:
        """Associates a host-owned in-memory array with a QUACS variable.
        
        `address` is where the value for the first column is, at the surface for variables
        that are per-grid-cell or per-interface.
        `column_stride` and `level_stride` are the distances in bytes from `address` to
        the next column and next level up, respectively. Either can be negative. `level_stride`
        is ignored for a variable without levels; both strides are ignored for variables with
        one value for the entire domain.
        
        A `bool` variable can be stored as integers that are 0 (false) or 1 (true).
        """
        host = self._grid()
        assert self._variables is not None
        spec = self._variables.get(name)
        if spec is None:
            raise UnknownVariableError(name, list(self._variables))
        if address == 0:
            raise ValueError(f"Variable '{name}' was bound to a null address.")
        if address < 0:
            raise ValueError(f"Invalid address for variable '{name}': {address}.")
        dtype = _STORAGE_TYPES[Storage(storage)]
        if spec.dtype == "bool" and dtype.kind == "i":
            # Only the byte that holds the 0 or 1 is used.
            if sys.byteorder == "big":
                address += dtype.itemsize - 1
            dtype = _STORAGE_TYPES[Storage.BOOL]
        shape = host.view_shape(spec.shape)
        strides = (column_stride, level_stride)[: len(shape)]
        host.bind(spec, np.asarray(_Memory(address, dtype, shape, strides)))

    def run_hook(self, label: str) -> None:
        """Runs the parameterizations configured for hook `label`, in order."""
        host = self._grid()
        if not self._started:
            assert self._variables is not None
            if unbound := sorted(set(self._variables) - set(host.grid_shapes())):
                raise HostConfigError(f"These variables have not been bound: {', '.join(unbound)}")
            if any(isinstance(check, CellCheck) for check in self._parameterizations):
                tag_columns(host.view)
            self._started = True
        self._dispatcher.run(label, QuacsState(host, self._name_map))

    def check_conformance(self) -> ConformanceReport:
        """Reports on the conformance checks among the configured parameterizations."""
        checks = [check for check in self._parameterizations if isinstance(check, Check)]
        return verify(checks, self._grid().view)

    def _grid(self) -> BoundHostAdapter:
        if self._host is None:
            raise HostConfigError("The grid has not been set.")
        return self._host


def _load_config(path: str) -> tuple[list[Parameterization], dict[str, list[str]], bool]:
    """Runs a QUACS configuration script and returns its parameterizations, hooks, and `compiled`.
    
    The folder of the file is added to the module search path, so that the file can
    import parameterizations that are kept alongside it.
    """
    folder = str(Path(path).resolve().parent)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    try:
        config = runpy.run_path(str(path))
    except Exception as error:
        raise HostConfigError(
            f"The configuration file '{path}' could not be run: {type(error).__name__}: {error}"
        )
    missing = [name for name in ("parameterizations", "hooks") if name not in config]
    if missing:
        raise HostConfigError(
            f"The configuration file '{path}' does not define: {', '.join(missing)}."
        )
    return (
        list(config["parameterizations"]),
        dict(config["hooks"]),
        bool(config.get("compiled", True)),
    )

def _merge(specs: list[VariableSpec]) -> dict[str, VariableSpec]:
    """Combines the requests for each variable into one, keeping the strongest permissions."""
    strength = {Permissions.READ_ONLY: 0, Permissions.UPDATE: 1, Permissions.OWN: 2}
    merged: dict[str, VariableSpec] = {}
    for spec in specs:
        known = merged.get(spec.name)
        if known is None:
            merged[spec.name] = spec
            continue
        if (known.shape, known.dtype) != (spec.shape, spec.dtype):
            raise ValueError(
                f"Variable '{spec.name}' is a {known.shape.name} {known.dtype} for "
                f"'{known.requested_by}' but a {spec.shape.name} {spec.dtype} for "
                f"'{spec.requested_by}'."
            )
        if strength[spec.permissions] > strength[known.permissions]:
            merged[spec.name] = spec
    return merged
