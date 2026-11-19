# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""A reusable host adapter for host applications that can present variables as NumPy views.

Hosts that are unable to use this adapter can implement the `HostAdapter` protocol directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Literal

import numpy as np

from quacs.accessors import Argument, Scalar
from quacs.compiled import CompiledEngine
from quacs.exceptions import OverwriteVariableError, VariableNotFoundError
from quacs.interpreted import InterpretedEngine
from quacs.operations import Field, Operation, Slot, resolve
from quacs.shapes import GridShape
from quacs.variables import Permissions, VariableSpec

# The kinds of NumPy data types that can store each QUACS data type
_KINDS: dict[str, str] = {"float": "f", "int": "iu", "bool": "b"}


class ArrayHostAdapter(ABC):
    """A host adapter for host application that can present each variable as a NumPy view.

    A host application subclasses this and provides `canonical_name_map`, `grid_shapes`,
    `view`, and `create_variable`. This class provides the `for_each_*` operations, the
    registration of variables, and what the QUACS profiler needs.

    `view` must return each variable in standard layout, without copying it:

    - one value per grid cell: shape `(columns, levels)`, with the surface first;
    - one value per interface: shape `(columns, levels + 1)`, with the surface first;
    - one value per column: shape `(columns,)`;
    - one value for the whole domain: one element.

    With `compiled=True`, parameterization functions are compiled with Numba. Otherwise
    they are called from a Python loop.
    """

    def __init__(self, ncolumns: int, nlevels: int, compiled: bool = True) -> None:
        self._ncolumns = ncolumns
        self._nlevels = nlevels
        self._engine: InterpretedEngine | CompiledEngine = (
            CompiledEngine(ncolumns, nlevels) if compiled else InterpretedEngine(ncolumns, nlevels)
        )

    @property
    def ncolumns(self) -> int:
        return self._ncolumns

    @property
    def nlevels(self) -> int:
        return self._nlevels

    # Host Application provided functions

    @abstractmethod
    def canonical_name_map(self) -> dict[str, str | None]:
        """Returns a mapping of QUACS canonical variable name -> host variable name."""

    @abstractmethod
    def grid_shapes(self) -> Mapping[str, GridShape]:
        """Returns the shape of every variable in the host application state, by host name."""

    @abstractmethod
    def view(self, host_name: str) -> np.ndarray:
        """Returns a view of a variable in the standard layout, without copying it."""

    @abstractmethod
    def create_variable(self, spec: VariableSpec) -> None:
        """Creates a variable the host application does not have, set to zero or `False`.

        `spec.name` is the host name. Afterwards the variable must be in `grid_shapes`
        and available from `view`.
        """

    # General-use functions

    def view_shape(self, shape: GridShape) -> tuple[int, ...]:
        """Returns the array shape of the standard layout for a variable shape."""
        if shape is GridShape.CELL:
            return (self._ncolumns, self._nlevels)
        if shape is GridShape.INTERFACE:
            return (self._ncolumns, self._nlevels + 1)
        if shape is GridShape.DOMAIN:
            return ()
        return (self._ncolumns,)

    # The grid-iterating operations

    def for_each_cell(self, func: Callable[..., None], *arguments: Argument) -> None:
        self._engine.run("cell", func, self._fields("cell", arguments))

    def for_each_column(
        self,
        func: Callable[..., None],
        *arguments: Argument,
        start: Literal["surface", "model_top"] = "surface",
    ) -> None:
        start_level = 0 if start == "surface" else self._nlevels - 1
        self._engine.run("column", func, self._fields("column", arguments), start_level)

    def for_each_interface(self, func: Callable[..., None], *arguments: Argument) -> None:
        self._engine.run("interface", func, self._fields("interface", arguments))

    # Variable registration

    def allocate_registered_variables(self, specs: list[VariableSpec]) -> None:
        """Creates the variables that parameterizations own and checks that the rest exist.

        `specs` is the list returned by `Runtime.register_variables`.
        """
        for spec in specs:
            if spec.permissions is Permissions.OWN:
                if spec.name in self.grid_shapes():
                    raise OverwriteVariableError(
                        spec.name, ["host application", spec.requested_by or "unknown"]
                    )
                self.create_variable(spec)
        for spec in specs:
            self._check(spec)

    # Functions used only by the profiler

    def create_variables(self, specs: list[VariableSpec]) -> None:
        """Creates every variable in `specs` that the host application does not have."""
        for spec in specs:
            if spec.name not in self.grid_shapes():
                self.create_variable(spec)
            self._check(spec)

    def grid_cell_count(self) -> int:
        return self._ncolumns * self._nlevels

    def domain_mean(self, host_name: str) -> float:
        return float(self.view(host_name).mean())

    def fill(self, host_name: str, value: Scalar) -> None:
        self.view(host_name)[...] = value

    def _fields(
        self, operation: Operation, arguments: tuple[Argument, ...]
    ) -> list[Field | Scalar]:
        """Attach storage to each variable named in a  `for_each_*` call."""
        fields: list[Field | Scalar] = []
        for argument in resolve(operation, arguments, self.grid_shapes()):
            if not isinstance(argument, Slot):
                fields.append(argument)
            elif argument.shape is GridShape.DOMAIN:
                fields.append(self.view(argument.name).item())
            else:
                array = self.view(argument.name)
                if argument.level_offset is None:
                    array = array[:, np.newaxis]
                fields.append(
                    Field(argument.name, array, argument.level_offset, argument.read_only)
                )
        return fields

    def _check(self, spec: VariableSpec) -> None:
        """Check that the host application has a variable that matches `spec`."""
        shapes = self.grid_shapes()
        if spec.name not in shapes:
            raise VariableNotFoundError(spec.name, spec.requested_by or "unknown")
        if shapes[spec.name] is not spec.shape:
            raise ValueError(
                f"Variable '{spec.name}' is {shapes[spec.name].name} in the host application "
                f"but was requested as {spec.shape.name}"
            )
        array = self.view(spec.name)
        if array.dtype.kind not in _KINDS[spec.dtype]:
            raise ValueError(
                f"Variable '{spec.name}' is stored as {array.dtype} in the host application "
                f"but was requested as {spec.dtype}."
            )
        if spec.shape is GridShape.DOMAIN:
            if array.size != 1:
                raise ValueError(
                    f"The view of variable '{spec.name}' has {array.size} elements, but a "
                    f"DOMAIN variable should have one."
                )
        elif array.shape != self.view_shape(spec.shape):
            raise ValueError(
                f"The view of variable '{spec.name}' has shape {array.shape}, but a "
                f"{spec.shape.name} variable needs shape {self.view_shape(spec.shape)}"
            )
