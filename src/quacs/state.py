# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Host application variable accessor passed to each Parameterization.run().

Provides access to NumPy views of host variables.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from quacs.accessors import Cell, Column, Value
from quacs.exceptions import UnknownVariableError
from quacs.host_adapter import HostAdapter


class QuacsState:
    """Resolves variable names to array views during a model run."""

    def __init__(self, host_adapter: HostAdapter, name_map: dict[str, str | None]) -> None:
        self._host_adapter = host_adapter
        self._name_map = name_map

    def for_each_cell(self, func: Callable[[Cell], None], *names: str) -> None:
        """Applies `func` to every grid cell in the model domain.
        
        No guarantee is made as to the order grid cells are operated on.
        """
        host_names = self._host_names(names)
        self._host_adapter.for_each_cell(
            lambda host_cell: func(_NamedCell(host_cell, host_names)),
            *host_names.values(),
        )

    def for_each_column(
        self,
        func: Callable[[Column], None],
        *names: str,
        start: Literal["surface", "model_top"] = "surface",
    ) -> None:
        """Applies `func` to every column in the model domain.
        
        The column's selected vertical cell start with `start` and can be advanced with
        the `step_toward_surface` and `step_toward_model_top` itertor functions.
        No guarantee is made of the order columns are operated on.
        """
        host_names = self._host_names(names)
        self._host_adapter.for_each_column(
            lambda host_column: func(_NamedColumn(host_column, host_names)),
            *host_names.values(),
            start=start,
        )

    def _host_names(self, names: tuple[str, ...]) -> dict[str, str]:
        """Map each parameterization variable name to its host variable name."""
        return {name: self._name_map.get(name) or name for name in names}
    

def _host_name(host_names: dict[str, str], name: str) -> str:
    try:
        return host_names[name]
    except KeyError:
        raise UnknownVariableError(name, list(host_names)) from None

class _NamedCell:
    """Presents a host `Cell` using QUACS variable names."""

    __slots__ = ("_cell", "_host_names")

    def __init__(self, cell: Cell, host_names: dict[str, str]) -> None:
        self._cell = cell
        self._host_names = host_names

    def get(self, name: str) -> Value:
        return self._cell.get(_host_name(self._host_names, name))

    def set(self, name: str, value: Value) -> None:
        self._cell.set(_host_name(self._host_names, name), value)


class _NamedColumn:
    """Presents a host `Column` using QUACS variable names."""

    __slots__ = ("_column", "_host_names")

    def __init__(self, column: Column, host_names: dict[str, str]) -> None:
        self._column = column
        self._host_names = host_names

    def get(self, name: str) -> Value:
        return self._column.get(_host_name(self._host_names, name))

    def set(self, name: str, value: Value) -> None:
        self._column.set(_host_name(self._host_names, name), value)

    def step_toward_surface(self, count: int = 1) -> None:
        self._column.step_toward_surface(count)

    def step_toward_model_top(self, count: int = 1) -> None:
        self._column.step_toward_model_top(count)

    def select_surface_cell(self) -> None:
        self._column.select_surface_cell()

    def select_model_top(self) -> None:
        self._column.select_model_top()

    def at_surface(self, buffer: int = 0) -> bool:
        return self._column.at_surface(buffer)

    def at_model_top(self, buffer: int = 0) -> bool:
        return self._column.at_model_top(buffer)
