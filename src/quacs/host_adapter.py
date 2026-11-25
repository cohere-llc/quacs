# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Host Adapter Protocol.

A concrete implementation of `HostAdapter` should exist for each host application,
and be versioned with the host application.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, runtime_checkable

from quacs.accessors import Argument


@runtime_checkable
class HostAdapter(Protocol):
    """Protocol that a host application must implement to work with QUACS.

    Each `for_each_*` operation calls `func` with one argument per entry in `arguments`,
    in the same order.

    - A local scalar variable is passed to `func` unchanged.
    - The name of a variable with one value for the whole domain is passed as the
      variable's current value (read-only).
    - Any other variable is passed as an accessor for that variable.
    - Variables are named by their host application names.
    """

    def for_each_cell(self, func: Callable[..., None], *variables: Argument) -> None:
        """Applies `func` to each grid cell in the model grid.

        Variables with one value per interface must be wrapped with `upper()` or
        `lower()`. No guarantee is made of the order cells are operated on.

        Only variables with one value per grid cell can be set. For the others, `set`
        must raise `ReadOnlyVariableError`.
        """
        ...

    def for_each_column(
        self,
        func: Callable[..., None],
        *variables: Argument,
        start: Literal["surface", "model_top"] = "surface",
    ) -> None:
        """Applies `func` to each column in the model grid.

        `func` is called with a `Column`, followed by the arguments. Variables with one
        entry per interface must be wrapped with `upper()` or `lower()`, and refer to the
        selected grid cell's interfaces. The column's selected grid cell starts at `start`.
        No guarantee is made of the order columns are operated on.
        """
        ...

    def for_each_interface(self, func: Callable[..., None], *variables: Argument) -> None:
        """Applies `func` to each interface in the model grid, including the surface and
        model top.

        Variables with one value per grid cell must be wrapped with `above()` or `below()`.
        Where there is no such grid cell, `func` is called with `None` in place of the
        accessor. No guarantee is made of the order interfaces are operated on.

        Only variables with one value per interface can be set. For the others, `set`
        must raise `ReadOnlyVariableError`.
        """
        ...
