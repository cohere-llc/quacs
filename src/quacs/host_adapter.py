# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Host Adapter Protocol.

A concrete implementation of `HostAdapter` should exist for each host application,
and be versioned with the host application.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, runtime_checkable

from quacs.accessors import Adjacent


@runtime_checkable
class HostAdapter(Protocol):
    """Protocol that a host application must implement to work with QUACS."""

    def canonical_name_map(self) -> dict[str, str | None]:
        """Returns a mapping of QUACS canonical variable name -> host variable name.

        For variables that are unavailable in the host application, set the value
        to `None`.
        """
        ...

    def for_each_cell(self, func: Callable[..., None], *variables: str | Adjacent) -> None:
        """Applies `func` to each grid cell in the model grid.

        `func` is called with one accessor per name in `variables`, in the same order.
        Variables with one value per interface must be wrapped with `upper()` or
        `lower()`. No guarantee is made of the order cells are operated on.

        If any named variable has one value per grid cell or per interface, only the
        variables with one value per grid cell can be set. For the others, `set` must
        raise `ReadOnlyVariableError`.
        """
        ...

    def for_each_column(
        self,
        func: Callable[..., None],
        *variables: str | Adjacent,
        start: Literal["surface", "model_top"] = "surface",
    ) -> None:
        """Applies `func` to each column in the model grid.

        `func` is called with a `Column`, followed by one accessor per entry in
        `variables`, in the same order. Variables with one entry per interface must be
        wrapped with `upper()` or `lower()`, and refer to the selected grid cell's
        interfaces. The column's selected grid cell starts at `start`.
        No guarantee is made of the order columns are operated on.
        """
        ...

    def for_each_interface(self, func: Callable[..., None], *variables: str | Adjacent) -> None:
        """Applies `func` to each interface in the model grid, including the surface and
        model top.

        `func` is called with one accessor per entry in `variables`, in the same order.
        Variables with one value per grid cell must be wrapped with `above()` or `below()`.
        Where there is no such grid cell, `func` is called with `None` in place of the
        accessor. No guarantee is made of the order interfaces are operated on.

        Only variables with one value per interface can be set. For the others, `set`
        must raise `ReadOnlyVariableError`.
        """
        ...
