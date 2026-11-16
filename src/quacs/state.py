# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Host application variable accessor passed to each Parameterization.run().

Provides access to host variables.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Literal

from quacs.accessors import Adjacent
from quacs.host_adapter import HostAdapter


class QuacsState:
    """Applies parameterization functions over the host grid.

    Functions passed to `for_each_cell`, `for_each_column`, and `for_each_interface` must
    use only plain numeric code, capture only plain numbers from their enclosing scope,
    and keep no state between calls.
    """

    def __init__(self, host_adapter: HostAdapter, name_map: dict[str, str | None]) -> None:
        self._host_adapter = host_adapter
        self._name_map = name_map

    def for_each_cell(self, func: Callable[..., None], *variables: str | Adjacent) -> None:
        """Applies `func` to every grid cell in the model domain.

        `func` is called with one accessor per entry in `variables`, in the same order.
        Wrap a variable with one value per interface with `upper()` or `lower()`.
        No guarantee is made as to the order grid cells are operated on.

        Only variables with one value per grid cell can be set. The others are read-only.
        Use `for_each_column` or `for_each_interface` to set them.
        """
        self._host_adapter.for_each_cell(func, *self._host_variables(variables))

    def for_each_column(
        self,
        func: Callable[..., None],
        *variables: str | Adjacent,
        start: Literal["surface", "model_top"] = "surface",
    ) -> None:
        """Applies `func` to every column in the model domain.

        `func` is called with a `Column`, followed by one accessor per entry in
        `variables`, in the same order. Wrap a variable with one value per interface with
        `upper()` or `lower()`. The column's selected grid cell starts at start and can
        be moved with `step_toward_surface` or `step_toward_model_top`.
        No guarantee is made of the order columns are operated on.
        """
        self._host_adapter.for_each_column(func, *self._host_variables(variables), start=start)

    def for_each_interface(self, func: Callable[..., None], *variables: str | Adjacent) -> None:
        """Applies `func` to every interface in the model domain, including the surface
        and the model top.

        `func` is called with one accessor per entry in `variables`, in the same order.
        Wrap a variable with one value per grid cell with `above()` or `below()`. Where
        there is no such grid cell, `func` recieves `None` in place of the accessor.
        No guarantee is made of the order interfaces are operated on.

        Only variables with one value per interface can be set. The others are read-only.
        Use `for_each_column` or `for_each_cell` to set them.
        """
        self._host_adapter.for_each_interface(func, *self._host_variables(variables))

    def _host_variables(self, variables: tuple[str | Adjacent, ...]) -> list[str | Adjacent]:
        """Replace each parameterization variable name to its host variable name."""
        return [
            dataclasses.replace(variable, name=self._host_name(variable.name))
            if isinstance(variable, Adjacent)
            else self._host_name(variable)
            for variable in variables
        ]

    def _host_name(self, name: str) -> str:
        return self._name_map.get(name) or name
