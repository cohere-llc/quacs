# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Host application variable accessor passed to each Parameterization.run().

Provides access to host variables.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Literal

from quacs.accessors import Adjacent, Argument
from quacs.host_adapter import HostAdapter


class QuacsState:
    """Applies parameterization functions over the host grid.

    Each `for_each_*` operation calls `func` with one argument per entry in `arguments`,
    in the same order:

    - A local scalar variable is passed to `func` unchanged.
    - The name of a variable with one value for the whole domain is passed as the
      variable's current value (read-only).
    - Any other variable name is passed as an accessor for that variable.

    Functions passed to these operations must use only plain numeric code and keep no
    state between calls. They may capture plain numbers from their enclosing scope, but
    only numbers that stay constant for the whole run. Pass a number that can change as
    an argument instead.
    """

    def __init__(self, host_adapter: HostAdapter, names: Mapping[str, str]) -> None:
        self._host_adapter = host_adapter
        self._names = names

    def for_each_cell(self, func: Callable[..., None], *arguments: Argument) -> None:
        """Applies `func` to every grid cell in the model domain.

        Wrap a variable with one value per interface with `upper()` or `lower()`.
        No guarantee is made as to the order grid cells are operated on.

        Only variables with one value per grid cell can be set. The others are read-only.
        Use `for_each_column` or `for_each_interface` to set them.
        """
        self._host_adapter.for_each_cell(func, *self._host_arguments(arguments))

    def for_each_column(
        self,
        func: Callable[..., None],
        *arguments: Argument,
        start: Literal["surface", "model_top"] = "surface",
    ) -> None:
        """Applies `func` to every column in the model domain.

        `func` is called with a `Column`, followed the arguments. Wrap a variable with
        one value per interface with `upper()` or `lower()`. The column's selected grid
        cell starts at `start` and can be moved with `step_toward_surface` or
        `step_toward_model_top`. No guarantee is made of the order columns are operated on.
        """
        self._host_adapter.for_each_column(func, *self._host_arguments(arguments), start=start)

    def for_each_interface(self, func: Callable[..., None], *arguments: Argument) -> None:
        """Applies `func` to every interface in the model domain, including the surface
        and the model top.

        Wrap a variable with one value per grid cell with `above()` or `below()`. Where
        there is no such grid cell, `func` recieves `None` in place of the accessor.
        No guarantee is made of the order interfaces are operated on.

        Only variables with one value per interface can be set. The others are read-only.
        Use `for_each_column` or `for_each_cell` to set them.
        """
        self._host_adapter.for_each_interface(func, *self._host_arguments(arguments))

    def _host_arguments(self, arguments: tuple[Argument, ...]) -> list[Argument]:
        """Replace each parameterization variable name to its host variable name."""
        return [self._host_argument(argument) for argument in arguments]

    def _host_argument(self, argument: Argument) -> Argument:
        if isinstance(argument, Adjacent):
            return dataclasses.replace(argument, name=self._host_name(argument.name))
        if isinstance(argument, str):
            return self._host_name(argument)
        return argument

    def _host_name(self, name: str) -> str:
        return self._names.get(name, name)
