# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Host application variable accessor passed to each Parameterization.run().

Provides access to NumPy views of host variables.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from quacs.host_adapter import HostAdapter


class QuacsState:
    """Applies parameterization functions over the host grid.

    Functions passed to `for_each_cell` and `for_each_column` must use only plain numeric
    code, capture only plain numbers from their enclosing scope, and keep no state
    between calls.
    """

    def __init__(self, host_adapter: HostAdapter, name_map: dict[str, str | None]) -> None:
        self._host_adapter = host_adapter
        self._name_map = name_map

    def for_each_cell(self, func: Callable[..., None], *names: str) -> None:
        """Applies `func` to every grid cell in the model domain.

        `func` is called with one `Variable` per name in `names`, in the same order.
        No guarantee is made as to the order grid cells are operated on.
        """
        self._host_adapter.for_each_cell(func, *self._host_names(names))

    def for_each_column(
        self,
        func: Callable[..., None],
        *names: str,
        start: Literal["surface", "model_top"] = "surface",
    ) -> None:
        """Applies `func` to every column in the model domain.

        `func` is called with a `Column`, followed by one `Variable` per name in `names`,
        in the same order. The column's selected grid cell starts at `start` and can be
        moved with `step_toward_surface` and `step_toward_model_top`.
        No guarantee is made of the order columns are operated on.
        """
        self._host_adapter.for_each_column(func, *self._host_names(names), start=start)

    def _host_names(self, names: tuple[str, ...]) -> list[str]:
        """Map each parameterization variable name to its host variable name."""
        return [self._name_map.get(name) or name for name in names]
