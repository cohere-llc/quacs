# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Host Adapter Protocol.

A concrete implementation of `HostAdapter` should exist for each host application,
and be versioned with the host application.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, runtime_checkable


@runtime_checkable
class HostAdapter(Protocol):
    """Protocol that a host application must implement to work with QUACS."""

    def canonical_name_map(self) -> dict[str, str | None]:
        """Returns a mapping of QUACS canonical variable name -> host variable name.

        For variables that are unavailable in the host application, set the value
        to `None`.
        """
        ...

    def for_each_cell(self, func: Callable[..., None], *host_names: str) -> None:
        """Applies `func` to each grid cell in the model grid.

        `func` is called with one `Variable` per name in `host_names`, in the same order.
        No guarantee is made of the order cells are operated on.
        """
        ...

    def for_each_column(
        self,
        func: Callable[..., None],
        *host_names: str,
        start: Literal["surface", "model_top"] = "surface",
    ) -> None:
        """Applies `func` to each column in the model grid.

        `func` is called with a `Column`, followed by one `Variable` per name in
        `host_names`, in the same order. The column's selected grid cell starts at `start`.
        No guarantee is made of the order columns are operated on.
        """
        ...
