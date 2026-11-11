# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Host Adapter Protocol.

A concrete implementation of `HostAdapter` should exist for each host application,
and be versioned with the host application.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from quacs.shapes import GridShape


@runtime_checkable
class HostAdapter(Protocol):
    """Protocol that a host application must implement to work with QUACS."""

    def canonical_name_map(self) -> dict[str, str | None]:
        """Return a mapping of QUACS canonical variable name -> host variable name.

        For variables that are unavailable in the host application, set the value
        to `None`.
        """
        ...

    def shape_sizes(self) -> dict[GridShape, tuple[int, ...]]:
        """Return the array shape and dimension sizes that correspond to each
        `GridShape`, for the host applications current configuration."""
        ...

    def get_array(self, host_name: str) -> np.ndarray:
        """Return a zero-copy NumPy view of the named host variable."""
        ...
