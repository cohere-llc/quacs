# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Host-agnostic variable shapes.

Parameterizations use `GridShape` to describe the size and shape of variables being requested
or registered with a host application. The host application is responsible for setting the
approriate dimensions for each `GridShape` according to the host application internal settings
and configuration.
"""

from __future__ import annotations

from enum import Enum, auto


class GridShape(Enum):
    """Supported variable shapes."""

    CELL = auto()  # one value per grid cell at every level, in every column
    INTERFACE = auto()  # one value for every horizontal cell boundary (including the surface and model top) in every column
    SURFACE_CELL = auto()  # one value per column
    MODEL_TOP_CELL = auto()  # one value per column

    @property
    def is_level_resolved(self) -> bool:
        return self in (GridShape.CELL, GridShape.INTERFACE)
