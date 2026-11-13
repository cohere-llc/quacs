# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Grid concepts for use by parameterizations."""

from __future__ import annotations

from typing import Protocol

import numpy as np

type Value = float | np.ndarray


class Cell(Protocol):
    """A single grid cell."""
    def get(self, name: str) -> Value:
        """Gets a named value for a grid cell."""
        ...

    def set(self, name: str, value: Value) -> None:
        """Sets the value of a name grid cell variable."""
        ...


class Column(Protocol):
    """A column of vertically stacked grid cells."""
    def get(self, name: str) -> Value:
        """Gets the value of a named variable in the currently selected grid cell in the column."""
        ...

    def set(self, name: str, value: Value) -> None:
        """Sets the value of a named variable in the currently selected grid cell in the column."""
        ...

    def step_toward_surface(self, count: int = 1) -> None:
        """Advance the selected grid cell toward the surface by `count` cells."""
        ...

    def step_toward_model_top(self, count: int = 1) -> None:
        """Advance the selected grid cell toward the model top by `count` cells."""
        ...

    def select_surface_cell(self) -> None:
        """Select the cell nearest the surface."""
        ...

    def select_model_top(self) -> None:
        """Select the cell nearest the model top."""
        ...

    def at_surface(self, buffer: int = 0) -> bool:
        """Returned whether the selected cell is within `buffer` cells of the cell nearest the surface."""
        ...

    def at_model_top(self, buffer: int = 0) -> bool:
        """Returns whether the selected cell is within `buffer` cells of the cell nearest the model top."""
        ...