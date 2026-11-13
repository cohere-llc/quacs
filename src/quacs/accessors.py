# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Grid concepts for use by parameterizations."""

from __future__ import annotations

from typing import Protocol


class Variable(Protocol):
    """One variable in one grid cell.

    In a `for_each_cell` function, the grid cell is the one the function is being applied
    to. In a `for_each_column` function, it is the column's currently selected grid cell.
    A variable with one value per column has that value regardless of the selected cell.
    """

    def get(self) -> float:
        """Returns the value of the variable.

        The returned number is a copy. Changing it does not change the host variable.
        """
        ...

    def set(self, value: float) -> None:
        """Writes `value` directly to the host's storage for the variable."""
        ...


class Column(Protocol):
    """A column of vertically stacked grid cells, one of which is selected."""

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
        """Returns whether the selected cell is within `buffer` cells of the cell nearest the surface."""
        ...

    def at_model_top(self, buffer: int = 0) -> bool:
        """Returns whether the selected cell is within `buffer` cells of the cell nearest the model top."""
        ...
