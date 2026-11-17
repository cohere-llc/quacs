# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Grid concepts for use by parameterizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


class Variable[T = float](Protocol):
    """A variable that describes a property at the midpoint of a grid cell, of the
    grid cell as a whole, or of an entire column.

    In a `for_each_cell` function, the grid cell is the one the function is being applied
    to. In a `for_each_column` function, it is the column's currently selected grid cell.
    A variable with one value per column has that value regardless of the selected cell.

    In a `for_each_interface` function, the grid cell is the one above or below the
    interface, chosen by wrapping the variable with `above()` or `below()` in the call.
    There is no grid cell above the upper-most interface or below the lower-most one.
    There, the function receives `None` in place of the `Variable`.

    `T` is the Python type of the variable's values: `float`, `int`, or `bool`, matching
    the `dtype` the variable was registered with.
    """

    def get(self) -> T:
        """Returns the value of the variable.

        The returned number is a copy. Changing it does not change the host variable.
        """
        ...

    def set(self, value: T) -> None:
        """Writes `value` directly to the host's storage for the variable.

        `value` must have the variable's type, except that an `int` can be given for a
        `float` variable. The host stores it with its own precision.

        Raises `ReadOnlyVariableError` if the function is not allowed to set the variable.
        """
        ...


class InterfaceVariable[T = float](Protocol):
    """A variable that describes a property of a horizontal interface: between two grid
    cells, between the lower-most grid cell and the surface, or between the upper-most
    grid cell and space above the model top.

    In a `for_each_interface` function, the interface is the one the function is being
    applied to. In a `for_each_cell` or `for_each_column` function, it is the interface
    at the top or bottom of the grid cell, chosen by wrapping the variable with `upper()`
    or `lower()` in the call.

    `T` is the Python type of the variable's values: `float`, `int`, or `bool`, matching
    the `dtype` the variable was registered with.
    """

    def get(self) -> T:
        """Returns the value of the interface variable.

        The returned number is a copy. Changing it does not change the host variable.
        """
        ...

    def set(self, value: T) -> None:
        """Writes the `value` directly to the host's storage for the variable.

        `value` must have the variable's type, except that an `int` can be given for a
        `float` variable. The host stores in with its own precision.

        Raises `ReadOnlyVariableError` if the function is not allowed to set the variable.
        """
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


@dataclass(frozen=True)
class Adjacent:
    """Names a variable at a location next to the one a function is applied to.

    Create these with `above`, `below`, `upper`, and `lower`.
    """

    name: str
    side: Literal["above", "below", "upper", "lower"]


def above(name: str) -> Adjacent:
    """In a `for_each_interface` call, names a variable in the grid cell above the interface."""
    return Adjacent(name, "above")


def below(name: str) -> Adjacent:
    """In a `for_each_interface` call, names a variable in the grid cell below the interface."""
    return Adjacent(name, "below")


def upper(name: str) -> Adjacent:
    """In a `for_each_cell` or `for_each_column` call, names a variable at the upper
    interface of the grid cell."""
    return Adjacent(name, "upper")


def lower(name: str) -> Adjacent:
    """In a `for_each_cell` or `for_each_column` call, names a variable at the lower
    interface of a grid cell."""
    return Adjacent(name, "lower")
