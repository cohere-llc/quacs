# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Rules for `for_each_*` operations.

`resolve` determines what each entry in a `for_each_*` call refers to, The rules depend
only on the shape of each variable, so any host adapter can apply them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from quacs.accessors import Adjacent, Argument, Scalar
from quacs.exceptions import UnknownVariableError
from quacs.shapes import GridShape

type Operation = Literal["cell", "column", "interface"]

# Allowed directional decorators for each GridShape
_SELECTORS: dict[tuple[str, GridShape], tuple[str | None, ...]] = {
    ("cell", GridShape.CELL): (None,),
    ("cell", GridShape.INTERFACE): ("upper", "lower"),
    ("column", GridShape.CELL): (None,),
    ("column", GridShape.INTERFACE): ("upper", "lower"),
    ("interface", GridShape.CELL): ("above", "below"),
    ("interface", GridShape.INTERFACE): (None,),
}

# Array index of a variable relative to the selected grid cell or interface.
_LEVEL_OFFSETS: dict[str | None, int] = {None: 0, "upper": 1, "lower": 0, "above": 0, "below": -1}

# Shapes that are settable for `for_each_cell` and `for_each_interface` functions.
# `for_each_column` can set both cell and interface variables.
_SETTABLE: dict[str, GridShape] = {"cell": GridShape.CELL, "interface": GridShape.INTERFACE}

READ_ONLY_REASONS: dict[str, str] = {
    "cell": (
        "this function is applied to every grid cell, where only variables with one value "
        "per grid cell can be set. Use for_each_column to set this variable."
    ),
    "interface": (
        "this function is applied to every interface, where only variables with one value "
        "per interface can be set. Use for_each_column to set this variable."
    ),
}


@dataclass(frozen=True)
class Slot:
    """Describes what a variable name passed to a `for_each_*` operation refers to.

    `level_offset` is the variable's array index relative to the selected grid cell
    or interface. Arrays are indexed starting at the surface. Interface `k` is the lower
    interface of grid cell `k`. `level_offset` is None for variables without levels.

    `read_only` is `True` if the function is not allowed to set the variable.
    """

    name: str
    shape: GridShape
    level_offset: int | None
    read_only: bool


@dataclass(frozen=True, eq=False)
class Field:
    """Describes a variable that will be used in a `for_each_*` run.

    `array` has one row per column. For a variable with levels, each row holds the
    levels index starting at the surface and increasing with altitude. For any other
    variable, each row holds one value. `level_offset` and `read_only` are as for `Slot`.
    """

    name: str
    array: np.ndarray
    level_offset: int | None
    read_only: bool


def resolve(
    operation: Operation,
    arguments: tuple[Argument, ...],
    shapes: Mapping[str, GridShape],
) -> list[Slot | Scalar]:
    """Determines what each entry in a `for_each_*` call refers to."

    `shapes` gives the shape of every variable in the host state, by host name.
    Primitive types are returned unchanged; each variable is returned as a `Slot`.

    Raises `UnknownVariableError` for a name that is not in `shapes`, and `ValueError`
    for a variable that is named with the wrong directional selector for the operation.
    """
    settable = _SETTABLE.get(operation)
    resolved: list[Slot | Scalar] = []
    for argument in arguments:
        if isinstance(argument, Adjacent):
            name, side = argument.name, argument.side
        elif isinstance(argument, str):
            name, side = argument, None
        elif isinstance(argument, (float, int, bool)):
            resolved.append(argument)
            continue
        else:
            raise TypeError(
                f"for_each_{operation} arguments must be variable names, floats, ints, "
                f"or bools, not {type(argument).__name__}."
            )
        if name not in shapes:
            raise UnknownVariableError(name, list(shapes))
        shape = shapes[name]
        allowed = _SELECTORS.get((operation, shape), (None,))
        if side not in allowed:
            expected = " or ".join(f"{s}()" for s in allowed if s) or "no selector"
            raise ValueError(
                f"In for_each_{operation}, the {shape.name} variable '{name}' must be "
                f"passed with {expected}."
            )
        resolved.append(
            Slot(
                name,
                shape,
                _LEVEL_OFFSETS[side] if shape.is_level_resolved else None,
                read_only=shape is GridShape.DOMAIN
                or (settable is not None and shape is not settable),
            )
        )
    return resolved
