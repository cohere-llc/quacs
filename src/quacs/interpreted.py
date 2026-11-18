# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Applies parameterization functions over array-backed variables in a Python loop."""

from __future__ import annotations

from collections.abc import Callable

from quacs.accessors import Scalar
from quacs.exceptions import ReadOnlyVariableError
from quacs.operations import READ_ONLY_REASONS, Field, Operation


def _matches(value: object, kind: str) -> bool:
    """Whether a Python value has the type stored by a NumPy array of the given kind."""
    if kind == "b":
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    if kind in "iu":
        return isinstance(value, int)
    return isinstance(value, (int, float))


class _Cursor:
    """The selected grid cell or interface of a column. Level 0 is at the surface."""

    def __init__(self, nlevels: int) -> None:
        self.column = 0
        self.level = 0
        self._nlevels = nlevels

    def step_toward_surface(self, count: int = 1) -> None:
        self._select(self.level - count)

    def step_toward_model_top(self, count: int = 1) -> None:
        self._select(self.level + count)

    def select_surface_cell(self) -> None:
        self.level = 0

    def select_model_top(self) -> None:
        self.level = self._nlevels - 1

    def at_surface(self, buffer: int = 0) -> bool:
        return self.level <= buffer

    def at_model_top(self, buffer: int = 0) -> bool:
        return self._nlevels - 1 - self.level <= buffer

    def _select(self, level: int) -> None:
        if level < 0 or level >= self._nlevels:
            raise IndexError("Cannot step beyond the surface or model top.")
        self.level = level


class _Accessor:
    """A variable at a location relative to the cursor."""

    def __init__(self, field: Field, cursor: _Cursor, read_only_reason: str | None) -> None:
        self._name = field.name
        self._array = field.array
        self._level_offset = field.level_offset
        self._cursor = cursor
        self._read_only_reason = read_only_reason

    def get(self) -> Scalar:
        value: Scalar = self._array[self._index()].item()
        return value

    def set(self, value: Scalar) -> None:
        if self._read_only_reason is not None:
            raise ReadOnlyVariableError(self._name, self._read_only_reason)
        if not _matches(value, self._array.dtype.kind):
            raise TypeError(
                f"Variable '{self._name}' holds {self._array.dtype} values and cannot be "
                f"set to {value!r}."
            )
        self._array[self._index()] = value

    def exists(self) -> bool:
        """Whether there is a grid cell or interface at the variable's location."""
        if self._level_offset is None:
            return True
        return 0 <= self._cursor.level + self._level_offset < self._array.shape[1]

    def _index(self) -> tuple[int, int]:
        if self._level_offset is None:
            return (self._cursor.column, 0)
        return (self._cursor.column, self._cursor.level + self._level_offset)


class InterpretedEngine:
    """Calls a function once per grid cell, column, or interface."""

    def __init__(self, ncolumns: int, nlevels: int) -> None:
        self._ncolumns = ncolumns
        self._nlevels = nlevels

    def run(
        self,
        operation: Operation,
        func: Callable[..., None],
        arguments: list[Field | Scalar],
        start_level: int = 0,
    ) -> None:
        cursor = _Cursor(self._nlevels)
        values: list[_Accessor | Scalar] = [
            _Accessor(
                argument, cursor, READ_ONLY_REASONS[operation] if argument.read_only else None
            )
            if isinstance(argument, Field)
            else argument
            for argument in arguments
        ]
        for column in range(self._ncolumns):
            cursor.column = column
            if operation == "column":
                cursor.level = start_level
                func(cursor, *values)
            elif operation == "cell":
                for level in range(self._nlevels):
                    cursor.level = level
                    func(*values)
            elif operation == "interface":
                for interface in range(self._nlevels + 1):
                    cursor.level = interface
                    func(
                        *(
                            None if isinstance(value, _Accessor) and not value.exists() else value
                            for value in values
                        )
                    )
            else:
                msg = f"Unsupported operation type {operation}"
                raise NotImplementedError(msg)
