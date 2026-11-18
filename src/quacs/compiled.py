# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Applies Numba-compiled parameterization functions over array-backed variables.

A function is compiled the first time it is run, together with the loop that applies
it to the model grid. Subsequent calls reuse the compiled code.

Inside compiled code, each accessor and the cursor are named tuples of plain integers:
memory addresses, strides, and offsets. Their methods are attached with Numba's extension
interface, and read and write the host application's memory directly.
"""

from __future__ import annotations

import functools
from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np
from numba import types
from numba.extending import intrinsic, overload_method

from quacs.accessors import Scalar
from quacs.exceptions import ReadOnlyVariableError
from quacs.operations import READ_ONLY_REASONS, Field, Operation

# Marks the place of a local scalar among the compiled accessors of a call.
_SCALAR = object()

# The cursor's state is three 64-bit integers: the selected column, the selected level,
# and the position in the call of a read-only variable that a function tried to set, or
# -1 if there was no such error. There are their positions in the cursor tuple, and
# `_WORD` is their size in bytes.
_COLUMN = 0
_LEVEL = 1
_VIOLATION = 2
_WORD = 8


@functools.cache
def _loader(value_type: Any) -> Any:
    """A compiled function that reads one value of `value_type` from a memory address."""

    @intrinsic
    def load(typingctx, address):
        def codegen(context, builder, signature, args):
            pointer_type = context.get_data_type(value_type).as_pointer()
            pointer = builder.inttoptr(args[0], pointer_type)
            return context.unpack_value(builder, value_type, pointer)

        return value_type(types.intp), codegen

    return load

@functools.cache
def _storer(value_type: Any) -> Any:
    """A compiled function that writes one value of `value_type` to a memory address."""

    @intrinsic
    def store(typingctx, address, value):
        def codegen(context, builder, signature, args):
            pointer_type = context.get_data_type(value_type).as_pointer()
            pointer = builder.inttoptr(args[0], pointer_type)
            context.pack_value(builder, value_type, args[1], pointer)
            return context.get_dummy_value()

        return types.void(types.intp, value_type), codegen

    return store


_load_int = _loader(types.int64)
_store_int = _storer(types.int64)


# The cursor: the selected grid cell or interface of a column. Level 0 is at the surface.
# `state` is the address of the cursor's state
_Cursor = namedtuple("_Cursor", ["state", "nlevels"])


@overload_method(types.BaseNamedTuple, "step_toward_surface")
def _overload_step_toward_surface(cursor, count=1):
    if cursor.instance_class is not _Cursor:
        return None

    def step_toward_surface(cursor, count=1):
        level = _load_int(cursor.state + _WORD * _LEVEL) - count
        if level < 0 or level >= cursor.nlevels:
            raise IndexError("Cannot step beyond the surface or model top.")
        _store_int(cursor.state + _WORD * _LEVEL, level)

    return step_toward_surface


@overload_method(types.BaseNamedTuple, "step_toward_model_top")
def _overload_step_toward_model_top(cursor, count=1):
    if cursor.instance_class is not _Cursor:
        return None

    def step_toward_model_top(cursor, count=1):
        level = _load_int(cursor.state + _WORD * _LEVEL) + count
        if level < 0 or level >= cursor.nlevels:
            raise IndexError("Cannot step beyond the surface or model top.")
        _store_int(cursor.state + _WORD * _LEVEL, level)

    return step_toward_model_top


@overload_method(types.BaseNamedTuple, "select_surface_cell")
def _overload_select_surface_cell(cursor):
    if cursor.instance_class is not _Cursor:
        return None

    def select_surface_cell(cursor):
        _store_int(cursor.state + _WORD * _LEVEL, 0)

    return select_surface_cell


@overload_method(types.BaseNamedTuple, "select_model_top")
def _overload_select_model_top(cursor):
    if cursor.instance_class is not _Cursor:
        return None

    def select_model_top(cursor):
        _store_int(cursor.state + _WORD * _LEVEL, cursor.nlevels - 1)

    return select_model_top


@overload_method(types.BaseNamedTuple, "at_surface")
def _overload_at_surface(cursor, buffer=0):
    if cursor.instance_class is not _Cursor:
        return None

    def at_surface(cursor, buffer=0):
        return _load_int(cursor.state + _WORD * _LEVEL) <= buffer

    return at_surface


@overload_method(types.BaseNamedTuple, "at_model_top")
def _overload_at_model_top(cursor, buffer=0):
    if cursor.instance_class is not _Cursor:
        return None

    def at_model_top(cursor, buffer=0):
        return cursor.nlevels - 1 - _load_int(cursor.state + _WORD * _LEVEL) <= buffer

    return at_model_top


# Accessors. `address` is where the variable's first value is, the strides are in bytes,
# `offset` is the variable's level relative to the cursor's, `state` is the address of
# the cursor's state, and `position` is the accessor's place in the call.
_ACCESSOR_FIELDS = ("address", "column_stride", "level_stride", "offset", "state", "position")

# The kind of variable behind each accessor class: value type, per column, read-only.
_KINDS: dict[type, tuple[Any, bool, bool]] = {}


@functools.cache
def _accessor_class(value_type: Any, per_column: bool, read_only: bool) -> Any:
    """A named tuple class for accessors to one kind of variable.

    There is a class for each kind so that `get` and `set` are chosen when a function is
    compiled, and have no branches when it runs.
    """
    cls = namedtuple("_Accessor", _ACCESSOR_FIELDS)
    _KINDS[cls] = (value_type, per_column, read_only)
    return cls


@overload_method(types.BaseNamedTuple, "get")
def _overload_get(accessor):
    kind = _KINDS.get(accessor.instance_class)
    if kind is None:
        return None
    value_type, per_column, _ = kind
    load = _loader(value_type)

    def get_column(accessor):
        column = _load_int(accessor.state + _WORD * _COLUMN)
        return load(accessor.address + column * accessor.column_stride)

    def get_level(accessor):
        column = _load_int(accessor.state + _WORD * _COLUMN)
        level = _load_int(accessor.state + _WORD * _LEVEL) + accessor.offset
        return load(
            accessor.address + column * accessor.column_stride + level * accessor.level_stride
        )

    return get_column if per_column else get_level


@overload_method(types.BaseNamedTuple, "set")
def _overload_set(accessor, value):
    kind = _KINDS.get(accessor.instance_class)
    if kind is None:
        return None
    value_type, per_column, read_only = kind
    store = _storer(value_type)

    def set_read_only(accessor, value):
        # Record any attempt to write to read-only memory
        _store_int(accessor.state + _WORD * _VIOLATION, accessor.position)

    def set_column(accessor, value):
        column = _load_int(accessor.state + _WORD * _COLUMN)
        store(accessor.address + column * accessor.column_stride, value)

    def set_level(accessor, value):
        column = _load_int(accessor.state + _WORD * _COLUMN)
        level = _load_int(accessor.state + _WORD * _LEVEL) + accessor.offset
        store(
            accessor.address + column * accessor.column_stride + level * accessor.level_stride,
            value,
        )

    if read_only:
        return set_read_only
    return set_column if per_column else set_level


def _accessor(field: Field, state: int, position: int) -> Any:
    per_column = field.level_offset is None
    cls = _accessor_class(numba.from_dtype(field.array.dtype), per_column, field.read_only)
    return cls(
        field.array.__array_interface__["data"][0],
        field.array.strides[0],
        0 if per_column else field.array.strides[1],
        field.level_offset or 0,
        state,
        position,
    )


def _check_shape(operation: Operation, field: Field, ncolumns: int, nlevels: int) -> None:
    """Check that a function cannot reach outside a variable's memory."""
    if field.array.ndim == 2:
        columns, levels = field.array.shape
        if field.level_offset is None:
            fits = levels == 1
        else:
            # The lowest and highest levels the operation selects where the variable
            # must exist. At the first and last interface, `_at` passes `None` instead.
            first = 1 if operation == "interface" else 0
            fits = first + field.level_offset >= 0 and nlevels - 1 + field.level_offset < levels
        if columns == ncolumns and fits:
            return
    raise ValueError(
        f"Variable '{field.name}' has shape {field.array.shape}, which does not fit a grid "
        f"of {ncolumns} columns and {nlevels} levels."
    )


def _memory(array: np.ndarray) -> tuple[Any, ...]:
    """Where an array's data are in memory and how they are organized."""
    return (array.__array_interface__["data"][0], array.shape, array.strides, array.dtype)


def _at(
    entries: tuple[Any, ...], arguments: list[Field | Scalar], level: int
) -> tuple[Any, ...]:
    """Variable entries for an interface.

    Adjacent grid cells that do not exist for this interface are set as `None`.
    """
    return tuple(
        None
        if isinstance(argument, Field)
        and argument.level_offset is not None
        and not 0 <= level + argument.level_offset < argument.array.shape[1]
        else entry
        for entry, argument in zip(entries, arguments)
    )


def _with_scalars(entries: tuple[Any, ...], arguments: list[Field | Scalar]) -> tuple[Any, ...]:
    """Put the local scalar values for this call in order among the accessors."""
    return tuple(
        argument if entry is _SCALAR else entry for entry, argument in zip(entries, arguments)
    )


@numba.njit
def _run_cells(kernel, state, ncolumns, nlevels, arguments):
    for column in range(ncolumns):
        _store_int(state + _WORD * _COLUMN, column)
        for level in range(nlevels):
            _store_int(state + _WORD * _LEVEL, level)
            kernel(*arguments)


@numba.njit
def _run_columns(kernel, cursor, ncolumns, start_level, arguments):
    for column in range(ncolumns):
        _store_int(cursor.state + _WORD * _COLUMN, column)
        _store_int(cursor.state + _WORD * _LEVEL, start_level)
        kernel(cursor, *arguments)


@numba.njit
def _run_interfaces(kernel, state, ncolumns, nlevels, at_surface, between, at_top):
    for column in range(ncolumns):
        _store_int(state + _WORD * _COLUMN, column)
        _store_int(state + _WORD * _LEVEL, 0)
        kernel(*at_surface)
        for interface in range(1, nlevels):
            _store_int(state + _WORD * _LEVEL, interface)
            kernel(*between)
        _store_int(state + _WORD * _LEVEL, nlevels)
        kernel(*at_top)


@dataclass(frozen=True, eq=False)
class _Plan:
    """The compiled objects for one function and the memory it is applied to.
    
    `state` is the cursor's state, which `cursor` and every accessor point to.
    `between`, `at_surface`, and `at_top` hold one entry per argument of the call. Only
    interface functions use `at_surface` and `at_top`.
    """

    kernel: Any
    state: np.ndarray
    cursor: Any
    between: tuple[Any, ...]
    at_surface: tuple[Any, ...]
    at_top: tuple[Any, ...]
    memory: tuple[Any, ...]


class CompiledEngine:
    """Compiles each function with Numba and applies it in a compiled loop."""

    def __init__(self, ncolumns: int, nlevels: int) -> None:
        self._ncolumns = ncolumns
        self._nlevels = nlevels
        self._kernels: dict[Any, Any] = {}
        self._plans: dict[Any, Any] = {}

    def run(
        self,
        operation: Operation,
        func: Callable[..., None],
        arguments: list[Field | Scalar],
        start_level: int = 0,
    ) -> None:
        plan = self._plan(operation, func, arguments)
        plan.state[_VIOLATION] = -1
        between = _with_scalars(plan.between, arguments)
        if operation == "cell":
            _run_cells(plan.kernel, plan.cursor.state, self._ncolumns, self._nlevels, between)
        elif operation == "column":
            _run_columns(plan.kernel, plan.cursor, self._ncolumns, start_level, between)
        elif operation == "interface":
            _run_interfaces(
                plan.kernel,
                plan.cursor.state,
                self._ncolumns,
                self._nlevels,
                _with_scalars(plan.at_surface, arguments),
                between,
                _with_scalars(plan.at_top, arguments),
            )
        else:
            msg = f"Operation '{operation}' not supported."
            raise NotImplementedError(msg)
        violation = int(plan.state[_VIOLATION])
        if violation >= 0:
            field = arguments[violation]
            assert isinstance(field, Field)
            raise ReadOnlyVariableError(field.name, READ_ONLY_REASONS[operation])


    def _plan(
        self, operation: Operation, func: Callable[..., None], arguments: list[Field | Scalar]
    ) -> _Plan:
        captured = tuple(
            (type(cell.cell_contents), cell.cell_contents) for cell in func.__closure__ or ()
        )
        kernel_key = (func.__code__, captured)
        layout = tuple(
            (argument.name, argument.level_offset, argument.read_only)
            if isinstance(argument, Field)
            else None
            for argument in arguments
        )
        memory = tuple(
            _memory(argument.array) if isinstance(argument, Field) else None
            for argument in arguments
        )
        key = (operation, kernel_key, layout)
        plan = self._plans.get(key)
        # The accessors are rebuilt if the host application has moved a variable.
        if plan is None or plan.memory != memory:
            for argument in arguments:
                if isinstance(argument, Field):
                    _check_shape(operation, argument, self._ncolumns, self._nlevels)
            if kernel_key not in self._kernels:
                self._kernels[kernel_key] = numba.njit(func)
            state = np.zeros(3, dtype=np.int64)
            cursor = _Cursor(state.ctypes.data, self._nlevels)
            entries = tuple(
                _accessor(argument, cursor.state, position)
                if isinstance(argument, Field)
                else _SCALAR
                for position, argument in enumerate(arguments)
            )
            plan = self._plans[key] = _Plan(
                kernel=self._kernels[kernel_key],
                state=state,
                cursor=cursor,
                between=entries,
                at_surface=_at(entries, arguments, 0),
                at_top=_at(entries, arguments, self._nlevels),
                memory=memory,
            )
        return plan