"""Spike: can per-cell and per-column functions be compiled with Numba?"""

from __future__ import annotations

import math
import time

import numpy as np
from numba import float64, int64, njit, typed, types
from numba.experimental import jitclass

NCOLUMNS = 20_000
NLEVELS = 50

RATE = 0.5
CHEMISTRY = (0.5, 0.02, 0.05)  # attenuation, production rate, loss fraction

_ARRAY = float64[:, ::1]


# Shape A: functions take and return plain numbers.


def make_heat_a(rate):
    def heat(temperature):
        return temperature + rate

    return heat


def make_react_a(attenuation, production_rate, loss_fraction):
    def react(actinic_flux, ozone, temperature):
        actinic_flux = actinic_flux * math.exp(-attenuation * ozone)
        production = production_rate * actinic_flux * temperature / 290.0
        return actinic_flux, (ozone + production) * (1.0 - loss_fraction)

    return react


@njit
def for_each_cell_a(kernel, temperature):
    for column in range(temperature.shape[0]):
        for level in range(temperature.shape[1]):
            temperature[column, level] = kernel(temperature[column, level])


@njit
def sweep_down_a(kernel, initial, ozone, temperature):
    for column in range(ozone.shape[0]):
        carried = initial
        for level in range(ozone.shape[1]):
            carried, new_ozone = kernel(carried, ozone[column, level], temperature[column, level])
            ozone[column, level] = new_ozone


# Shape B: functions use get/set on a handle, as in the current QUACS API.


def make_heat_b(rate):
    def heat(cell):
        cell.set("temperature", cell.get("temperature") + rate)

    return heat


def make_react_b(attenuation, production_rate, loss_fraction):
    def react(column):
        actinic_flux = 1.0
        while True:
            ozone = column.get("ozone")
            actinic_flux = actinic_flux * math.exp(-attenuation * ozone)
            production = production_rate * actinic_flux * column.get("temperature") / 290.0
            column.set("ozone", (ozone + production) * (1.0 - loss_fraction))
            if column.at_surface():
                break
            column.step_toward_surface()

    return react


@jitclass(
    [
        ("arrays", types.ListType(_ARRAY)),
        ("slots", types.DictType(types.unicode_type, int64)),
        ("column", int64),
        ("level", int64),
        ("nlevels", int64),
    ]
)
class Handle:
    """Compiled stand-in for both `Cell` and `Column`. Level 0 is the model top."""

    def __init__(self, arrays, slots, nlevels):
        self.arrays = arrays
        self.slots = slots
        self.column = 0
        self.level = 0
        self.nlevels = nlevels

    def get(self, name):
        return self.arrays[self.slots[name]][self.column, self.level]

    def set(self, name, value):
        self.arrays[self.slots[name]][self.column, self.level] = value

    def step_toward_surface(self):
        self.level += 1

    def at_surface(self):
        return self.level == self.nlevels - 1


class PyHandle:
    """Interpreted stand-in for `Handle`."""

    def __init__(self, arrays, nlevels):
        self.arrays = arrays
        self.column = 0
        self.level = 0
        self.nlevels = nlevels

    def get(self, name):
        return self.arrays[name][self.column, self.level]

    def set(self, name, value):
        self.arrays[name][self.column, self.level] = value

    def step_toward_surface(self):
        self.level += 1

    def at_surface(self):
        return self.level == self.nlevels - 1


@njit
def for_each_cell_b(kernel, handle, ncolumns, nlevels):
    for column in range(ncolumns):
        for level in range(nlevels):
            handle.column = column
            handle.level = level
            kernel(handle)


@njit
def for_each_column_b(kernel, handle, ncolumns):
    for column in range(ncolumns):
        handle.column = column
        handle.level = 0
        kernel(handle)


# Shape C: a cursor plus one accessor per variable, with names resolved before the loop.


def make_heat_c(rate):
    def heat(temperature):
        temperature.set(temperature.get() + rate)

    return heat


def make_react_c(attenuation, production_rate, loss_fraction):
    def react(column, ozone, temperature):
        actinic_flux = 1.0
        while True:
            current = ozone.get()
            actinic_flux = actinic_flux * math.exp(-attenuation * current)
            production = production_rate * actinic_flux * temperature.get() / 290.0
            ozone.set((current + production) * (1.0 - loss_fraction))
            if column.at_surface():
                break
            column.step_toward_surface()

    return react


@jitclass([("column", int64), ("level", int64), ("nlevels", int64)])
class Cursor:
    """The selected grid cell. Level 0 is the model top."""

    def __init__(self, nlevels):
        self.column = 0
        self.level = 0
        self.nlevels = nlevels

    def step_toward_surface(self):
        self.level += 1

    def at_surface(self):
        return self.level == self.nlevels - 1


@jitclass([("array", _ARRAY), ("cursor", Cursor.class_type.instance_type)])
class Variable:
    """One variable at the cursor's selected grid cell."""

    def __init__(self, array, cursor):
        self.array = array
        self.cursor = cursor

    def get(self):
        return self.array[self.cursor.column, self.cursor.level]

    def set(self, value):
        self.array[self.cursor.column, self.cursor.level] = value


@njit
def for_each_cell_c(kernel, cursor, temperature, ncolumns, nlevels):
    for column in range(ncolumns):
        for level in range(nlevels):
            cursor.column = column
            cursor.level = level
            kernel(temperature)


@njit
def for_each_column_c(kernel, cursor, ozone, temperature, ncolumns):
    for column in range(ncolumns):
        cursor.column = column
        cursor.level = 0
        kernel(cursor, ozone, temperature)


def new_state(ncolumns):
    return {
        "temperature": np.full((ncolumns, NLEVELS), 290.0),
        "ozone": np.zeros((ncolumns, NLEVELS)),
    }


def compiled_handle(state):
    arrays = typed.List.empty_list(_ARRAY)
    slots = typed.Dict.empty(types.unicode_type, int64)
    for name, array in state.items():
        slots[name] = len(arrays)
        arrays.append(array)
    return Handle(arrays, slots, NLEVELS)


def timed(label, func, *args):
    start = time.perf_counter()
    func(*args)
    print(f"{label:<24}{time.perf_counter() - start:>10.4f} s")


def main():
    heat_a, react_a = njit(make_heat_a(RATE)), njit(make_react_a(*CHEMISTRY))
    heat_b, react_b = njit(make_heat_b(RATE)), njit(make_react_b(*CHEMISTRY))
    heat_c, react_c = njit(make_heat_c(RATE)), njit(make_react_c(*CHEMISTRY))

    # Compile everything on a small state so the timings below exclude compilation.
    warmup = new_state(2)
    for_each_cell_a(heat_a, warmup["temperature"])
    sweep_down_a(react_a, 1.0, warmup["ozone"], warmup["temperature"])
    warmup_handle = compiled_handle(warmup)
    for_each_cell_b(heat_b, warmup_handle, 2, NLEVELS)
    for_each_column_b(react_b, warmup_handle, 2)
    warmup_cursor = Cursor(NLEVELS)
    warmup_ozone = Variable(warmup["ozone"], warmup_cursor)
    warmup_temperature = Variable(warmup["temperature"], warmup_cursor)
    for_each_cell_c(heat_c, warmup_cursor, warmup_temperature, 2, NLEVELS)
    for_each_column_c(react_c, warmup_cursor, warmup_ozone, warmup_temperature, 2)

    interpreted = new_state(NCOLUMNS)
    py_handle = PyHandle(interpreted, NLEVELS)
    timed(
        "interpreted, cells",
        for_each_cell_b.py_func,
        make_heat_b(RATE),
        py_handle,
        NCOLUMNS,
        NLEVELS,
    )
    timed(
        "interpreted, columns",
        for_each_column_b.py_func,
        make_react_b(*CHEMISTRY),
        py_handle,
        NCOLUMNS,
    )

    shape_a = new_state(NCOLUMNS)
    timed("shape A, cells", for_each_cell_a, heat_a, shape_a["temperature"])
    timed("shape A, columns", sweep_down_a, react_a, 1.0, shape_a["ozone"], shape_a["temperature"])

    shape_b = new_state(NCOLUMNS)
    handle = compiled_handle(shape_b)
    timed("shape B, cells", for_each_cell_b, heat_b, handle, NCOLUMNS, NLEVELS)
    timed("shape B, columns", for_each_column_b, react_b, handle, NCOLUMNS)

    shape_c = new_state(NCOLUMNS)
    cursor = Cursor(NLEVELS)
    ozone = Variable(shape_c["ozone"], cursor)
    temperature = Variable(shape_c["temperature"], cursor)
    timed("shape C, cells", for_each_cell_c, heat_c, cursor, temperature, NCOLUMNS, NLEVELS)
    timed("shape C, columns", for_each_column_c, react_c, cursor, ozone, temperature, NCOLUMNS)

    for name in interpreted:
        np.testing.assert_allclose(shape_a[name], interpreted[name], rtol=1e-12)
        np.testing.assert_allclose(shape_b[name], interpreted[name], rtol=1e-12)
        np.testing.assert_allclose(shape_c[name], interpreted[name], rtol=1e-12)
    print("All three give the same results.")


if __name__ == "__main__":
    main()
