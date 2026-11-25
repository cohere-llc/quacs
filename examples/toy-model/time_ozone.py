# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Measures where the compiled toy ozone chemistry spends its time.

Reports the fixed cost of each `for_each_*` operation, and the time per grid cell on
grids of increasing size. Each operation is compared with a hand-written loop that does
the same arithmetic on the same memory.

Run from the repository root with:
    uv run python examples/toy-model/time_ozone.py
"""

from __future__ import annotations

import math
import time

import config
import numba
import numpy as np
from host_adapter import ToyHostAdapter
from parameterizations import OzoneChemistry

from quacs.registry import ParameterizationRegistry
from quacs.state import QuacsState
from quacs.variables import host_name_map

NAMES = host_name_map(config.names)

NLEVELS = 50
COLUMN_COUNTS = (100, 1_000, 10_000, 100_000)

PRODUCTION_RATE = 0.02
LOSS_FRACTION = 0.05
ATTENUATION = 0.5

OPERATIONS = ("attenuate", "react")
CHECKED = ("ozone", "actinic_flux", "absorbed_flux", "chemistry_updates")


@numba.njit
def reference_attenuate(ozone, is_daytime, flux, attenuation):
    ncolumns, nlevels = ozone.shape
    for column in range(ncolumns):
        flux[column, nlevels] = 1.0 if is_daytime[column] else 0.0
        for level in range(nlevels - 1, -1, -1):
            flux[column, level] = flux[column, level + 1] * math.exp(
                -attenuation * ozone[column, level]
            )


@numba.njit
def reference_react(ozone, temperature, absorbed, updates, flux, production_rate, loss_fraction):
    ncolumns, nlevels = ozone.shape
    for column in range(ncolumns):
        for level in range(nlevels):
            current = ozone[column, level]
            production = production_rate * flux[column, level] * temperature[column, level] / 290.0
            ozone[column, level] = (current + production) * (1.0 - loss_fraction)
            absorbed[column, level] = flux[column, level + 1] - flux[column, level]
            updates[column, level] += 1


class TimingHost:
    """Passes every operation on to a host, and keeps the fastest time of each one."""

    def __init__(self, host):
        self._host = host
        self.times = {}

    def for_each_cell(self, func, *arguments):
        start = time.perf_counter()
        self._host.for_each_cell(func, *arguments)
        self._record(func, start)

    def for_each_column(self, func, *arguments, start="surface"):
        began = time.perf_counter()
        self._host.for_each_column(func, *arguments, start=start)
        self._record(func, began)

    def for_each_interface(self, func, *arguments):
        start = time.perf_counter()
        self._host.for_each_interface(func, *arguments)
        self._record(func, start)

    def _record(self, func, start):
        elapsed = time.perf_counter() - start
        self.times[func.__name__] = min(elapsed, self.times.get(func.__name__, math.inf))


def new_chemistry():
    return OzoneChemistry(PRODUCTION_RATE, LOSS_FRACTION, ATTENUATION)


def new_host(ncolumns, nlevels):
    """A compiled toy host holding the ozone chemistry's variables, with fixed random values."""
    host = ToyHostAdapter(ncolumns, nlevels)
    registry = ParameterizationRegistry()
    registry.register(new_chemistry())
    host.create_variables(registry.collect_variable_specs(NAMES))
    rng = np.random.default_rng(0)
    host.view("ozone")[...] = rng.uniform(0.0, 1.0, (ncolumns, nlevels))
    host.view("t_host")[...] = rng.uniform(200.0, 320.0, (ncolumns, nlevels))
    host.view("is_daytime")[...] = rng.random(ncolumns) < 0.9
    return host


def reference_arguments(host):
    ozone = host.view("ozone")
    flux = host.view("actinic_flux")
    return {
        "attenuate": (ozone, host.view("is_daytime"), flux, ATTENUATION),
        "react": (
            ozone,
            host.view("t_host"),
            host.view("absorbed_flux"),
            host.view("chemistry_updates"),
            flux,
            PRODUCTION_RATE,
            LOSS_FRACTION,
        ),
    }


REFERENCES = {"attenuate": reference_attenuate, "react": reference_react}


def check_reference():
    """Check that the hand-written loops give the same results as the parameterization."""
    quacs_host = new_host(100, NLEVELS)
    reference_host = new_host(100, NLEVELS)
    new_chemistry().run(QuacsState(quacs_host, NAMES))
    arguments = reference_arguments(reference_host)
    for name in OPERATIONS:
        REFERENCES[name](*arguments[name])
    for name in CHECKED:
        np.testing.assert_allclose(
            reference_host.view(name), quacs_host.view(name), rtol=1e-12, err_msg=name
        )


def time_quacs(host, repeats):
    """Returns the fastest whole run, and the fastest time of each operation, in seconds."""
    timing = TimingHost(host)
    state = QuacsState(timing, NAMES)
    chemistry = new_chemistry()
    chemistry.run(state)  # compiles the functions
    timing.times.clear()
    whole = math.inf
    for _ in range(repeats):
        start = time.perf_counter()
        chemistry.run(state)
        whole = min(whole, time.perf_counter() - start)
    return whole, timing.times


def time_reference(host, repeats):
    """Returns the fastest time of each hand-written loop, in seconds."""
    arguments = reference_arguments(host)
    best = {}
    for name in OPERATIONS:
        REFERENCES[name](*arguments[name])  # compiles the loop
        best[name] = math.inf
        for _ in range(repeats):
            start = time.perf_counter()
            REFERENCES[name](*arguments[name])
            best[name] = min(best[name], time.perf_counter() - start)
    return best


def main():
    check_reference()
    print("The hand-written loops give the same results as the parameterization.\n")

    _, fixed = time_quacs(new_host(1, 1), repeats=2_000)
    print("Fixed cost of one operation (1 column x 1 level):")
    for name in OPERATIONS:
        print(f"  {name:<10}{fixed[name] * 1e6:>8.1f} us")

    print("\nTime per grid cell in ns. The fixed cost is removed from the QUACS columns.")
    print(
        f"{'columns':>8}{'cells':>11}"
        f"{'attenuate':>12}{'reference':>11}{'react':>9}{'reference':>11}{'whole run':>11}"
    )
    for ncolumns in COLUMN_COUNTS:
        cells = ncolumns * NLEVELS
        repeats = max(5, min(500, 20_000_000 // cells))
        host = new_host(ncolumns, NLEVELS)
        whole, quacs = time_quacs(host, repeats)
        reference = time_reference(host, repeats)
        row = f"{ncolumns:>8,}{cells:>11,}"
        for name, width in zip(OPERATIONS, (12, 9)):
            row += f"{(quacs[name] - fixed[name]) / cells * 1e9:>{width}.2f}"
            row += f"{reference[name] / cells * 1e9:>11.2f}"
        row += f"{whole / cells * 1e9:>11.2f}"
        print(row)


if __name__ == "__main__":
    main()
