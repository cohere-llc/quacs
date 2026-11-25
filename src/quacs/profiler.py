# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""A simple profiler for parameterization developers.

`profile` runs one parameterization many times on a host, starting each run from
conditions determined by Latin hypercube sampling. It reports the mean per-grid-cell
runtime and standard deviation, and whether the parameterization functions can be
compiled by Numba.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numba
import numpy as np
from scipy.stats import qmc

from quacs.accessors import Adjacent, Argument, Scalar
from quacs.host_adapter import HostAdapter
from quacs.parameterization import Parameterization
from quacs.registry import ParameterizationRegistry
from quacs.shapes import GridShape
from quacs.state import QuacsState
from quacs.variables import Permissions, VariableDefinition, VariableSpec, host_name_map

_ZEROS: dict[str, Scalar] = {"float": 0.0, "int": 0, "bool": False}

_Call = tuple[str, Callable[..., None], tuple[Argument, ...]]

_SCALAR_TYPES: dict[str, Any] = {"float": numba.float64, "int": numba.int64, "bool": numba.boolean}


class ProfilingHost(HostAdapter, Protocol):
    """A host application that will create and manage the state for the profiling run."""

    def create_variables(self, specs: list[VariableSpec]) -> None:
        """Creates every variable in `specs` that the host does not already have."""
        ...

    def grid_cell_count(self) -> int:
        """Returns the number of grid cells in the host's domain."""
        ...

    def domain_mean(self, host_name: str) -> float:
        """Returns the mean of a variable over the host's whole domain."""
        ...

    def fill(self, host_name: str, value: Scalar) -> None:
        """Sets a variable to `value` everywhere in the host's domain."""
        ...


@dataclass(frozen=True)
class FunctionReport:
    """Describes whether a function passed to a `for_each_*` operation can be compiled by Numba.

    `compiles` is `None` if the check could not be made.
    """

    name: str
    operation: str
    compiles: bool | None
    detail: str


@dataclass(frozen=True)
class ProfileReport:
    """The result of profiling one parameterization.


    `times` holds the runtime of each sample in seconds.
    `starting_values` holds the value each sampled variable started from in each sample.
    `ending_means` holds the domain mean of every variable the parameterization uses at
    the end of each sample.
    """

    parameterization: str
    grid_cells: int
    times: np.ndarray
    starting_values: dict[str, np.ndarray]
    ending_means: dict[str, np.ndarray]
    functions: list[FunctionReport]

    @property
    def cell_times(self) -> np.ndarray:
        """The run time of each sample per grid cell, in seconds."""
        return self.times / self.grid_cells

    @property
    def mean(self) -> float:
        return float(self.cell_times.mean())

    @property
    def standard_deviation(self) -> float:
        return float(self.cell_times.std(ddof=1))

    def __str__(self) -> str:
        statuses = {
            True: "compiles with Numba",
            False: "does not compile with Numba",
            None: "compilation status unknown",
        }
        cell_times = self.cell_times
        lines = [
            (
                f"Profile of '{self.parameterization}' over {len(self.times)} samples "
                f"on {self.grid_cells} grid cells. Times are per-grid-cell."
            ),
            f"  mean:                   {_format_time(self.mean)}",
            f"  standard deviation:     {_format_time(self.standard_deviation)}",
            f"  fastest sample:         {_format_time(float(cell_times.min()))}",
            f"  slowest sample:         {_format_time(float(cell_times.max()))}",
            "  variables, from lowest to highest across the samples:",
        ]
        rows = [["", "starting value", "ending domain mean"]]
        for name, means in self.ending_means.items():
            start = self.starting_values.get(name)
            rows.append(
                [name, "0" if start is None else _format_range(start), _format_range(means)]
            )
        widths = [max(len(row[column]) for row in rows) for column in range(3)]
        for row in rows:
            cells = [cell.ljust(width) for cell, width in zip(row, widths)]
            lines.append(("    " + "  ".join(cells)).rstrip())
        lines.append("  functions:")
        for function in self.functions:
            lines.append(
                f"    {function.name} (for_each_{function.operation}): "
                f"{statuses[function.compiles]}"
            )
            lines.extend(f"      {line}" for line in function.detail.splitlines())
        return "\n".join(lines)


def profile(
    parameterization: Parameterization,
    host: ProfilingHost,
    ranges: dict[str, tuple[float, float]],
    samples: int = 50,
    seed: int = 0,
    names: Mapping[VariableDefinition, str] | None = None,
) -> ProfileReport:
    """Runs `parameterization` on `host` for `samples` different starting conditions and
    reports back timing and results.

    `ranges` gives the expected range of each `float` or `int` variable, by the name the
    parameterization uses. A range is needed for every such variable the parameterization
    uses but does not own. Each variable owned by a parameterization is set to zero or
    `False`, unless a range is provided.

    `names` gives a mapping between the current set of QUACS parameterizations variable
    names and the host application variable names, when they differ.

    The starting conditions are from a Latin hypercube sample. In each sample, the same
    value is applied to every grid cell/interface/column in the domain.
    """
    if samples < 2:
        raise ValueError("At least two samples are needed to calculate a standard deviation.")
    specs = {spec.name: spec for spec in parameterization.register_variables()}
    _check_ranges(parameterization.name, specs, ranges)

    registry = ParameterizationRegistry()
    registry.register(parameterization)
    name_map = host_name_map(names)
    host_specs = registry.collect_variable_specs(name_map)
    host.create_variables(host_specs)

    state = QuacsState(host, name_map)
    grid_cells = host.grid_cell_count()
    sampled = [spec for spec in specs.values() if _is_sampled(spec, ranges)]
    starting_values = _sample(sampled, ranges, samples, np.random.default_rng(seed))

    host_names = {name: name_map.get(name, name) or name for name in specs}

    def start_from(sample: int) -> None:
        for name, spec in specs.items():
            if name in starting_values:
                host.fill(host_names[name], starting_values[spec.name][sample].item())
            else:
                host.fill(host_names[name], _ZEROS[spec.dtype])

    # An untimed first run records the functions the parameterization uses, and gives a
    # host that compiles them the chance to do so.
    recorder = _RecordingHost(host)
    start_from(0)
    parameterization.run(QuacsState(recorder, name_map))

    times = np.empty(samples)
    ending_means = {name: np.empty(samples) for name in specs}
    for sample in range(samples):
        start_from(sample)
        start = time.perf_counter()
        parameterization.run(state)
        times[sample] = time.perf_counter() - start
        for name, host_name in host_names.items():
            ending_means[name][sample] = host.domain_mean(host_name)

    variables = {spec.name: spec for spec in host_specs}
    functions = _check_functions(recorder.calls, variables)
    return ProfileReport(
        parameterization.name, grid_cells, times, starting_values, ending_means, functions
    )


def _is_sampled(spec: VariableSpec, ranges: dict[str, tuple[float, float]]) -> bool:
    if spec.dtype == "bool":
        return spec.permissions is not Permissions.OWN
    return spec.name in ranges


def _check_ranges(
    parameterization: str, specs: dict[str, VariableSpec], ranges: dict[str, tuple[float, float]]
) -> None:
    unknown = sorted(name for name in ranges if name not in specs)
    if unknown:
        raise ValueError(
            f"Ranges were given for variables that '{parameterization}' does not use: "
            f"{', '.join(unknown)}"
        )
    booleans = sorted(name for name in ranges if specs[name].dtype == "bool")
    if booleans:
        raise ValueError(f"Ranges cannot be given for bool variables: {', '.join(booleans)}")
    missing = sorted(
        name
        for name, spec in specs.items()
        if spec.permissions is not Permissions.OWN and not _is_sampled(spec, ranges)
    )
    if missing:
        raise ValueError(
            f"A range is needed for every variable '{parameterization}' does not own. "
            f"Missing: {', '.join(missing)}"
        )


def _bounds(spec: VariableSpec, ranges: dict[str, tuple[float, float]]) -> tuple[float, float]:
    """The interval to sample for a variable.

    Integers are sampled from `[low, high + 1)` and rounded down, so that every integer
    in the range is equally likely to be sampled. Booleans are sampled from `[0, 1)`.
    """
    if spec.dtype == "bool":
        return (0.0, 1.0)
    low, high = ranges[spec.name]
    return (low, high + 1) if spec.dtype == "int" else (low, high)


def _sample(
    sampled: list[VariableSpec],
    ranges: dict[str, tuple[float, float]],
    samples: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Draw a Latin hypercube sample.

    Each variable's range is split into `samples` equal bins, and every bin is used
    exactly once.
    """
    if not sampled:
        return {}
    lower, upper = zip(*(_bounds(spec, ranges) for spec in sampled))
    unit = qmc.LatinHypercube(d=len(sampled), rng=rng).random(n=samples)
    scaled = qmc.scale(unit, lower, upper)
    points: dict[str, np.ndarray] = {}
    for column, spec in enumerate(sampled):
        values = scaled[:, column]
        if spec.dtype == "bool":
            points[spec.name] = values < 0.5
        elif spec.dtype == "int":
            points[spec.name] = np.minimum(np.floor(values), upper[column] - 1).astype(int)
        else:
            points[spec.name] = values
    return points


class _RecordingHost:
    """Passes every operation on to a host, and records the functions it was given."""

    def __init__(self, host: HostAdapter) -> None:
        self._host = host
        self.calls: list[_Call] = []

    def for_each_cell(self, func: Callable[..., None], *arguments: Argument) -> None:
        self.calls.append(("cell", func, arguments))
        self._host.for_each_cell(func, *arguments)

    def for_each_column(
        self,
        func: Callable[..., None],
        *arguments: Argument,
        start: Literal["surface", "model_top"] = "surface",
    ) -> None:
        self.calls.append(("column", func, arguments))
        self._host.for_each_column(func, *arguments, start=start)

    def for_each_interface(self, func: Callable[..., None], *arguments: Argument) -> None:
        self.calls.append(("interface", func, arguments))
        self._host.for_each_interface(func, *arguments)


def _name(argument: Argument) -> str:
    return (
        argument.name
        if isinstance(argument, Adjacent)
        else argument
        if isinstance(argument, str)
        else "local scalar"
    )


def _side(argument: Argument) -> str | None:
    return argument.side if isinstance(argument, Adjacent) else None


def _argument_type(
    argument: Argument, variables: dict[str, VariableSpec], accessors: dict[str, Any]
) -> Any:
    """The Numba type a function receives for one entry in a `for_each_*` call."""
    if not isinstance(argument, (str, Adjacent)):
        return numba.typeof(argument)
    spec = variables[_name(argument)]
    if spec.shape is GridShape.DOMAIN:
        return _SCALAR_TYPES[spec.dtype]
    return accessors[spec.dtype]


def _check_functions(
    calls: list[_Call], variables: dict[str, VariableSpec]
) -> list[FunctionReport]:
    """Try to compile each recorded function with Numba, without running it."""
    accessors, column = _stand_in_types()
    reports = []
    for operation, func, arguments in calls:
        present = [_argument_type(argument, variables, accessors) for argument in arguments]
        variants = [present]
        for side in ("above", "below"):
            if any(_side(argument) == side for argument in arguments):
                variants.append(
                    [
                        numba.types.none if _side(argument) == side else argument_type
                        for argument, argument_type in zip(arguments, present)
                    ]
                )
        leading = [column] if operation == "column" else []
        try:
            dispatcher = numba.njit(func)
            for variant in variants:
                dispatcher.compile(tuple(leading + variant))
        except Exception as error:  # noqa: BLE001
            reports.append(FunctionReport(func.__name__, operation, False, str(error)))
        else:
            reports.append(FunctionReport(func.__name__, operation, True, ""))
    return reports


def _stand_in_types() -> tuple[dict[str, Any], Any]:
    """Numba types that stand in for a host's accessors and column.

    They are only used to check that a function compiles. They are never run.
    """
    from numba import boolean, float64, int64
    from numba.experimental import jitclass

    def accessor(value_type: Any) -> Any:
        @jitclass([("value", value_type)])
        class AccessorStandIn:
            def __init__(self, value):
                self.value = value

            def get(self):
                return self.value

            def set(self, value):
                self.value = value

        return AccessorStandIn.class_type.instance_type

    @jitclass([("level", int64)])
    class ColumnStandIn:
        def __init__(self):
            self.level = 0

        def step_toward_surface(self, count=1):
            self.level += count

        def step_toward_model_top(self, count=1):
            self.level -= count

        def select_surface_cell(self):
            self.level = 0

        def select_model_top(self):
            self.level = 0

        def at_surface(self, buffer=0):
            return self.level <= buffer

        def at_model_top(self, buffer=0):
            return self.level <= buffer

    accessors = {"float": accessor(float64), "int": accessor(int64), "bool": accessor(boolean)}
    return accessors, ColumnStandIn.class_type.instance_type


def _format_time(seconds: float) -> str:
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6)):
        if seconds >= scale:
            return f"{seconds / scale:.3f} {unit}"
    return f"{seconds / 1e-9:.0f} ns"


def _format_range(values: np.ndarray) -> str:
    return f"{float(values.min()):.4g} to {float(values.max()):.4g}"
