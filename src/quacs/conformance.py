# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Checks that a host application runs QUACS parameterizations correctly.

Each check is a parameterization that writes a know pattern into variables that it owns,
using one part of the QUACS interface. A check knows nothing about the host application.
`verify` then compares the variables with the values they should hold, which needs a
view of each variable from the host application.

A host application built on `ArrayHostAdapter` can run everything with `run_conformance`.
Any other host application adds the checks to its hooks like any parmeterization, and
calls `verify` to get a conformance report.

Every variable is named `quacs_check_*`, so that the patterns can also be looked at in
the host applications own output without name conflicts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from quacs.accessors import Column, InterfaceVariable, Variable, above, below, lower, upper
from quacs.array_host import ArrayHostAdapter
from quacs.exceptions import ReadOnlyVariableError
from quacs.runtime import Runtime
from quacs.shapes import GridShape
from quacs.state import QuacsState
from quacs.variables import DType, Permissions, VariableSpec

# Returns a host variable by name, in the standard layout of `ArrayHostAdapter.view`.
type View = Callable[[str], np.ndarray]

# The values a variable should hold and the rule that describes the logic used.
type Expected = dict[str, tuple[Any, str]]

_COLUMN_TAG = "quacs_check_column_tag"


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one check. `detail` describes the failure, when there is one."""

    name: str
    rule: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class ConformanceReport:
    results: list[CheckResult]

    @property
    def passed(self):
        return all(result.passed for result in self.results)

    def __str__(self) -> str:
        passes = sum(result.passed for result in self.results)
        lines = [f"QUACS conformance: {passes} of {len(self.results)} checks passed."]
        for result in self.results:
            lines.append(f"  {'PASS' if result.passed else 'FAIL'} {result.name}: {result.rule}")
            if result.detail:
                lines.append(f"        {result.detail}")
        return "\n".join(lines)


class Check(Protocol):
    """A parameterization that can report whether the host application ran it correctly.
    
    `runs` is the number of times the check has been run.
    """

    name: str
    runs: int

    def register_variables(self) -> list[VariableSpec]: ...

    def run(self, state: QuacsState) -> None: ...

    def results(self, view: View) -> list[CheckResult]: ...


def _own(
    name: str, shape: GridShape, dtype: DType, description: str, transported: bool = False
) -> VariableSpec:
    return VariableSpec(
        name, shape, dtype, "1", description, Permissions.OWN, transported=transported
    )


def _compare(
    view: View, runs: int, expected: Expected, tolerance: float = 0.0
) -> list[CheckResult]:
    """Compare each variable with the expected values.
    
    This expects that all values start at zero or False before the parameterization is run.
    """
    results = []
    for name, (values, rule) in expected.items():
        if runs == 0:
            values, rule = 0, "is zero or False before the check has run"
        actual = np.asarray(view(name))
        try:
            expected = np.broadcast_to(values, actual.shape)
        except ValueError:
            detail = f"The view has shape {actual.shape}, which does not fit the grid."
            results.append(CheckResult(name, rule, False, detail))
            continue
        if actual.dtype.kind == "f":
            matches = np.isclose(actual, expected, rtol=0.0, atol=tolerance)
        else:
            matches = actual == expected
        if matches.all():
            results.append(CheckResult(name, rule, True))
            continue
        first = tuple(int(index) for index in np.argwhere(~matches)[0])
        detail = (
            f"{int((~matches).sum())} of {matches.size} values differ. The first is at index "
            f"{first} of the view: expected {expected[first].item()}, found {actual[first].item()}."
        )
        results.append(CheckResult(name, rule, False, detail=detail))
    return results


class CellCheck:
    """Checks `for_each_cell`, and that variables keep their values between runs."""

    name = "quacs_check_cells"

    def __init__(self) -> None:
        self.runs = 0

    def register_variables(self) -> list[VariableSpec]:
        return [
            _own("quacs_check_cell_visits", GridShape.CELL, "int", "Times the cell was visited"),
            _own("quacs_check_cell_flag", GridShape.CELL, "bool", "Whether visits is odd"),
            _own("quacs_check_cell_value", GridShape.CELL, "float", "Half of visits"),
            _own("quacs_check_cell_tag", GridShape.CELL, "int", "Copy if the column's tag"),
            _own(_COLUMN_TAG, GridShape.COLUMN, "int", "A number set by the host application"),
        ]

    def run(self, state: QuacsState) -> None:
        def visit(
            visits: Variable[int],
            flag: Variable[bool],
            value: Variable,
            cell_tag: Variable[int],
            column_tag: Variable[int],
            scale: float,
        ) -> None:
            count = visits.get() + 1
            visits.set(count)
            flag.set(count % 2 == 1)
            value.set(scale * count)
            cell_tag.set(column_tag.get())

        state.for_each_cell(
            visit,
            "quacs_check_cell_visits",
            "quacs_check_cell_flag",
            "quacs_check_cell_value",
            "quacs_check_cell_tag",
            _COLUMN_TAG,
            0.5,
        )
        self.runs += 1

    def results(self, view: View) -> list[CheckResult]:
        tags = np.asarray(view(_COLUMN_TAG))[:, np.newaxis]
        expected: Expected = {
            "quacs_check_cell_visits": (
                self.runs,
                "every grid cell is visited once per run and keeps its value between runs",
            ),
            "quacs_check_cell_flag": (
                self.runs % 2 == 1,
                "a bool variable holds what it was set to",
            ),
            "quacs_check_cell_value": (
                0.5 * self.runs,
                "a float variable holds a fraction, and a local scalar reaches the function",
            ),
            "quacs_check_cell_tag": (tags, "a grid cell reads the variables of its own column"),
        }
        return _compare(view, self.runs, expected)


class ColumnCheck:
    """Checks `for_each_column` from both ends of the column, and the `Column` methods."""

    name = "quacs_check_columns"

    def __init__(self) -> None:
        self.runs = 0

    def register_variables(self) -> list[VariableSpec]:
        return [
            _own("quacs_check_level", GridShape.CELL, "int", "Cells counted up from the surface"),
            _own("quacs_check_depth", GridShape.CELL, "int", "Cells counted down from the top"),
            _own("quacs_check_column_levels", GridShape.COLUMN, "int", "Cells in the column"),
            _own("quacs_check_column_swept", GridShape.COLUMN, "bool", "Whether the column ran"),
            _own("quacs_check_surface", GridShape.SURFACE_CELL, "float", "Half the surface level"),
            _own("quacs_check_model_top", GridShape.MODEL_TOP_CELL, "float", "Half the top level"),
        ]

    def run(self, state: QuacsState) -> None:
        def count_up(
            column: Column,
            level: Variable[int],
            column_levels: Variable[int],
            swept: Variable[bool],
            surface: Variable,
            model_top: Variable,
        ) -> None:
            count = 1
            while True:
                level.set(count)
                if column.at_model_top():
                    break
                column.step_toward_model_top()
                count += 1
            column_levels.set(count)
            swept.set(True)
            column.select_surface_cell()
            surface.set(0.5 * level.get())
            column.select_model_top()
            model_top.set(0.5 * level.get())

        def count_down(column: Column, depth: Variable[int]) -> None:
            count = 1
            while True:
                depth.set(count)
                if column.at_surface():
                    break
                column.step_toward_surface()
                count += 1

        state.for_each_column(
            count_up,
            "quacs_check_level",
            "quacs_check_column_levels",
            "quacs_check_column_swept",
            "quacs_check_surface",
            "quacs_check_model_top",
            start="surface",
        )
        state.for_each_column(count_down, "quacs_check_depth", start="model_top")
        self.runs += 1

    def results(self, view: View) -> list[CheckResult]:
        nlevels = view("quacs_check_level").shape[1]
        levels = np.arange(1, nlevels + 1)
        expected: Expected = {
            "quacs_check_level": (levels, "a column that starts at the surface steps upward"),
            "quacs_check_depth": (levels[::-1], "a column that starts at the top steps downward"),
            "quacs_check_column_levels": (nlevels, "a column has every level of the grid"),
            "quacs_check_column_swept": (True, "every column is visited"),
            "quacs_check_surface": (0.5, "select_surface_cell selects the lowest grid cell"),
            "quacs_check_model_top": (
                0.5 * nlevels,
                "select_model_top selects the highest grid cell",
            ),
        }
        return _compare(view, self.runs, expected)


class InterfaceCheck:
    """Checks `for_each_interface`, and how grid cells and interfaces line up."""

    name = "quacs_check_interfaces"

    def __init__(self) -> None:
        self.runs = 0

    def register_variables(self) -> list[VariableSpec]:
        cell, interface = GridShape.CELL, GridShape.INTERFACE
        return [
            _own("quacs_check_interface_number", interface, "int", "Counted up from the surface"),
            _own("quacs_check_interface_cell", cell, "int", "Cells counted up from the surface"),
            _own("quacs_check_neighbours", interface, "float", "Mean number of the adjacent cells"),
            _own("quacs_check_is_boundary", interface, "bool", "Whether a neighbour is missing"),
            _own("quacs_check_cell_interfaces", cell, "int", "100 x upper + lower interface"),
        ]

    def run(self, state: QuacsState) -> None:
        def number(
            column: Column,
            cell: Variable[int],
            upper_number: InterfaceVariable[int],
            lower_number: InterfaceVariable[int],
        ) -> None:
            count = 1
            while True:
                cell.set(count)
                lower_number.set(count)
                if column.at_model_top():
                    break
                column.step_toward_model_top()
                count += 1
            upper_number.set(count + 1)

        def read_cells(
            neighbours: InterfaceVariable,
            is_boundary: InterfaceVariable[bool],
            cell_above: Variable[int] | None,
            cell_below: Variable[int] | None,
        ) -> None:
            total = 0
            if cell_above is not None:
                total += cell_above.get()
            if cell_below is not None:
                total += cell_below.get()
            neighbours.set(0.5 * total)
            is_boundary.set(cell_above is None or cell_below is None)

        def read_interfaces(
            cell_interfaces: Variable[int],
            upper_number: InterfaceVariable[int],
            lower_number: InterfaceVariable[int],
        ) -> None:
            cell_interfaces.set(100 * upper_number.get() + lower_number.get())

        state.for_each_column(
            number,
            "quacs_check_interface_cell",
            upper("quacs_check_interface_number"),
            lower("quacs_check_interface_number"),
        )
        state.for_each_interface(
            read_cells,
            "quacs_check_neighbours",
            "quacs_check_is_boundary",
            above("quacs_check_interface_cell"),
            below("quacs_check_interface_cell"),
        )
        state.for_each_cell(
            read_interfaces,
            "quacs_check_cell_interfaces",
            upper("quacs_check_interface_number"),
            lower("quacs_check_interface_number"),
        )
        self.runs += 1

    def results(self, view: View) -> list[CheckResult]:
        nlevels = view("quacs_check_interface_cell").shape[1]
        cells = np.arange(1, nlevels + 1)
        interfaces = np.arange(1, nlevels + 2)
        # There are no cells above the uppermost- or below the lowermost-interface,
        # and thus count as zero.
        above_cell = np.append(cells, 0)
        below_cell = np.append(0, cells)
        expected: Expected = {
            "quacs_check_interface_number": (
                interfaces,
                "upper() and lower() name the interfaces of the selected grid cell",
            ),
            "quacs_check_interface_cell": (
                cells,
                "a column that starts at the surface steps upward",
            ),
            "quacs_check_neighbours": (
                0.5 * (above_cell + below_cell),
                "above() and below() name the grid cells next to an interface",
            ),
            "quacs_check_is_boundary": (
                (above_cell == 0) | (below_cell == 0),
                "only the surface and the model top are missing a grid cell",
            ),
            "quacs_check_cell_interfaces": (
                100 * (cells + 1) + cells,
                "a grid cell reads its own upper and lower interfaces",
            ),
        }
        return _compare(view, self.runs, expected)


class ReadOnlyCheck:
    """Checks that a function cannot set a variable that should be read-only."""

    name = "quacs_check_read_only"

    def __init__(self) -> None:
        self.runs = 0
        self._allowed: list[str] = []

    def register_variables(self) -> list[VariableSpec]:
        return [
            _own("quacs_check_fixed_cell", GridShape.CELL, "float", "Never set"),
            _own("quacs_check_fixed_interface", GridShape.INTERFACE, "float", "Never set"),
            _own("quacs_check_fixed_column", GridShape.COLUMN, "float", "Never set"),
        ]

    def run(self, state: QuacsState) -> None:
        def set_variable(variable: Variable) -> None:
            variable.set(1.0)

        def set_cell_above(cell: Variable | None) -> None:
            if cell is not None:
                cell.set(1.0)

        attempts: list[tuple[str, Callable[[], None]]] = [
            (
                "an interface variable in for_each_cell",
                lambda: state.for_each_cell(set_variable, lower("quacs_check_fixed_interface")),
            ),
            (
                "a per-column variable in for_each_cell",
                lambda: state.for_each_cell(set_variable, "quacs_check_fixed_column"),
            ),
            (
                "a grid cell variable in for_each_interface",
                lambda: state.for_each_interface(set_cell_above, above("quacs_check_fixed_cell")),
            ),
        ]
        self._allowed = []
        for description, attempt in attempts:
            try:
                attempt()
            except ReadOnlyVariableError:
                continue
            self._allowed.append(description)
        self.runs += 1

    def results(self, view: View) -> list[CheckResult]:
        rule = "is not changed by a function that is not allowed to set it"
        expected: Expected = {
            "quacs_check_fixed_cell": (0.0, rule),
            "quacs_check_fixed_interface": (0.0, rule),
            "quacs_check_fixed_column": (0.0, rule),
        }
        if self.runs == 0:
            return _compare(view, self.runs, expected)
        detail = "; ".join(self._allowed)
        raised = CheckResult(
            self.name,
            "setting a variable of the wrong shape raises ReadOnlyVariableError",
            not self._allowed,
            f"No error was raised for: {detail}." if detail else "",
        )
        return [raised, *_compare(view, self.runs, expected)]


class TransportCheck:
    """Checks that a transported variable that is the same everywhere stays that way.

    The variable is set once, the first time the check runs. `tolerance` is how far the
    host application's transport is allowed to move it from that value.
    """

    name = "quacs_check_transport"

    def __init__(self, tolerance: float = 1.0e-6) -> None:
        self.runs = 0
        self._tolerance = tolerance

    def register_variables(self) -> list[VariableSpec]:
        return [
            _own("quacs_check_tracer", GridShape.CELL, "float", "Set to 1 once", transported=True)
        ]

    def run(self, state: QuacsState) -> None:
        def fill(tracer: Variable, value: float) -> None:
            tracer.set(value)

        if self.runs == 0:
            state.for_each_cell(fill, "quacs_check_tracer", 1.0)
        self.runs += 1

    def results(self, view: View) -> list[CheckResult]:
        expected: Expected = {
            "quacs_check_tracer": (1.0, "a transported variable that starts uniform stays uniform")
        }
        return _compare(view, self.runs, expected, self._tolerance)


class VerticalOrderCheck:
    """Checks that a variable of the host application changes in one direction with height.

    This is the only check that can tell whether the host application has its levels
    upside down. `variable` is the name of a `float` variable with one value per grid
    cell.
    """

    name = "quacs_check_vertical_order"

    def __init__(self, variable: str, units: str, increases_upward: bool) -> None:
        self.runs = 0
        self._variable = variable
        self._units = units
        self._increases_upward = increases_upward

    def register_variables(self) -> list[VariableSpec]:
        return [
            VariableSpec(
                self._variable,
                GridShape.CELL,
                "float",
                self._units,
                "A variable that changes in one direction with height",
                Permissions.READ_ONLY,
            ),
            _own("quacs_check_vertical_order", GridShape.COLUMN, "bool", "Whether it does"),
        ]

    def run(self, state: QuacsState) -> None:
        def compare(
            column: Column, value: Variable, ordered: Variable[bool], direction: float
        ) -> None:
            result = True
            previous = value.get()
            while not column.at_model_top():
                column.step_toward_model_top()
                current = value.get()
                if direction * (current - previous) <= 0.0:
                    result = False
                previous = current
            ordered.set(result)

        state.for_each_column(
            compare,
            self._variable,
            "quacs_check_vertical_order",
            1.0 if self._increases_upward else -1.0,
        )
        self.runs += 1

    def results(self, view: View) -> list[CheckResult]:
        direction = "increases" if self._increases_upward else "decreases"
        expected: Expected = {
            "quacs_check_vertical_order": (
                True,
                f"'{self._variable}' {direction} from the surface to the model top",
            )
        }
        return _compare(view, self.runs, expected)


def default_checks() -> list[Check]:
    """Returns one of each check that can be run with any host that implements ArrayHostAdapeter."""
    return [CellCheck(), ColumnCheck(), InterfaceCheck(), ReadOnlyCheck(), TransportCheck()]


def tag_columns(view: View) -> None:
    """Gives neighboring columns different tags, so that `CellCheck` can tell them apart."""
    tags = view(_COLUMN_TAG)
    tags[...] = np.arange(tags.shape[0]) % 7 + 1


def verify(checks: Iterable[Check], view: View) -> ConformanceReport:
    """Reports whether the variables of `checks` hold the expected values.
    
    `view` should return a variable by name in the standard layout of `ArrayHostAdapter.view`.
    It can return a copy.
    """
    return ConformanceReport([result for check in checks for result in check.results(view)])


def run_conformance(
    host: ArrayHostAdapter, checks: Iterable[Check] | None = None, runs: int = 3
) -> ConformanceReport:
    """Runs `checks` on `host` and reports the outcome.
    
    If `checks` is `None`, the default checks are run.

    Each check is run `runs` times. `host` must not have run the checks before, because
    this create their variables.
    """
    checks = default_checks() if checks is None else list(checks)
    runtime = Runtime(host, [*checks], {"quacs_check": [check.name for check in checks]})
    host.allocate_registered_variables(runtime.register_variables())

    owned = [
        spec.name
        for check in checks
        for spec in check.register_variables()
        if spec.permissions is Permissions.OWN
    ]
    results = [_check_name_map(host), _check_views(host, owned)]
    results += verify(checks, host.view).results
    if _COLUMN_TAG in owned:
        tag_columns(host.view)
    for _ in range(runs):
        runtime.run_hook("quacs_check")
    results += verify(checks, host.view).results
    return ConformanceReport(results)


def _check_name_map(host: ArrayHostAdapter) -> CheckResult:
    shapes = host.grid_shapes()
    missing = sorted(
        f"{name} -> {host_name}"
        for name, host_name in host.canonical_name_map().items()
        if host_name is not None and host_name not in shapes
    )
    return CheckResult(
        "canonical_name_map",
        "every host name it gives is a variable in grid_shapes",
        not missing,
        f"Not in grid_shapes: {', '.join(missing)}." if missing else "",
    )

def _check_views(host: ArrayHostAdapter, names: list[str]) -> CheckResult:
    copies = sorted(
        name
        for name in names
        if not np.shares_memory(host.view(name), host.view(name))
        or not host.view(name).flags.writeable
    )
    return CheckResult(
        "view",
        "returns the host application's own memory, which can be written to",
        not copies,
        f"A copy or a read-only array was returned for: {', '.join(copies)}." if copies else "",
    )
