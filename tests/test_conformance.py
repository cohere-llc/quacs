# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Tests of the conformance checks, on a correct host and on hosts with known mistakes."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from quacs import Runtime, VariableDefinition, VariableSpec
from quacs.array_host import ArrayHostAdapter
from quacs.conformance import (
    CellCheck,
    ConformanceReport,
    VerticalOrderCheck,
    default_checks,
    run_conformance,
    verify,
)
from quacs.interpreted import InterpretedEngine
from quacs.shapes import GridShape

_STORAGE_TYPES = {"float": np.float64, "int": np.int32, "bool": np.bool_}

# Host calls their height variable `z`.
HEIGHT = VariableDefinition("height", GridShape.CELL, "float", "m", "Height of a grid cell")
NAMES = {HEIGHT: "z"}


class Host(ArrayHostAdapter):
    """A host that stores every variable in the standard layout, and has a height."""

    def __init__(self, ncolumns: int = 4, nlevels: int = 3, compiled: bool = False) -> None:
        super().__init__(ncolumns, nlevels, compiled=compiled)
        heights = 100.0 * np.arange(1, nlevels + 1)
        self._arrays: dict[str, np.ndarray] = {"z": np.tile(heights, (ncolumns, 1))}
        self._shapes: dict[str, GridShape] = {"z": GridShape.CELL}

    def grid_shapes(self) -> Mapping[str, GridShape]:
        return self._shapes

    def view(self, host_name: str) -> np.ndarray:
        return self._arrays[host_name]

    def create_variable(self, spec: VariableSpec) -> None:
        self._arrays[spec.name] = np.zeros(
            self.view_shape(spec.shape), dtype=_STORAGE_TYPES[spec.dtype]
        )
        self._shapes[spec.name] = spec.shape


class CopyingHost(Host):
    def view(self, host_name: str) -> np.ndarray:
        return super().view(host_name).copy()


class UnzeroedHost(Host):
    def create_variable(self, spec: VariableSpec) -> None:
        super().create_variable(spec)
        self._arrays[spec.name][...] = 1


class UpsideDownHost(Host):
    """Stores the model top first, and forgets to reverse its views."""

    def __init__(self) -> None:
        super().__init__()
        self._arrays["z"] = self._arrays["z"][:, ::-1].copy()


class ShortHost(Host):
    """Leaves out the last column."""

    def __init__(self) -> None:
        super().__init__()
        self._engine = InterpretedEngine(self.ncolumns - 1, self.nlevels)


def failures(report: ConformanceReport) -> set[str]:
    return {result.name for result in report.results if not result.passed}


@pytest.mark.parametrize("compiled", [False, True])
def test_correct_host_passes(compiled: bool) -> None:
    checks = [*default_checks(), VerticalOrderCheck(HEIGHT, increases_upward=True)]
    report = run_conformance(Host(compiled=compiled), checks, names=NAMES)
    assert report.passed, str(report)


def test_each_column_is_told_apart() -> None:
    host = Host(ncolumns=9)
    assert run_conformance(host).passed
    np.testing.assert_array_equal(
        host.view("quacs_check_cell_tag")[:, 0], [1, 2, 3, 4, 5, 6, 7, 1, 2]
    )


def test_copying_view_fails() -> None:
    failed = failures(run_conformance(CopyingHost()))
    assert {"view", "quacs_check_cell_visits", "quacs_check_level"} <= failed


def test_unzeroed_variables_fail() -> None:
    report = run_conformance(UnzeroedHost())
    zero_rule = "is zero or False before the check has run"
    assert any(not result.passed and result.rule == zero_rule for result in report.results)


def test_upside_down_levels_fail_only_the_vertical_order() -> None:
    checks = [*default_checks(), VerticalOrderCheck(HEIGHT, increases_upward=True)]
    assert failures(run_conformance(UpsideDownHost(), checks, names=NAMES)) == {
        "quacs_check_vertical_order"
    }


def test_decreasing_variable_passes() -> None:
    checks = [VerticalOrderCheck(HEIGHT, increases_upward=False)]
    assert run_conformance(UpsideDownHost(), checks, names=NAMES).passed


def test_skipped_column_fails() -> None:
    report = run_conformance(ShortHost())
    assert {"quacs_check_cell_visits", "quacs_check_column_swept"} <= failures(report)
    visits = next(r for r in report.results if r.name == "quacs_check_cell_visits" and not r.passed)
    assert "index (3, 0)" in visits.detail


def test_values_lost_between_runs_fail() -> None:
    """A host application that runs the checks from its own hooks, and loses a variable."""
    host = Host()
    checks = [CellCheck()]
    runtime = Runtime(host, [*checks], {"step": ["quacs_check_cells"]})
    host.allocate_registered_variables(runtime.register_variables())

    runtime.run_hook("step")
    assert verify(checks, host.view).passed

    host.view("quacs_check_cell_visits")[...] = 0
    runtime.run_hook("step")
    report = verify(checks, host.view)
    assert "quacs_check_cell_visits" in failures(report)
    assert "expected 2, found 1" in str(report)


def test_owned_variable_can_have_another_name_in_the_host() -> None:
    host = Host()
    level = VariableDefinition("quacs_check_level", GridShape.CELL, "int", "1", "")
    assert run_conformance(host, names={level: "host_level"}).passed
    assert "host_level" in host.grid_shapes()
    assert "quacs_check_level" not in host.grid_shapes()


def test_vertical_order_needs_a_float_variable_with_one_value_per_cell() -> None:
    surface_height = VariableDefinition("surface_height", GridShape.COLUMN, "float", "m", "")
    with pytest.raises(ValueError, match="cannot be used to check the vertical order"):
        VerticalOrderCheck(surface_height, increases_upward=True)
