# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Tests of `EmbeddedRuntime`, with NumPy arrays standing in for a host application's memory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quacs.embed import EmbeddedRuntime, Storage
from quacs.exceptions import HostConfigError, UnknownVariableError
from quacs.shapes import GridShape
from quacs.variables import Permissions, VariableSpec

NCOLUMNS = 5
NLEVELS = 4

CHECKS = """
from quacs.conformance import VerticalOrderCheck, default_checks

parameterizations = [*default_checks(), VerticalOrderCheck("height", "m", increases_upward=True)]
hooks = {{"step": [check.name for check in parameterizations]}}
compiled = {compiled}
names = {{"height": "z"}}
"""

# A parameterization that multiplies a host variable by a value the host application sets.
SCALING = """
from quacs import VariableSpec
from quacs.shapes import GridShape
from quacs.variables import Permissions


class Scaling:
    name = "scaling"

    def register_variables(self):
        return [
            VariableSpec("factor", GridShape.DOMAIN, "float", "1", "", Permissions.READ_ONLY),
            VariableSpec("ozone", GridShape.CELL, "float", "1", "", Permissions.UPDATE),
        ]

    def run(self, state):
        def scale(ozone, factor):
            ozone.set(ozone.get() * factor)

        state.for_each_cell(scale, "ozone", "factor")


parameterizations = [Scaling()]
hooks = {"step": ["scaling"]}
"""


def write(folder: Path, text: str) -> Path:
    path = folder / "config.py"
    path.write_text(text)
    return path


def allocate(spec: VariableSpec, storage: Storage) -> np.ndarray:
    """Host memory for a variable, with the model top first for a variable with levels."""
    types = {
        Storage.FLOAT32: np.float32,
        Storage.FLOAT64: np.float64,
        Storage.INT32: np.int32,
        Storage.INT64: np.int64,
        Storage.BOOL: np.bool_,
    }
    shapes = {
        GridShape.CELL: (NCOLUMNS, NLEVELS),
        GridShape.INTERFACE: (NCOLUMNS, NLEVELS + 1),
        GridShape.DOMAIN: (),
    }
    return np.zeros(shapes.get(spec.shape, (NCOLUMNS,)), dtype = types[storage])


def bind(runtime: EmbeddedRuntime, name: str, array: np.ndarray, storage: Storage) -> None:
    """Binds an array that holds the model top first, as a host application might."""
    if array.ndim == 2:
        surface_first = array[:, ::-1]
        runtime.bind_variable(name, surface_first.ctypes.data, storage, *surface_first.strides)
    else:
        runtime.bind_variable(name, array.ctypes.data, storage, *array.strides)


def start(
    folder: Path, compiled: bool, storages: dict[str, Storage]
) -> tuple[EmbeddedRuntime, dict[str, np.ndarray]]:
    """Runtime for the conformance checks, with every variable created and bound."""
    runtime = EmbeddedRuntime(write(folder, CHECKS.format(compiled=compiled)))
    runtime.set_grid(NCOLUMNS, NLEVELS)
    memory = {}
    for spec in runtime.variables():
        storage = storages[spec.dtype]
        memory[spec.name] = allocate(spec, storage)
        bind(runtime, spec.name, memory[spec.name], storage)
    memory["z"][...] = 100.0 * np.arange(NLEVELS, 0, -1)
    return runtime, memory


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(
    "storages",
    [
        {"float": Storage.FLOAT64, "int": Storage.INT32, "bool": Storage.BOOL},
        {"float": Storage.FLOAT32, "int": Storage.INT64, "bool": Storage.INT32},
    ]
)
def test_conformance_checks_pass_in_host_memory(
    tmp_path: Path, compiled: bool, storages: dict[str, Storage]
) -> None:
    runtime, memory = start(tmp_path, compiled, storages)
    assert runtime.check_conformance().passed
    for _ in range(3):
        runtime.run_hook("step")
    report = runtime.check_conformance()
    assert report.passed, str(report)

    # The host should see the results in its native layout, with the model top first.
    np.testing.assert_array_equal(memory["quacs_check_level"][0], [4, 3, 2, 1])
    np.testing.assert_array_equal(memory["quacs_check_interface_number"][0], [5, 4, 3, 2, 1])
    np.testing.assert_array_equal(memory["quacs_check_cell_tag"][:, 0], [1, 2, 3, 4, 5])

    # A bool stored as an integer is exactly 0 or 1.
    assert set(np.unique(memory["quacs_check_column_swept"]).tolist()) == {1}
    assert set(np.unique(memory["quacs_check_is_boundary"]).tolist()) == {0, 1}


def test_variables_are_described_for_the_host(tmp_path: Path) -> None:
    runtime = EmbeddedRuntime(write(tmp_path, CHECKS.format(compiled=False)))
    specs = {spec.name: spec for spec in runtime.variables()}
    assert "height" not in specs
    assert specs["z"].permissions is Permissions.READ_ONLY
    assert specs["quacs_check_tracer"].permissions is Permissions.OWN
    assert specs["quacs_check_tracer"].transported
    names = [runtime.variable_info(index)[0] for index in range(len(specs))]
    assert names == list(specs)
    assert runtime.variable_info(names.index("quacs_check_tracer")) == (
        "quacs_check_tracer",
        "1",
        "Set to 1 once",
        1,
        1,
        3,
        True,
    )


def test_variable_can_move_in_memory(tmp_path: Path) -> None:
    storages = {"float": Storage.FLOAT64, "int": Storage.INT32, "bool": Storage.BOOL}
    runtime, memory = start(tmp_path, True, storages)
    runtime.run_hook("step")
    moved = memory["quacs_check_cell_visits"].copy()
    bind(runtime, "quacs_check_cell_visits", moved, Storage.INT32)
    runtime.run_hook("step")
    assert runtime.check_conformance().passed
    assert (moved == 2).all()
    assert (memory["quacs_check_cell_visits"] == 1).all()


def test_domain_value_is_read_from_host_memory(tmp_path: Path) -> None:
    runtime = EmbeddedRuntime(write(tmp_path, SCALING))
    runtime.set_grid(NCOLUMNS, NLEVELS)
    factor = np.array(2.0)
    ozone = np.ones((NCOLUMNS, NLEVELS))
    bind(runtime, "factor", factor, Storage.FLOAT64)
    bind(runtime, "ozone", ozone, Storage.FLOAT64)
    runtime.run_hook("step")
    factor[...] = 5.0
    runtime.run_hook("step")
    assert (ozone == 10.0).all()


def test_unbound_variable_is_reported(tmp_path: Path) -> None:
    runtime = EmbeddedRuntime(write(tmp_path, SCALING))
    with pytest.raises(HostConfigError, match="grid has not been set"):
        runtime.run_hook("step")
    runtime.set_grid(NCOLUMNS, NLEVELS)
    bind(runtime, "ozone", np.ones((NCOLUMNS, NLEVELS)), Storage.FLOAT64)
    with pytest.raises(HostConfigError, match="not been bound: factor"):
        runtime.run_hook("step")


def test_wrong_bindings_are_reported(tmp_path: Path) -> None:
    runtime = EmbeddedRuntime(write(tmp_path, SCALING))
    runtime.set_grid(NCOLUMNS, NLEVELS)
    ozone = np.ones((NCOLUMNS, NLEVELS), dtype=np.int32)
    with pytest.raises(ValueError, match="stored as int32"):
        bind(runtime, "ozone", ozone, Storage.INT32)
    with pytest.raises(UnknownVariableError):
        bind(runtime, "temperature", ozone, Storage.INT32)
    with pytest.raises(ValueError, match="null address"):
        runtime.bind_variable("ozone", 0, Storage.FLOAT64)


def test_bad_configuration_files_are_reported(tmp_path: Path) -> None:
    with pytest.raises(HostConfigError, match="does not define: hooks"):
        EmbeddedRuntime(write(tmp_path, "parameterizations = []"))
    with pytest.raises(HostConfigError, match="could not be run: NameError"):
        EmbeddedRuntime(write(tmp_path, "parameterizations = missing"))
    with pytest.raises(HostConfigError, match="could not be run: FileNotFoundError"):
        EmbeddedRuntime(str(tmp_path / "missing.py"))
    with pytest.raises(HostConfigError, match="`names` must be a dict"):
        EmbeddedRuntime(write(tmp_path, "parameterizations = []\nhooks = {}\nnames = {'a': 1}"))


def test_conflicting_requests_are_reported(tmp_path: Path) -> None:
    conflict = SCALING.replace(
        "parameterizations = [Scaling()]",
        """
class Other:
    name = "other"

    def register_variables(self):
        return [VariableSpec("flux", GridShape.INTERFACE, "float", "1", "", Permissions.OWN)]

    def run(self, state):
        pass


parameterizations = [Scaling(), Other()]
names = {"flux": "ozone"}""",
    )
    runtime = EmbeddedRuntime(write(tmp_path, conflict))
    with pytest.raises(ValueError, match="'ozone' is a INTERFACE float for 'other'"):
        runtime.variables()
