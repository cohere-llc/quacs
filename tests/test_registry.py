# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Tests of the rules the registry applies to the variables of a parameterization."""

from __future__ import annotations

import pytest

from quacs import QuacsState, VariableSpec
from quacs.registry import ParameterizationRegistry
from quacs.shapes import GridShape
from quacs.variables import DType, Permissions


class OneVariable:
    name = "one_variable"

    def __init__(self, spec: VariableSpec) -> None:
        self._spec = spec

    def register_variables(self) -> list[VariableSpec]:
        return [self._spec]

    def run(self, state: QuacsState) -> None:
        pass


def tracer(shape: GridShape, dtype: DType, permissions: Permissions) -> VariableSpec:
    return VariableSpec("tracer", shape, dtype, "1", "A tracer", permissions, transported=True)


def test_owned_float_cell_variable_can_be_transported() -> None:
    registry = ParameterizationRegistry()
    registry.register(OneVariable(tracer(GridShape.CELL, "float", Permissions.OWN)))
    assert registry.collect_variable_specs({})[0].transported


@pytest.mark.parametrize(
    ("shape", "dtype", "permissions"),
    [
        (GridShape.INTERFACE, "float", Permissions.OWN),
        (GridShape.CELL, "int", Permissions.OWN),
        (GridShape.CELL, "float", Permissions.UPDATE),
    ],
)
def test_other_variables_cannot_be_transported(
    shape: GridShape, dtype: DType, permissions: Permissions
) -> None:
    with pytest.raises(ValueError, match="cannot be transported"):
        ParameterizationRegistry().register(OneVariable(tracer(shape, dtype, permissions)))
