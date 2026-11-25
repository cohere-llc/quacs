# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Tests of the rules the registry applies to the variables of a parameterization."""

from __future__ import annotations

import pytest

from quacs import QuacsState, VariableDefinition, VariableSpec
from quacs.registry import ParameterizationRegistry
from quacs.shapes import GridShape
from quacs.variables import DType, Permissions


class OneVariable:
    name = "one_variable"

    def __init__(self, spec: VariableSpec, name: str = "one_variable") -> None:
        self.name = name
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

def test_definition_makes_requests_with_each_permission() -> None:
    ozone = VariableDefinition("ozone", GridShape.CELL, "float", "mol mol-1", "Ozone")
    assert ozone.own() == VariableSpec(
        "ozone", GridShape.CELL, "float", "mol mol-1", "Ozone", Permissions.OWN
    )
    assert ozone.update().permissions is Permissions.UPDATE
    assert ozone.read_only().permissions is Permissions.READ_ONLY


def test_transported_definition_is_transported_for_its_owner_only() -> None:
    ozone = VariableDefinition("ozone", GridShape.CELL, "float", "1", "Ozone", transported=True)
    assert ozone.own().transported
    assert not ozone.update().transported
    assert not ozone.read_only().transported
    with pytest.raises(ValueError, match="cannot be transported"):
        VariableDefinition("flux", GridShape.INTERFACE, "float", "1", "A flux", transported=True)


def test_names_give_the_host_names_of_variables() -> None:
    temperature = VariableDefinition("temperature", GridShape.CELL, "float", "K", "Temperature")
    ozone = VariableDefinition("ozone", GridShape.CELL, "float", "1", "Ozone")
    registry = ParameterizationRegistry()
    registry.register(OneVariable(temperature.read_only(), "first"))
    registry.register(OneVariable(ozone.own(), "second"))
    specs = registry.collect_variable_specs({"temperature": "t_host", "not_used": "other"})
    assert [spec.name for spec in specs] == ["ozone", "t_host"]


def test_different_definitions_of_one_name_are_reported() -> None:
    kelvin = VariableDefinition("temperature", GridShape.CELL, "float", "K", "Temperature")
    celsius = VariableDefinition("temperature", GridShape.CELL, "float", "degC", "Temperature")
    registry = ParameterizationRegistry()
    registry.register(OneVariable(kelvin.read_only(), "first"))
    registry.register(OneVariable(kelvin.update(), "same_definition"))
    with pytest.raises(ValueError, match="defined differently by 'first' .* and 'second'"):
        registry.register(OneVariable(celsius.read_only(), "second"))