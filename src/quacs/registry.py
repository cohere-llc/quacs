# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Aggregates and validates registered Parameterizations and their VariableSpecs."""

from __future__ import annotations

import dataclasses
import itertools

from quacs.exceptions import NameConflictError, OverwriteVariableError
from quacs.parameterization import Parameterization
from quacs.variables import Permissions, VariableSpec


class ParameterizationRegistry:
    """Holds all configured Parameterization instances for a run."""

    def __init__(self) -> None:
        self._parameterizations: dict[str, Parameterization] = {}
        self._owned_variables: dict[str, VariableSpec] = {}
        self._accessed_variables: dict[str, VariableSpec] = {}

    def register(self, parameterization: Parameterization) -> None:
        """Add a parameterization.

        Parameterizations must have unique names.
        """
        if parameterization.name in self._parameterizations:
            raise NameConflictError(
                f"Multiple parameterizations with the name '{parameterization.name}'"
            )
        vars = parameterization.register_variables()
        for var in vars:
            if var.permissions == Permissions.OWN:
                if var.name in self._owned_variables:
                    raise OverwriteVariableError(
                        var.name,
                        [
                            self._owned_variables[var.name].requested_by or "unknown",
                            parameterization.name,
                        ],
                    )
                self._owned_variables[var.name] = dataclasses.replace(
                    var, requested_by=parameterization.name
                )
            else:
                self._accessed_variables[var.name] = dataclasses.replace(
                    var, requested_by=parameterization.name
                )
        self._parameterizations[parameterization.name] = parameterization

    def get(self, name: str) -> Parameterization:
        """Look up a parameterization by name."""
        if not (param := self._parameterizations.get(name)):
            raise KeyError(f"Parameterization '{name}' not registered.")
        return param

    def __getitem__(self, name: str) -> Parameterization:
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._parameterizations

    def collect_variable_specs(self, host_name_map: dict[str, str | None]) -> list[VariableSpec]:
        """Collect and validate Parameterization VariableSpecs.

        Variables must have exactly one owner. Variables can be owned either by the host application
        (pre-existing variables) or a parameterization (registered with `OWN` permissions)
        """
        return [
            dataclasses.replace(var, name=host_name_map[var.name] or var.name)
            if var.name in host_name_map
            else var
            for var in itertools.chain(
                self._owned_variables.values(), self._accessed_variables.values()
            )
        ]
