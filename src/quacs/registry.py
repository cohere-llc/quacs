# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Aggregates and validates registered Parameterizations and their VariableSpecs."""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Mapping

from quacs.exceptions import NameConflictError, OverwriteVariableError, ReadOnlyVariableError
from quacs.parameterization import Parameterization
from quacs.shapes import GridShape
from quacs.variables import Permissions, VariableSpec, requested_variables


class ParameterizationRegistry:
    """Holds all configured Parameterization instances for a run."""

    def __init__(self) -> None:
        self._parameterizations: dict[str, Parameterization] = {}
        self._owned_variables: dict[str, VariableSpec] = {}
        self._accessed_variables: dict[str, VariableSpec] = {}
        self._requests: dict[str, list[VariableSpec]] = {}

    def register(self, parameterization: Parameterization) -> None:
        """Add a parameterization.

        Parameterizations must have unique names.
        """
        if parameterization.name in self._parameterizations:
            raise NameConflictError(
                f"Multiple parameterizations with the name '{parameterization.name}'"
            )
        vars = requested_variables(parameterization)
        for var in vars:
            if var.shape == GridShape.DOMAIN and var.permissions is not Permissions.READ_ONLY:
                raise ReadOnlyVariableError(
                    var.name,
                    "it has one value for the whole domain, which only the host application "
                    "can set. Register it as READ_ONLY.",
                )
            if var.transported and (
                var.permissions is not Permissions.OWN
                or var.shape is not GridShape.CELL
                or var.dtype != "float"
            ):
                raise ValueError(
                    f"Variable '{var.name}' cannot be transported. Only a float variable "
                    f"with one value per grid cell can be, and only its owner can request it."
                )
            var = dataclasses.replace(var, requested_by=parameterization.name)
            if var.name in self._requests and len(self._requests[var.name]) > 0:
                prior = self._requests[var.name][0]
                if (prior.shape, prior.dtype, prior.units) != (var.shape, var.dtype, var.units):
                    raise ValueError(
                        f"Variable '{var.name}' is defined differently by '{prior.requested_by}' "
                        f"({prior.shape.name}, {prior.dtype}, {prior.units}) and "
                        f"'{var.requested_by}' ({var.shape.name}, {var.dtype}, {var.units})."
                    )
                self._requests[var.name].append(var)
            else:
                self._requests[var.name] = [var]
            if var.permissions == Permissions.OWN:
                if var.name in self._owned_variables:
                    raise OverwriteVariableError(
                        var.name,
                        [
                            self._owned_variables[var.name].requested_by or "unknown",
                            parameterization.name,
                        ],
                    )
                self._owned_variables[var.name] = var
            else:
                self._accessed_variables[var.name] = var
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

    def collect_variable_specs(self, names: Mapping[str, str]) -> list[VariableSpec]:
        """Collect and validate Parameterization VariableSpecs.

        `names` defines a mapping from variable names used by the current set of QUACS
        parameterizations to the host applications variable names, when they differ.

        Variables must have exactly one owner. Variables can be owned either by the host application
        (pre-existing variables) or a parameterization (registered with `OWN` permissions)
        """
        return [
            dataclasses.replace(var, name=names.get(var.name, var.name))
            for var in itertools.chain(
                self._owned_variables.values(), self._accessed_variables.values()
            )
        ]
