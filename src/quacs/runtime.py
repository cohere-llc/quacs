# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Top-level QUACS API for host application runtime requests."""

from __future__ import annotations

from collections.abc import Mapping

from quacs.hooks import HookDispatcher
from quacs.host_adapter import HostAdapter
from quacs.parameterization import Parameterization
from quacs.registry import ParameterizationRegistry
from quacs.state import QuacsState
from quacs.variables import VariableDefinition, VariableSpec, host_name_map


class Runtime:
    """Runs a set of parameterizations on a host application.
    
    `hooks` gives the names of the parameterizations to run for each hook label, in order.

    `names` gives a mapping from VariableDefinitions used by the current set of QUACS
    parameterizations to the host applications variable names, when they differ.
    """
    def __init__(
        self,
        host_adapter: HostAdapter,
        parameterizations: list[Parameterization],
        hooks: dict[str, list[str]],
        names: Mapping[VariableDefinition, str] | None = None,
    ) -> None:
        self._host_adapter = host_adapter
        self._names = host_name_map(names)
        self._registry = ParameterizationRegistry()
        for param in parameterizations:
            self._registry.register(param)
        self._dispatcher = HookDispatcher(self._registry, hooks)

    def register_variables(self) -> list[VariableSpec]:
        """Returns a list of host variables to register or locate."""
        return self._registry.collect_variable_specs(self._names)

    def run_hook(self, label: str) -> None:
        state = QuacsState(self._host_adapter, self._names)
        self._dispatcher.run(label, state)
