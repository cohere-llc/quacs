# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Top-level QUACS API for host application runtime requests."""

from __future__ import annotations

from quacs.hooks import HookDispatcher
from quacs.host_adapter import HostAdapter
from quacs.parameterization import Parameterization
from quacs.registry import ParameterizationRegistry
from quacs.variables import VariableSpec
from quacs.state import QuacsState


class Runtime:
    def __init__(
        self,
        host_adapter: HostAdapter,
        parameterizations: list[Parameterization],
        hooks: dict[str, list[str]],
    ) -> None:
        self._host_adapter = host_adapter
        self._registry = ParameterizationRegistry()
        for param in parameterizations:
            self._registry.register(param)
        self._dispatcher = HookDispatcher(self._registry, hooks)

    def register_variables(self) -> list[VariableSpec]:
        """Returns a list of host variables to register or locate."""
        return self._registry.collect_variable_specs(self._host_adapter.canonical_name_map())

    def run_hook(self, label: str) -> None:
        state = QuacsState(self._host_adapter, self._host_adapter.canonical_name_map())
        self._dispatcher.run(label, state)
