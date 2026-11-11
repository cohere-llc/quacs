# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Parameterization Protocol.

QUACS parameterizations must each implement the `Parameterization` protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from quacs.state import QuacsState
from quacs.variables import VariableSpec


@runtime_checkable
class Parameterization(Protocol):
    """A single science parameterization.

    `name` should uniquely identify this parameterization and will be used in the
    host application configuration to determine when it runs.
    """

    name: str

    def register_variables(self) -> list[VariableSpec]:
        """Return the set of state variables used by the parameterization."""
        ...

    def run(self, state: QuacsState) -> None:
        """Run this parameterization for the current state.

        State variables registered as `UPDATE` or `OWN` may be updated. `UPDATE`
        variables should be adjusted rather than simply overwritten.
        """
        ...
