# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Exception classes for QUACS."""

from __future__ import annotations


class QuacsError(Exception):
    """Base class for all QUACS errors."""


class NameConflictError(QuacsError):
    """Raised when multiple parameterizations have the same name."""


class OverwriteVariableError(QuacsError):
    """Raised when more than one parameterization tries to register a variable with `OWN` permissions."""

    def __init__(self, name: str, parameterizations: list[str]) -> None:
        self.message = f"Variable '{name}' will be set by multiple parameterizations: {', '.join(parameterizations)}"
        super().__init__(self.message)


class UnknownVariableError(QuacsError):
    """Raised when a variable is requested that does not exist in the QUACS state."""

    def __init__(self, name: str, known_variables: list[str]) -> None:
        self.message = f"Variable '{name}' is not present in the list of known variables: {', '.join(known_variables)}"
        super().__init__(self.message)


class ReadOnlyVariableError(QuacsError):
    """Raised when a function sets a variable it is only allowed to read."""

    def __init__(self, name: str, reason: str) -> None:
        self.message = f"Variable '{name}' cannot be set: {reason}"
        super().__init__(self.message)


class UnknownHookLabelError(QuacsError):
    """Raised when run_hook() is called with a label unknown to the host config."""

    def __init__(self, label: str) -> None:
        self.message = f"Hook label '{label}' is not handled by the host configuration."
        super().__init__(self.message)


class UnknownParameterizationError(QuacsError):
    """Raised when a parameterization does not exist in the registry."""

    def __init__(self, label: str) -> None:
        self.message = f"Parameterization '{label}' does not exist in the registry."
        super().__init__(self.message)


class VariableNotFoundError(QuacsError):
    """Raised when a parameterization requests a variable with `READ_ONLY` or `UPDATE` permissions with no canonical
    mapping and no direct host-name fallback."""

    def __init__(self, name: str, parameterization: str) -> None:
        self.message = f"Variable '{name}' requested by '{parameterization}' cannot be found in the set of host variables."
        super().__init__(self.message)


class HostConfigError(QuacsError):
    """Raised when a host-specific configuration model cannot be imported or does not conform
    to the QUACS API."""
