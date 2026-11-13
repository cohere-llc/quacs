# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""QUACS Scripting Interface for Chemistry and Aerosols."""

from . import exceptions, shapes
from .hooks import HookDispatcher
from .host_adapter import HostAdapter
from .parameterization import Parameterization
from .registry import ParameterizationRegistry
from .runtime import Runtime
from .state import QuacsState
from .variables import VariableSpec

__version__ = "0.1.0"

__all__ = [
    "HookDispatcher",
    "HostAdapter",
    "Parameterization",
    "ParameterizationRegistry",
    "QuacsState",
    "Runtime",
    "VariableSpec",
    "exceptions",
    "shapes",
]
