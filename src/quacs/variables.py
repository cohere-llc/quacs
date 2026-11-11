# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Variable specification.

Parameterizations register and request variables from a host application using the
`VariableSpec` class.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

from quacs.shapes import GridShape

DType = Literal["float", "int", "bool", "char"]


class Permissions(Enum):
    """Variable permissions requested by a parameterization.

    `OWN` should be used when a parameterization expects to be responsible for setting the
    value of the variable.

    `UPDATE` should be used when a parameterization
    uses the existing value of a variable in calculating its new value (e.g., raising the
    temperature by 1 degree).

    When a parameterization requests to `OWN` a variable that already exists in the host
    application and error is thrown.

    When more than one paramterization requests to `OWN` the same variable, an error is
    thrown.

    When a parameterization requests `READ_ONLY` or `UPDATE` for a variable that does
    not exist in the host application and has not been created with `OWN` permissions by
    another parameterization an error is thrown.
    """

    READ_ONLY = auto()
    OWN = auto()
    UPDATE = auto()


@dataclass(frozen=True)
class VariableSpec:
    """Describes one state variable.

    An instance of `VariableSpec` includes enough information for a host model to locate or
    add a state variable requested by a parameterization.

    `requested_by` will be set by the registry using the requesting parameterization name,
    so it may be left unset by the requesting parameterization.

    `name` is the name of the variable used by the parameterization. When it exists in the
    list of canonical QUACS variable names, the host application configuration will include a
    mapping to its internal namespace (if the variable is available). When it is not a
    known QUACS name, a direct match to a host application variable will be attempted, unless
    the parameterization is decorated with a host-application-specific name mapping.
    """

    name: str
    shape: GridShape
    dtype: DType
    units: str
    description: str
    permissions: Permissions
    requested_by: str | None = None
