# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Variable definition and requests.

Variables are defined once for a set of parameterizations using `VariableDefinition`. Each
parameterization requests the variables it needs from the set of defined variables,
specifying the permissions it needs. The combination of variable definition and
requested permissions is stored in `VariableSpec` instances.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

from quacs.shapes import GridShape

DType = Literal["float", "int", "bool"]


@dataclass(frozen=True)
class VariableDefinition:
    """Describes a variable that a set of parameterizations uses.
    
    A set of parameterizations defines each of its variables once, typically in a
    dedicated module. The parameterizations select needed variables from the common
    set of defined variables.

    An example for defining a variable for ozone mixing ratio:
    ```
    ozone = VariableDefinition(
        "ozone",
        GridShape.CELL,
        "float",
        "mol mol-1",
        "Ozone mixing ratio"
    )
    ```

    `name` is the name used in the QUACS variable definition. It may or may not match
    the name used by the host application.

    `dtype` is the type of the variable's values as parameterizations see them: Python
    `float`, `int` or `bool`. The host application chooses the precision it uses to store
    `float` and `int` values.

    `transported` asks the host application to move the variable horizontally with the
    modelled air movement, as it does for its own tracers. Not all models (e.g., box or
    column models) support this, so it is not a guarantee of transport. Only `float`
    variables with one value per grid cell can be transported.
    """

    name: str
    shape: GridShape
    dtype: DType
    units: str
    description: str
    transported: bool = False

    def __post_init__(self) -> None:
        if self.transported and (self.shape is not GridShape.CELL or self.dtype != "float"):
            raise ValueError(
                f"Variable '{self.name}' cannot be transported. Only float variables with "
                f"one value per grid cell are eligible."
            )

    def read_only(self) -> VariableSpec:
        """Requests the variable for a parameterization that only reads it."""
        return self._request(Permissions.READ_ONLY)

    def update(self) -> VariableSpec:
        """Requests a variable for a parameterization that adjusts its existing value."""
        return self._request(Permissions.UPDATE)

    def own(self) -> VariableSpec:
        """Requests the variable for a parameterization that is responsible for setting
        its value.
        
        The host application creates the variable.
        """
        return self._request(Permissions.OWN)

    def _request(self, permissions: Permissions) -> VariableSpec:
        return VariableSpec(
            self.name,
            self.shape,
            self.dtype,
            self.units,
            self.description,
            permissions,
            transported=self.transported and permissions is Permissions.OWN,
        )


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

    An instance of `VariableSpec` includes enough information for a host application to
    locate or add a state variable requested by a parameterization. It is usually made
    from a `VariableDefinition` using `read_only()`, `update()`, or `own()`.

    `name`, `shape`, `dtype`, `units`, `description`, and `transported` are as for
    `VariableDefinition`.
    """

    name: str
    shape: GridShape
    dtype: DType
    units: str
    description: str
    permissions: Permissions
    requested_by: str | None = None
    transported: bool = False


def host_name_map(names: Mapping[VariableDefinition, str] | None) -> dict[str, str]:
    """Returns a host application's names for variable, keyed with the names the
    parameterizations use.
    
    `names` gives a mapping from the variable names used by QUACS parameterizations
    to the host variable names, when they differ:

    ```
    names = {variables.temperature: "t_host"}
    ```

    Raises `TypeError` if a key is not a `VariableDefinition` or a host name is not
    a string, and `ValueError` if a variable is given two different host names.
    """
    host_names: dict[str, str] = {}
    if names is None:
        return host_names
    if not isinstance(names, Mapping):
        raise TypeError(f"`names` must be a dict, not {type(names).__name__}.")
    for definition, host_name in names.items():
        if not isinstance(definition, VariableDefinition) \
            or not isinstance(host_name, str):
            raise TypeError(
                f"`names` must give the host application's name for a variable "
                f"as a string, for a QUACS VariableDefinition. It has "
                f"{definition!r}: {host_name!r}"
            )
        if host_names.setdefault(definition.name, host_name) != host_name:
            raise ValueError(
                f"Variable '{definition.name}' is mapped to two names in the host "
                f"application: '{host_names[definition.name]}' and '{host_name}'."
            )
    return host_names
