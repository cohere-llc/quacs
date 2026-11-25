# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Variables used by the toy parameterization set.

Each variable is defined once, here. The set of toy parameterizations request variables
they use from this module. Mapping to host application variable names is set in the QUACS
configuration file, when needed.
"""

from __future__ import annotations

from quacs import VariableDefinition
from quacs.shapes import GridShape

temperature = VariableDefinition(
    "temperature",
    GridShape.CELL,
    "float",
    "K",
    "Air temperature",
)
ozone = VariableDefinition(
    "ozone",
    GridShape.CELL,
    "float",
    "mol mol-1",
    "Ozone mixing ratio",
)
absorbed_flux = VariableDefinition(
    "absorbed_flux",
    GridShape.CELL,
    "float",
    "1",
    "Actinic flux absorbed in the grid cell",
)
chemistry_updates = VariableDefinition(
    "chemistry_updates",
    GridShape.CELL,
    "int",
    "1",
    "Number of times the chemistry has updated the grid cell",
)
actinic_flux = VariableDefinition(
    "actinic_flux",
    GridShape.INTERFACE,
    "float",
    "1",
    "Actinic flux relative to the flux at the model top",
)
ozone_flux = VariableDefinition(
    "ozone_flux",
    GridShape.INTERFACE,
    "float",
    "mol m-2 s-1",
    "Downward ozone flux",
)
cos_solar_zenith_angle = VariableDefinition(
    "cos_solar_zenith_angle",
    GridShape.COLUMN,
    "float",
    "1",
    "Cosine of the solar zenith angle",
)
column_heating = VariableDefinition(
    "column_heating",
    GridShape.COLUMN,
    "float",
    "K",
    "Temperature increase summed over the grid cells of the column",
)
is_daytime = VariableDefinition(
    "is_daytime",
    GridShape.COLUMN,
    "bool",
    "1",
    "Whether the sun is above the horizon",
)
surface_wind_speed = VariableDefinition(
    "surface_wind_speed",
    GridShape.SURFACE_CELL,
    "float",
    "m s-1",
    "10 m wind speed",
)
surface_ozone_flux = VariableDefinition(
    "surface_ozone_flux",
    GridShape.SURFACE_CELL,
    "float",
    "mol m-2 s-1",
    "Downward ozone deposition flux at the surface",
)
model_top_ozone_flux = VariableDefinition(
    "model_top_ozone_flux",
    GridShape.MODEL_TOP_CELL,
    "float",
    "mol m-2 s-1",
    "Downward ozone flux at the model top",
)
time_step = VariableDefinition(
    "time_step",
    GridShape.DOMAIN,
    "float",
    "s",
    "Model time step"
)
