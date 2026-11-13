# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Three toy parameterizations.

Note that these parameterizations make no assumptions about how the host state is organized.
"""

from __future__ import annotations

import math

from quacs import QuacsState, VariableSpec
from quacs.accessors import Column, Variable
from quacs.shapes import GridShape
from quacs.variables import Permissions


class HeatingScheme:
    """Raises `temperature` in every grid cell, in proportion to the sunlight on its column,
    and records the net heating of each column."""

    name = "heating"

    def __init__(self, rate_per_call: float = 0.5) -> None:
        self._rate_per_call = rate_per_call

    def register_variables(self) -> list[VariableSpec]:
        return [
            VariableSpec(
                name="temperature",
                shape=GridShape.CELL,
                dtype="float",
                units="K",
                description="Air temperature",
                permissions=Permissions.UPDATE,
            ),
            VariableSpec(
                name="cos_solar_zenith_angle",
                shape=GridShape.COLUMN,
                dtype="float",
                units="1",
                description="Cosine of the solar zenith angle",
                permissions=Permissions.READ_ONLY,
            ),
            VariableSpec(
                name="column_heating",
                shape=GridShape.COLUMN,
                dtype="float",
                units="K",
                description="Temperature increase summed over the grid cells of the column",
                permissions=Permissions.OWN,
            ),
        ]

    def run(self, state: QuacsState) -> None:
        rate = self._rate_per_call

        def heat(
            column: Column,
            temperature: Variable,
            cos_solar_zenith_angle: Variable,
            column_heating: Variable,
        ) -> None:
            net_heating = 0.0
            while True:
                heating = rate * cos_solar_zenith_angle.get()
                temperature.set(temperature.get() + heating)
                net_heating += heating
                if column.at_surface():
                    break
                column.step_toward_surface()
            column_heating.set(net_heating)

        state.for_each_column(
            heat, "temperature", "cos_solar_zenith_angle", "column_heating", start="model_top"
        )


class OzoneChemistry:
    """Produces and removes `ozone`, working down each column from the model top.

    Production is driven by an actinic flux that is attenuated by the ozone in each cell
    it passes through, so each cell depends on the cells above it.
    """

    name = "ozone_chemistry"

    def __init__(
        self,
        production_rate: float = 0.02,
        loss_fraction: float = 0.05,
        attenuation: float = 0.5,
    ) -> None:
        self._production_rate = production_rate
        self._loss_fraction = loss_fraction
        self._attenuation = attenuation

    def register_variables(self) -> list[VariableSpec]:
        return [
            VariableSpec(
                name="ozone",
                shape=GridShape.CELL,
                dtype="float",
                units="mol mol-1",
                description="Ozone mixing ratio",
                permissions=Permissions.OWN,
            ),
            VariableSpec(
                name="temperature",
                shape=GridShape.CELL,
                dtype="float",
                units="K",
                description="Air temperature",
                permissions=Permissions.READ_ONLY,
            ),
        ]

    def run(self, state: QuacsState) -> None:
        production_rate = self._production_rate
        loss_fraction = self._loss_fraction
        attenuation = self._attenuation

        def react(column: Column, ozone: Variable, temperature: Variable) -> None:
            actinic_flux = 1.0
            while True:
                current = ozone.get()
                actinic_flux = actinic_flux * math.exp(-attenuation * current)
                production = production_rate * actinic_flux * temperature.get() / 290.0
                ozone.set((current + production) * (1.0 - loss_fraction))
                if column.at_surface():
                    break
                column.step_toward_surface()

        state.for_each_column(react, "ozone", "temperature", start="model_top")


class SurfaceDeposition:
    """Calculates `surface_ozone_flux` from ozone in the cell nearest the surface."""

    name = "surface_deposition"

    def __init__(self, deposition_velocity: float = 0.01) -> None:
        self._deposition_velocity = deposition_velocity

    def register_variables(self) -> list[VariableSpec]:
        return [
            VariableSpec(
                name="surface_ozone_flux",
                shape=GridShape.SURFACE_CELL,
                dtype="float",
                units="mol m-2 s-1",
                description="Downward ozone deposition flux at the surface",
                permissions=Permissions.OWN,
            ),
            VariableSpec(
                name="ozone",
                shape=GridShape.CELL,
                dtype="float",
                units="mol mol-1",
                description="Ozone mixing ratio",
                permissions=Permissions.READ_ONLY,
            ),
            VariableSpec(
                name="surface_wind_speed",
                shape=GridShape.SURFACE_CELL,
                dtype="float",
                units="m s-1",
                description="10 m wind speed",
                permissions=Permissions.READ_ONLY,
            ),
        ]

    def run(self, state: QuacsState) -> None:
        deposition_velocity = self._deposition_velocity

        def deposit(column: Column, ozone: Variable, wind_speed: Variable, flux: Variable) -> None:
            flux.set(deposition_velocity * (1.0 + 0.1 * wind_speed.get()) * ozone.get())

        state.for_each_column(
            deposit, "ozone", "surface_wind_speed", "surface_ozone_flux", start="surface"
        )
