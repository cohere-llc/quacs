# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Three toy parameterizations.

Note that these parameterizations make no assumptions about how the host state is organized.
"""

from __future__ import annotations

import math

from quacs import QuacsState, VariableSpec
from quacs.accessors import Column, InterfaceVariable, Variable, above, below, lower, upper
from quacs.shapes import GridShape
from quacs.variables import Permissions


class HeatingScheme:
    """Raises `temperature` in every grid cell, in proportion to the sunlight on its column,
    and records the net heating of each column."""

    name = "heating"

    def __init__(self, heating_rate: float = 3.0) -> None:
        """`heating_rate` is the temperature increase under an overhead sun, in K per hour."""
        self._heating_rate = heating_rate

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
            VariableSpec(
                name="is_daytime",
                shape=GridShape.COLUMN,
                dtype="bool",
                units="1",
                description="Whether the sun is above the horizon",
                permissions=Permissions.OWN,
            ),
            VariableSpec(
                name="time_step",
                shape=GridShape.DOMAIN,
                dtype="float",
                units="s",
                description="Model time step",
                permissions=Permissions.READ_ONLY,
            ),
        ]

    def run(self, state: QuacsState) -> None:
        def heat(
            temperature: Variable,
            cos_solar_zenith_angle: Variable,
            time_step: float,
            heating_rate: float
        ) -> None:
            heating = heating_rate * time_step / 3600.0 * cos_solar_zenith_angle.get()
            temperature.set(temperature.get() + heating)

        def total_column_heating(
            column: Column,
            cos_solar_zenith_angle: Variable,
            column_heating: Variable,
            is_daytime: Variable[bool],
            time_step: float,
            heating_rate: float,
        ) -> None:
            net_heating = 0.0
            while True:
                net_heating += heating_rate * time_step / 3600.0 * cos_solar_zenith_angle.get()
                if column.at_model_top():
                    break
                column.step_toward_model_top()
            column_heating.set(net_heating)
            is_daytime.set(cos_solar_zenith_angle.get() > 0.0)

        state.for_each_cell(
            heat,
            "temperature",
            "cos_solar_zenith_angle",
            "time_step",
            self._heating_rate,
        )
        state.for_each_column(
            total_column_heating,
            "cos_solar_zenith_angle",
            "column_heating",
            "is_daytime",
            "time_step",
            self._heating_rate,
            start="surface",
        )


class OzoneChemistry:
    """Produces and removes `ozone` in every grid cell.

    Production is driven by an actinic flux that is attenuated by the ozone in each cell
    it passes through. The flux is calculated at the interfaces between cells, working
    down each column from the model top.
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
                name="actinic_flux",
                shape=GridShape.INTERFACE,
                dtype="float",
                units="1",
                description="Actinic flux relative to the flux at the model top",
                permissions=Permissions.OWN,
            ),
            VariableSpec(
                name="absorbed_flux",
                shape=GridShape.CELL,
                dtype="float",
                units="1",
                description="Actinic flux absorbed in the grid cell",
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
            VariableSpec(
                name="chemistry_updates",
                shape=GridShape.CELL,
                dtype="int",
                units="1",
                description="Number of times chemistry has updated the grid cell",
                permissions=Permissions.OWN,
            ),
            VariableSpec(
                name="is_daytime",
                shape=GridShape.COLUMN,
                dtype="bool",
                units="1",
                description="Whether the sun is above the horizon",
                permissions=Permissions.READ_ONLY,
            ),
        ]

    def run(self, state: QuacsState) -> None:
        production_rate = self._production_rate
        loss_fraction = self._loss_fraction
        attenuation = self._attenuation

        def attenuate(
            column: Column,
            ozone: Variable,
            is_daytime: Variable[bool],
            upper_flux: InterfaceVariable,
            lower_flux: InterfaceVariable,
        ) -> None:
            upper_flux.set(1.0 if is_daytime.get() else 0.0)
            while True:
                lower_flux.set(upper_flux.get() * math.exp(-attenuation * ozone.get()))
                if column.at_surface():
                    break
                column.step_toward_surface()

        def react(
            ozone: Variable,
            temperature: Variable,
            absorbed_flux: Variable,
            chemistry_updates: Variable[int],
            upper_flux: InterfaceVariable,
            lower_flux: InterfaceVariable,
        ) -> None:
            current = ozone.get()
            production = production_rate * lower_flux.get() * temperature.get() / 290.0
            ozone.set((current + production) * (1.0 - loss_fraction))
            absorbed_flux.set(upper_flux.get() - lower_flux.get())
            chemistry_updates.set(chemistry_updates.get() + 1)

        state.for_each_column(
            attenuate,
            "ozone",
            "is_daytime",
            upper("actinic_flux"),
            lower("actinic_flux"),
            start="model_top",
        )
        state.for_each_cell(
            react,
            "ozone",
            "temperature",
            "absorbed_flux",
            "chemistry_updates",
            upper("actinic_flux"),
            lower("actinic_flux"),
        )


class SurfaceDeposition:
    """Calculates the downward flux of ozone through every interface: mixing between grid
    cells, and deposition at the surface."""

    name = "surface_deposition"

    def __init__(self, deposition_velocity: float = 0.01, mixing_rate: float = 0.1) -> None:
        self._deposition_velocity = deposition_velocity
        self._mixing_rate = mixing_rate

    def register_variables(self) -> list[VariableSpec]:
        return [
            VariableSpec(
                name="ozone_flux",
                shape=GridShape.INTERFACE,
                dtype="float",
                units="mol m-2 s-1",
                description="Downward ozone flux",
                permissions=Permissions.OWN,
            ),
            VariableSpec(
                name="surface_ozone_flux",
                shape=GridShape.SURFACE_CELL,
                dtype="float",
                units="mol m-2 s-1",
                description="Downward ozone deposition flux at the surface",
                permissions=Permissions.OWN,
            ),
            VariableSpec(
                name="model_top_ozone_flux",
                shape=GridShape.MODEL_TOP_CELL,
                dtype="float",
                units="mol m-2 s-1",
                description="Downward ozone flux at the model top",
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
        mixing_rate = self._mixing_rate

        def calculate_flux(
            ozone_flux: InterfaceVariable,
            ozone_above: Variable | None,
            ozone_below: Variable | None,
            wind_speed: Variable,
        ) -> None:
            if ozone_above is None:
                # Nothing enters through the model top.
                ozone_flux.set(0.0)
            elif ozone_below is None:
                # Deposition at the surface.
                ozone_flux.set(
                    deposition_velocity * (1.0 + 0.1 * wind_speed.get()) * ozone_above.get()
                )
            else:
                ozone_flux.set(mixing_rate * (ozone_above.get() - ozone_below.get()))

        def record_boundaries(
            column: Column,
            upper_flux: InterfaceVariable,
            lower_flux: InterfaceVariable,
            surface_flux: Variable,
            model_top_flux: Variable,
        ) -> None:
            column.select_model_top()
            model_top_flux.set(upper_flux.get())
            column.select_surface_cell()
            surface_flux.set(lower_flux.get())

        state.for_each_interface(
            calculate_flux, "ozone_flux", above("ozone"), below("ozone"), "surface_wind_speed"
        )
        state.for_each_column(
            record_boundaries,
            upper("ozone_flux"),
            lower("ozone_flux"),
            "surface_ozone_flux",
            "model_top_ozone_flux",
        )
