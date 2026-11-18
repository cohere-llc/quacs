# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Toy Model driver.

Run from the repository root with:
    uv run python examples/toy-model/main.py
"""

from __future__ import annotations

import sys

from host_adapter import NCOLUMNS, NLEVELS, ToyHostAdapter
from parameterizations import HeatingScheme, OzoneChemistry, SurfaceDeposition

from quacs import Runtime

NUM_TIMESTEPS = 5
PROFILE_COLUMN = 0

HOOKS = {
    "before_dynamics": ["heating", "ozone_chemistry"],
    "after_dynamics": ["ozone_chemistry", "surface_deposition"],
}


def run_dynamics(host: ToyHostAdapter) -> None:
    """The toy model's own work between the two QUACS hooks."""
    temperature = host.array("t_host")
    temperature *= 0.999


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Print a titled table with the first column left-aligned and the rest right-aligned."""
    widths = [max(len(cell) for cell in column) for column in zip(headers, *rows)]
    print(f"\n{title}")
    for row in [headers, *rows]:
        cells = [row[0].ljust(widths[0])]
        cells += [cell.rjust(width) for cell, width in zip(row[1:], widths[1:])]
        print("  " + "  ".join(cells))


def domain_means(step: int, host: ToyHostAdapter) -> list[str]:
    return [
        str(step),
        f"{host.array('t_host').mean():.4f}",
        f"{host.array('ozone').mean():.6f}",
        f"{host.array('surface_ozone_flux').mean():.6f}",
    ]


def column_rows(host: ToyHostAdapter) -> list[list[str]]:
    return [
        [
            str(column + 1),
            f"{host.array('cos_solar_zenith_angle')[column]:.4f}",
            str(bool(host.array("is_daytime")[column])),
            f"{host.array('column_heating')[column]:.4f}",
            f"{host.array('surface_ozone_flux')[column]:.6f}",
            f"{host.array('model_top_ozone_flux')[column]:.6f}",
        ]
        for column in range(NCOLUMNS)
    ]


def profile_rows(host: ToyHostAdapter, column: int) -> list[list[str]]:
    """One row per interface and per grid cell, from the model top down."""
    rows = []
    for level in range(NLEVELS + 1):
        label = {0: "model top", NLEVELS: "surface"}.get(level, "interface")
        rows.append(
            [
                label,
                "",
                "",
                "",
                "",
                f"{host.array('actinic_flux')[column, level]:.6f}",
                f"{host.array('ozone_flux')[column, level]:.6f}",
            ]
        )
        if level < NLEVELS:
            rows.append(
                [
                    f"cell {level + 1}",
                    f"{host.array('t_host')[column, level]:.4f}",
                    f"{host.array('ozone')[column, level]:.6f}",
                    f"{host.array('absorbed_flux')[column, level]:.6f}",
                    str(host.array("chemistry_updates")[column, level]),
                    "",
                    "",
                ]
            )
    return rows


def main() -> None:
    host = ToyHostAdapter(compiled="--compiled" in sys.argv)
    runtime = Runtime(host, [HeatingScheme(), OzoneChemistry(), SurfaceDeposition()], HOOKS)
    host.allocate_registered_variables(runtime.register_variables())

    means = []
    for step in range(1, NUM_TIMESTEPS + 1):
        runtime.run_hook("before_dynamics")
        run_dynamics(host)
        runtime.run_hook("after_dynamics")
        means.append(domain_means(step, host))

    print(f"QUACS toy model: {NCOLUMNS} columns x {NLEVELS} levels, {NUM_TIMESTEPS} timesteps")
    print_table(
        "Domain means after each timestep",
        ["step", "T (K)", "O3", "surface O3 flux"],
        means,
    )
    print_table(
        "Each column after the last timestep",
        ["column", "cos(SZA)", "daytime", "heating (K)", "surface O3 flux", "model top O3 flux"],
        column_rows(host),
    )
    print_table(
        f"Column {PROFILE_COLUMN + 1} after the last timestep, from the model top down",
        ["", "T (K)", "O3", "absorbed flux", "updates", "actinic flux", "O3 flux"],
        profile_rows(host, PROFILE_COLUMN),
    )


if __name__ == "__main__":
    main()
