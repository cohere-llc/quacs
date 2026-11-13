# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Toy Model driver.

Run from the repository root with:
    uv run python examples/toy-model/main.py
"""

from __future__ import annotations

from host_adapter import ToyHostAdapter
from parameterizations import HeatingScheme, OzoneChemistry, SurfaceDeposition

from quacs import Runtime

NUM_TIMESTEPS = 5

HOOKS = {
    "before_dynamics": ["heating", "ozone_chemistry"],
    "after_dynamics": ["ozone_chemistry", "surface_deposition"],
}

DIAGNOSTICS = ("t_host", "ozone", "surface_ozone_flux")


def run_dynamics(host: ToyHostAdapter) -> None:
    """The toy model's own work between the two QUACS hooks."""
    temperature = host.array("t_host")
    temperature *= 0.999


def main() -> None:
    host = ToyHostAdapter()
    runtime = Runtime(host, [HeatingScheme(), OzoneChemistry(), SurfaceDeposition()], HOOKS)
    host.allocate_registered_variables(runtime.register_variables())

    print(f"{'step':>4}  {'mean T (K)':>12}  {'mean O3':>12}  {'mean O3 flux':>14}")
    for step in range(1, NUM_TIMESTEPS + 1):
        runtime.run_hook("before_dynamics")
        run_dynamics(host)
        runtime.run_hook("after_dynamics")
        mean_t, mean_o3, mean_flux = (host.array(name).mean() for name in DIAGNOSTICS)
        print(f"{step:>4}  {mean_t:>12.4f}  {mean_o3:>12.6f}  {mean_flux:>14.6f}")


if __name__ == "__main__":
    main()
