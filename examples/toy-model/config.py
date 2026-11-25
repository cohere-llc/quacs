# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The QUACS configuration of the toy model.

Defines the parameterizations and when to run them for a simulation.

This is the one file that is aware of the host model and parameterization details.

Hooks are embedded in the host application with string labels. Each hook in the host
application must be included in `hooks` even if no parameterizations are run for them.
Parameterizations for each hook are run in the order specified here.

`names` defines a mapping between variable names used in the configured set of
`parameterizations` and those used in the host application, when they differ.
If the host and parameterization variable names all match, `names` need not
be defined.
"""

from __future__ import annotations

import variables
from parameterizations import HeatingScheme, OzoneChemistry, SurfaceDeposition

parameterizations = [HeatingScheme(), OzoneChemistry(), SurfaceDeposition()]

hooks = {
    "before_dynamics": ["heating", "ozone_chemistry"],
    "after_dynamics": ["ozone_chemistry", "surface_deposition"],
}

names = {variables.temperature: "t_host"}
