# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Profiles the toy ozone chemistry.

Run from the repository root with:
    uv run python examples/toy-model/profile_ozone.py
"""

from __future__ import annotations

import config
import numpy as np
from host_adapter import ToyHostAdapter
from parameterizations import OzoneChemistry

from quacs.profiler import profile

NCOLUMNS = 500
NLEVELS = 20
RANGES = {"ozone": (0.0, 1.0), "temperature": (200.0, 320.0)}

interpreted = profile(
    OzoneChemistry(),
    ToyHostAdapter(NCOLUMNS, NLEVELS, compiled=False),
    RANGES,
    names=config.names,
)
compiled = profile(
    OzoneChemistry(),
    ToyHostAdapter(NCOLUMNS, NLEVELS),
    RANGES,
    names=config.names,
)

print("Interpreted")
print(interpreted)
print("\nCompiled")
print(compiled)

for name, means in interpreted.ending_means.items():
    np.testing.assert_allclose(compiled.ending_means[name], means, rtol=1e-12)
print(
    f"\nThe two runs give the same results. Compiled is "
    f"{interpreted.mean / compiled.mean:.1f} times as fast as interpreted."
)
