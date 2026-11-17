# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Profiles the toy ozone chemistry.

Run from the repository root with:
    uv run python examples/toy-model/profile_ozone.py
"""

from __future__ import annotations

from host_adapter import ToyHostAdapter
from parameterizations import OzoneChemistry

from quacs.profiler import profile

report = profile(
    OzoneChemistry(),
    ToyHostAdapter(),
    ranges={"ozone": (0.0, 1.0), "temperature": (200.0, 320.0)},
    samples=50,
)
print(report)