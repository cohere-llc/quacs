# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The configuration file for `test_host.c`: runs every conformance check at hook "step"."""

from quacs import VariableDefinition
from quacs.conformance import VerticalOrderCheck, default_checks
from quacs.shapes import GridShape

height = VariableDefinition("height", GridShape.CELL, "float", "m", "Height of the grid cell")

parameterizations = [*default_checks(), VerticalOrderCheck(height, increases_upward=True)]
hooks = {"step": [check.name for check in parameterizations]}
names = {height: "z"}
