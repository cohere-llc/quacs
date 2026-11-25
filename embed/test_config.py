# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""The configuration file for `test_host.c`: runs every conformance check at hook "step"."""

from quacs.conformance import VerticalOrderCheck, default_checks

parameterizations = [*default_checks(), VerticalOrderCheck("height", "m", increases_upward=True)]
hooks = {"step": [check.name for check in parameterizations]}
names = {"height": "z"}
