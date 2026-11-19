# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Runs the QUACS conformance checks on the toy host.

Run from the repository root with:
    uv run python examples/toy-model/check_host.py [--compiled]
"""

from __future__ import annotations

import sys

from host_adapter import ToyHostAdapter

from quacs.conformance import run_conformance


def main() -> None:
    report = run_conformance(ToyHostAdapter(compiled="--compiled" in sys.argv))
    print(report)
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
