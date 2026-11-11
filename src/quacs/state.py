# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Host application variable accessor passed to each Parameterization.run().

Provides access to NumPy views of host variables.
"""

from __future__ import annotations

import numpy as np

from quacs.host_adapter import HostAdapter


class QuacsState:
    """Resolves variable names to array views during a model run."""

    def __init__(self, host_adapter: HostAdapter, name_map: dict[str, str | None]) -> None:
        self._host_adapter = host_adapter
        self._name_map = name_map
        self._cache: dict[str, np.array] = {}

    def get(self, name: str) -> np.array:
        """Resolve `name` to an array."""
        return self._host_adapter.get_array(self._name_map.get(name) or name)

    def __get_item__(self, name: str) -> np.ndarray:
        return self.get(name)
