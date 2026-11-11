# Copyright (C) 2026 University of Arizona and University Corporation for Atmospheric Research
# SPDX-License-Identifier: Apache-2.0
"""Hook dispatch

Maps a string label (e.g., "step A") to an ordered list of parameterizations to run and
executes them.

The set of mappings between host hook labels and ordered lists of parameterizations allows
parameterizations to be run in specified orders at each point in the host application code
that allows transfer of control to QUACS.

If, for example, a host application runs the dispatcher at two points in a single timestep:
before and after physics, the hook labels could be "before physics" and "after physics".

As these mappings are host-specifiec, they are maintained as part of the host's QUACS
configuration.
"""

from __future__ import annotations

import itertools

from quacs.exceptions import UnknownHookLabelError, UnknownParameterizationError
from quacs.registry import ParameterizationRegistry
from quacs.state import QuacsState


class HookDispatcher:
    def __init__(
        self,
        registry: ParameterizationRegistry,
        hooks: dict[str, list[str]],
    ) -> None:
        """
        Args:
            registry: the populated ParameterizationRegistry to resolve names against
            hooks: mapping of hook label -> ordered list of parameterization names
                to run when that label fires.
        """
        for param in itertools.chain(hooks.values()):
            if param not in registry:
                raise UnknownParameterizationError(param)
        self._hooks = {
            hook_label: [registry[registry_label] for registry_label in hooks[hook_label]]
            for hook_label in hooks
        }

    def run(self, label: str, state: QuacsState) -> None:
        """Run every parameterization for `label`, in order."""
        if label not in self._hooks:
            raise UnknownHookLabelError(label)
        for param in self._hooks[label]:
            param.run(state)

    @property
    def labels(self) -> list[str]:
        """All hook labels with at least one parameterization configured."""
        return [label for label, params in self._hooks.items() if len(params) > 0]
