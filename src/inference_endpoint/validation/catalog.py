# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The ruleset's catalog: model and dataset values as plain dictionaries, nothing more.

`Catalog.from_ruleset(name)` is the checker's loader: it derives the mappings
from the packaged ruleset the cohort names (`config/ruleset_registry.py`,
e.g. `mlperf-inference-v6.1`), so the checker and the benchmark read one
source. `Catalog.from_directory(root)` loads a development
`models.yaml` + `datasets.yaml` pair instead. Values stay raw data, not
pseudo-enum classes; a check addresses them through the `ruleset` namespace of
`Ref` (`ruleset.models.deepseek-r1.dataset.size`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from inference_endpoint.config.ruleset_registry import get_ruleset
from inference_endpoint.config.rulesets.mlcommons.rules import RoundRuleset
from inference_endpoint.validation.ranges import Interval
from inference_endpoint.validation.tree import StrictLoader


class Catalog:
    """`models` and `datasets` mappings of a ruleset directory."""

    __slots__ = ("datasets", "models")

    def __init__(self, models: dict[str, Any], datasets: dict[str, Any]) -> None:
        self.models = models
        self.datasets = datasets

    @classmethod
    def from_ruleset(cls, name: str) -> Catalog:
        """The catalog the registered ruleset `name` carries.

        The checker reads packaged rulesets only: a cohort's manifest names
        one, and this view exposes its plain data -- one source with the
        benchmark side. Datasets key on their `description` (the Python
        `_Dataset` carries no short name); a round with no per-model rules
        (e.g. an Endpoints seed set) yields empty mappings.

        Args:
            name (str): A registered ruleset name (`config/ruleset_registry.py`).
        Returns:
            Catalog: The ruleset's `models` and `datasets` mappings, values as
                plain dictionaries.
        Raises:
            ValueError: When `name` is not registered, or the registered
                ruleset is not a round of per-model rules.
        """
        try:
            ruleset = get_ruleset(name)
        except KeyError as exc:
            raise ValueError(exc.args[0]) from None
        if not isinstance(ruleset, RoundRuleset):
            raise ValueError(
                f"ruleset {name!r} carries no model rounds: expected a RoundRuleset"
            )
        models: dict[str, Any] = {}
        datasets: dict[str, Any] = {}
        for model in ruleset.benchmark_rulesets:
            models[model.name] = asdict(model)
            datasets[model.dataset.description] = asdict(model.dataset)
        return cls(models, datasets)

    @classmethod
    def from_directory(cls, root: Path) -> Catalog:
        """Load `models.yaml` and `datasets.yaml` under `root`.

        Args:
            root (Path): The ruleset directory holding both catalog files.
        Returns:
            Catalog: The loaded `models` and `datasets` mappings.
        Raises:
            NotADirectoryError: When `root` is missing or is not a directory: the
                message names which and the path.
            FileNotFoundError: When `models.yaml` or `datasets.yaml` is missing:
                the ruleset is incomplete without its catalog.
            ValueError: When a document does not parse or is not a mapping: a
                load error, refused like every other bad cohort input.
        """
        if not root.exists():
            raise NotADirectoryError(f"the catalog directory {root} does not exist")
        if not root.is_dir():
            raise NotADirectoryError(f"the catalog path {root} is not a directory")
        return cls(
            _load_mapping(root / "models.yaml"),
            _load_mapping(root / "datasets.yaml"),
        )

    def model_entry(self, name: Any) -> Mapping[str, Any] | None:
        """The catalog entry for model `name`, or None when there is none.

        Args:
            name (Any): A declared `model_name` (anything not a string has none).
        Returns:
            Mapping[str, Any] | None: The entry, e.g. `models.yaml`'s
                `wan-2.2-t2v-a14b` block; None for an unknown or non-string name.
        """
        entry = self.models.get(name) if isinstance(name, str) else None
        return entry if isinstance(entry, Mapping) else None

    def frames_per_sample(self, name: Any) -> float | None:
        """Frames in one sample of model `name` when the catalog says it is video.

        The one place the checker decides "video or tokens": a model whose
        entry carries a numeric `frames_per_sample` answers in frames, anything
        else (no entry, no key, a non-numeric value) in tokens.

        Args:
            name (Any): A declared `model_name`.
        Returns:
            float | None: 81 for `wan-2.2-t2v-a14b`, None for a token model.
        """
        entry = self.model_entry(name)
        frames = entry.get("frames_per_sample") if entry is not None else None
        if isinstance(frames, int | float) and not isinstance(frames, bool):
            return frames
        return None

    def accuracy_gates(self, name: Any) -> dict[str, float]:
        """Each accuracy set's gate for model `name`: `factor x lower` of its `targets`.

        Only a set's own `targets` in the `{ lower, factor }` form gate here (the
        set's score is then one number to compare). Bands, means over several
        sets and agentic targets are left to their own rows.

        Args:
            name (Any): A declared `model_name`.
        Returns:
            dict[str, float]: Set name to its gate, e.g.
                `{"wan22_vbench": 69.7752}`; empty when no set declares one.
        """
        entry = self.model_entry(name)
        sets = entry.get("accuracy_datasets") if entry is not None else None
        if not isinstance(sets, Mapping):
            return {}
        gates: dict[str, float] = {}
        for set_name, spec in sets.items():
            targets = spec.get("targets") if isinstance(spec, Mapping) else None
            if not isinstance(targets, Mapping) or len(targets) != 1:
                continue
            (target,) = targets.values()
            gate = _lower_gate(target)
            if gate is not None:
                gates[set_name] = gate
        return gates

    def scopes(self) -> dict[str, dict[str, Any]]:
        """What the `ruleset` namespace of `Ref.resolve` answers, for `Node.scopes`."""
        return {"models": self.models, "datasets": self.datasets}


def _lower_gate(target: Any) -> float | None:
    """`factor x lower` of a `{ lower, factor }` target, or None for any other form."""
    if not isinstance(target, Mapping) or set(target) != {"lower", "factor"}:
        return None
    lower, factor = target["lower"], target["factor"]
    numbers = (lower, factor)
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in numbers):
        return None
    return Interval.for_accuracy_gate(lower, factor).lo


def _load_mapping(path: Path) -> dict[str, Any]:
    """That file's top-level mapping. A malformed document is a load error.

    Raises:
        ValueError: When the document does not parse (a YAML error is a
            refusal, not a traceback escaping the CLI) or does not hold a
            mapping.
    """
    text = path.read_text(encoding="utf-8")
    try:
        value = yaml.load(text, Loader=StrictLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path.name} is not a catalog: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must hold a mapping, got {type(value).__name__}")
    return value
