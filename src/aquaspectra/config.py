"""Configuration loading and lightweight validation."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]
    root: str

    # ------------------------------------------------------------------ helpers
    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node

    def path(self, *keys: str, default: Any = None) -> str | None:
        """Resolve a config value that is a path, relative to the config root."""
        val = self.get(*keys, default=default)
        if val is None:
            return None
        if os.path.isabs(val):
            return val
        return os.path.normpath(os.path.join(self.root, val))


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    root = os.path.dirname(os.path.abspath(path))
    return Config(raw=raw, root=root)
