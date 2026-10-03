"""Configuration loading, project paths, seeding and small I/O helpers."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML config; a top-level `extends: other.yaml` is merged underneath."""
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _merge(load_config(path.parent / parent), cfg)
    cfg["_source"] = str(path)
    return cfg


def get_path(cfg: dict, key: str, mkdir: bool = True) -> Path:
    p = Path(cfg["paths"][key])
    if not p.is_absolute():
        p = ROOT / p
    if mkdir:
        p.mkdir(parents=True, exist_ok=True)
    return p


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dir_size_bytes(path: str | Path) -> int:
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def cli_config(description: str, extra=None) -> tuple[dict, "argparse.Namespace"]:
    """Parse --config (plus optional extra arguments) and load the config."""
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML config path")
    if extra:
        extra(parser)
    args = parser.parse_args()
    return load_config(args.config), args
