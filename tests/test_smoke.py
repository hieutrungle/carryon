"""Repo-level guards that must hold from M0 on: the package imports and every config parses."""

import importlib
from pathlib import Path

import pytest
import yaml

import carryon

REPO = Path(__file__).resolve().parents[1]
SUBPACKAGES = ["envs", "context", "agent", "rollout", "train", "eval", "viz"]
CONFIGS = sorted((REPO / "configs").rglob("*.yaml"))


def test_version() -> None:
    assert carryon.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    importlib.import_module(f"carryon.{name}")


@pytest.mark.parametrize("path", CONFIGS, ids=[str(p.relative_to(REPO)) for p in CONFIGS])
def test_config_parses(path: Path) -> None:
    assert isinstance(yaml.safe_load(path.read_text()), dict)
