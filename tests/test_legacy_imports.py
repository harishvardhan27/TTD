"""Regression test for audit M6: every legacy script's imports must resolve
(no running of the scripts themselves, which download weights / train)."""
import ast
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LEGACY = sorted(p for d in ("CNN", "DistilBERT", "FNN", "HO", "THEORY_VERIFY", "VIT") for p in (ROOT / d).glob("*.py"))
OPTIONAL = {"transformers", "datasets", "evaluate", "tensorly", "gensim", "setuptools"}


@pytest.mark.parametrize("path", LEGACY, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_legacy_imports_resolve(path, monkeypatch):
    monkeypatch.syspath_prepend(str(path.parent))
    if path.parent.name == "HO":
        monkeypatch.syspath_prepend(str(ROOT / "FNN"))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mod = node.module
        elif isinstance(node, ast.Import):
            mod = node.names[0].name
        else:
            continue
        if mod.split(".")[0] in OPTIONAL:
            continue
        assert importlib.util.find_spec(mod) is not None, f"{path.name}: cannot resolve '{mod}'"
