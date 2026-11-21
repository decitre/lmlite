import sys
import types

import pytest


@pytest.fixture(autouse=True)
def fake_js_bridge(monkeypatch):
    """Identity ``pyodide.ffi.to_js``, so code under test has a Python-to-JavaScript bridge."""
    ffi = types.ModuleType("pyodide.ffi")
    ffi.to_js = lambda value, **kwargs: value
    monkeypatch.setitem(sys.modules, "pyodide", types.ModuleType("pyodide"))
    monkeypatch.setitem(sys.modules, "pyodide.ffi", ffi)
