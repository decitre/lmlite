# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch


class TestRuntimeBridge:
    def test_js_eval_fallback_returns_none(self):
        from lmlite.bridge import runtime

        assert runtime.js_eval("1 + 1") is None

    def test_ensure_runtime_noop_when_already_loaded(self):
        from lmlite.bridge import runtime

        previous = runtime._runtime_loaded
        runtime._runtime_loaded = True
        try:
            with patch("importlib.resources.files", side_effect=AssertionError("should not be called")):
                runtime.ensure_runtime()
        finally:
            runtime._runtime_loaded = previous

    def test_ensure_runtime_loads_js_once(self):
        from lmlite.bridge import runtime

        previous = runtime._runtime_loaded
        runtime._runtime_loaded = False
        try:
            file_obj = MagicMock()
            file_obj.read_text.return_value = "globalThis.__runtime_test__ = true;"
            package_obj = MagicMock()
            package_obj.joinpath.return_value = file_obj

            with patch("importlib.resources.files", return_value=package_obj):
                with patch("lmlite.bridge.runtime.js_eval") as mock_js_eval:
                    runtime.ensure_runtime()

            mock_js_eval.assert_called_once_with("globalThis.__runtime_test__ = true;")
            assert runtime._runtime_loaded is True
        finally:
            runtime._runtime_loaded = previous
