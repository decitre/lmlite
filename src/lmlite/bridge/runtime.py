# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

try:
    from js import eval as js_eval
except ImportError:

    def js_eval(x):
        return None


_runtime_loaded = False


def ensure_runtime():
    global _runtime_loaded

    if _runtime_loaded:
        return

    from importlib.resources import files

    js_code = files("lmlite").joinpath("lmlite.js").read_text()

    js_eval(js_code)

    _runtime_loaded = True
