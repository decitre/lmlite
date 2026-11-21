# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""OnnxModel: the native path on a real tiny model, and the browser path against a fake JavaScript handle."""

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from lmlite import OnnxModel

np = pytest.importorskip("numpy")
FIXTURE = Path(__file__).parent / "fixtures/onnx/scale_add.onnx"  # y = x * 2 + b, x: (batch, 3), b: (3,)
needs_native = pytest.mark.skipif(not importlib.util.find_spec("onnxruntime"), reason="needs lmlite[native]")
B = np.array([0, 1, 2], dtype=np.float32)


@needs_native
class TestNative:
    @pytest.fixture(autouse=True)
    def no_browser(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "js", raising=False)

    async def test_loads_a_path_describes_it_and_runs(self):
        model = await OnnxModel.load(str(FIXTURE), licence="CC0")

        assert model.inputs == [
            {"name": "x", "dtype": "float32", "shape": ["batch", 3]},
            {"name": "b", "dtype": "float32", "shape": [3]},
        ]
        assert model.outputs == [{"name": "y", "dtype": "float32", "shape": ["batch", 3]}]
        info = model.model_info()
        assert info["sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        assert info["backend"] == "native" and info["execution_provider"] == "CPUExecutionProvider"
        assert info["licence"] == "CC0"
        for rows in (1, 4):  # dynamic batch dimension
            out = await model.run({"x": np.ones((rows, 3), np.float32), "b": B})
            assert out["y"].shape == (rows, 3) and (out["y"] == [2, 3, 4]).all()
        await model.dispose()

    async def test_loads_bytes(self):
        model = await OnnxModel.load(FIXTURE.read_bytes())
        assert [i["name"] for i in model.inputs] == ["x", "b"]

    async def test_sha256_is_checked_for_bytes_and_paths(self):
        digest = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()

        assert (await OnnxModel.load(str(FIXTURE), sha256=digest.upper())).model_info()["sha256"] == digest
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            await OnnxModel.load(FIXTURE.read_bytes(), sha256="0" * 64)

    async def test_float16_runs(self):
        model = await OnnxModel.load(str(FIXTURE.with_name("scale_add_fp16.onnx")))

        out = await model.run({"x": np.ones((1, 3), np.float16), "b": B.astype(np.float16)})
        assert out["y"].dtype == np.float16 and (out["y"] == [2, 3, 4]).all()

    async def test_hosted_ids_need_a_browser(self):
        with pytest.raises(FileNotFoundError, match="Hosted IDs resolve in a browser kernel"):
            await OnnxModel.load("lmlite/scale-add")

    async def test_missing_file_is_reported(self):
        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            await OnnxModel.load("missing.onnx")

    async def test_errors_name_the_input(self):
        model = await OnnxModel.load(str(FIXTURE))
        x = np.ones((1, 3), np.float32)

        with pytest.raises(ValueError, match="Missing input 'b'"):
            await model.run({"x": x})
        with pytest.raises(ValueError, match="Unknown input 'z'"):
            await model.run({"x": x, "b": B, "z": B})
        with pytest.raises(TypeError, match="Input 'x' must be float32, got float64"):
            await model.run({"x": x.astype(np.float64), "b": B})
        with pytest.raises(ValueError, match="Input 'x' must have rank 2"):
            await model.run({"x": np.ones(3, np.float32), "b": B})
        with pytest.raises(ValueError, match="Input 'b' must have size 3 on axis 0"):
            await model.run({"x": x, "b": B[:2]})


class TestBrowser:
    """The bridge contract: one JSON description and one buffer per input out, one buffer per output back."""

    @pytest.fixture
    def js(self, monkeypatch):
        handle = MagicMock()
        handle.describe.return_value = json.dumps(
            {
                "backend": "wasm",
                "execution_provider": "wasm",
                "sha256": "abc",
                "size": 150,
                "inputs": [
                    {"name": "x", "dtype": "float32", "shape": ["batch", 3]},
                    {"name": "b", "dtype": "float32", "shape": [3]},
                ],
                "outputs": [{"name": "y", "dtype": "float32", "shape": ["batch", 3]}],
            }
        )

        async def run(meta, buffers):
            specs = json.loads(meta)
            x = np.frombuffer(buffers[0], np.float32).reshape(specs[0]["dims"])
            y = x * 2 + np.frombuffer(buffers[1], np.float32)
            return [json.dumps([{"name": "y", "dtype": "float32", "dims": list(y.shape)}]), y.tobytes()]

        handle.run = AsyncMock(side_effect=run)
        handle.dispose = AsyncMock()
        module = types.ModuleType("js")
        module.lmLite = types.SimpleNamespace(loadOnnxModel=AsyncMock(return_value=handle))
        monkeypatch.setitem(sys.modules, "js", module)
        monkeypatch.setattr("lmlite.llm.ensure_runtime", MagicMock())
        return module, handle

    async def test_url_and_bytes_reach_the_browser_and_results_come_back_as_numpy(self, js):
        module, handle = js

        model = await OnnxModel.load("https://example.test/m.onnx", backend="webgpu")
        module.lmLite.loadOnnxModel.assert_awaited_once_with("https://example.test/m.onnx", "webgpu", None, True)

        out = await model.run({"b": B, "x": np.ones((2, 3), np.float32)})  # any key order
        meta, buffers = handle.run.call_args.args
        assert json.loads(meta) == [
            {"name": "x", "dtype": "float32", "dims": [2, 3]},
            {"name": "b", "dtype": "float32", "dims": [3]},
        ]
        assert buffers == [np.ones((2, 3), np.float32).tobytes(), B.tobytes()]
        assert out["y"].dtype == np.float32 and (out["y"] == [[2, 3, 4]] * 2).all()
        assert model.model_info()["sha256"] == "abc"

        await model.load(FIXTURE.read_bytes())
        assert module.lmLite.loadOnnxModel.await_args.args[0] == FIXTURE.read_bytes()
        await model.dispose()
        handle.dispose.assert_awaited_once()

    async def test_hosted_id_loads_model_onnx_from_the_manifest_base(self, js):
        module, _ = js
        module.lmLite.getHostedModelBase = AsyncMock(return_value="https://example.test/files/models/")

        await OnnxModel.load("lmlite/scale-add", hosted_models_url="https://example.test/files/models/", cache=False)

        module.lmLite.getHostedModelBase.assert_awaited_once_with(
            "lmlite/scale-add", "https://example.test/files/models/"
        )
        module.lmLite.loadOnnxModel.assert_awaited_once_with(
            "https://example.test/files/models/lmlite/scale-add/model.onnx", "wasm", None, False
        )

    async def test_unlisted_hosted_id_is_reported(self, js):
        module, _ = js
        module.lmLite.getHostedModelBase = AsyncMock(return_value=None)

        with pytest.raises(FileNotFoundError, match="not a hosted model"):
            await OnnxModel.load("lmlite/other")

    async def test_sha256_of_bytes_is_checked_before_crossing_the_bridge(self, js):
        module, _ = js

        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            await OnnxModel.load(FIXTURE.read_bytes(), sha256="0" * 64)
        module.lmLite.loadOnnxModel.assert_not_awaited()

    async def test_input_errors_are_raised_before_crossing_the_bridge(self, js):
        _, handle = js
        model = await OnnxModel.load("https://example.test/m.onnx")

        with pytest.raises(TypeError, match="Input 'x' must be float32"):
            await model.run({"x": np.ones((1, 3), np.int64), "b": B})
        handle.run.assert_not_awaited()
