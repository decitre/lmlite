# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

"""OnnxModel: the native path on a real tiny model, and the browser path against a fake JavaScript handle."""

import asyncio
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


@needs_native
class TestMap:
    @pytest.fixture(autouse=True)
    def no_browser(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "js", raising=False)

    @staticmethod
    def feeds(count):
        return [{"x": np.full((1 + n % 3, 3), n, np.float32), "b": B} for n in range(count)]

    async def test_outputs_equal_one_run_per_input_in_order_and_progress_follows(self):
        model = await OnnxModel.load(str(FIXTURE))
        feeds = self.feeds(100)
        progress = []

        mapped = await model.map(feeds, progress_callback=lambda done, total: progress.append((done, total)))

        expected = [await model.run(feed) for feed in feeds]
        assert len(mapped) == 100 and all(np.array_equal(m["y"], e["y"]) for m, e in zip(mapped, expected))
        assert progress == [(n, 100) for n in range(1, 101)]  # once per input

    async def test_a_generator_is_consumed_lazily_and_has_no_total(self):
        model = await OnnxModel.load(str(FIXTURE))
        made = []

        def generate():
            for feed in self.feeds(5):
                made.append(feed)
                yield feed

        progress = []
        mapped = await model.map(generate(), progress_callback=lambda done, total: progress.append((done, total)))

        assert len(mapped) == 5 and progress == [(n, None) for n in range(1, 6)]

    async def test_an_async_generator_works_too(self):
        model = await OnnxModel.load(str(FIXTURE))

        async def generate():
            for feed in self.feeds(3):
                await asyncio.sleep(0)
                yield feed

        assert len(await model.map(generate())) == 3

    async def test_on_error_raise_skip_and_return(self):
        model = await OnnxModel.load(str(FIXTURE))
        good, wrong_dtype, wrong_shape = (
            self.feeds(1)[0],
            {"x": np.ones((1, 3)), "b": B},
            {"x": np.ones(3, np.float32), "b": B},
        )
        feeds = [good, wrong_dtype, good, wrong_shape, good]

        with pytest.raises(TypeError, match="Input 'x' must be float32"):
            await model.map(feeds)
        skipped = await model.map(feeds, on_error="skip")
        returned = await model.map(feeds, on_error="return")

        assert len(skipped) == 3 and all(isinstance(item, dict) for item in skipped)
        assert [type(item) for item in returned] == [dict, TypeError, dict, ValueError, dict]
        assert "rank 2" in str(returned[3]) and (returned[0]["y"] == returned[4]["y"]).all()  # the rest succeed

    async def test_progress_counts_failed_inputs_as_done(self):
        model = await OnnxModel.load(str(FIXTURE))
        progress = []

        await model.map(
            [{"x": np.ones(3, np.float32), "b": B}, self.feeds(1)[0]],
            on_error="skip",
            progress_callback=lambda done, total: progress.append((done, total)),
        )

        assert progress == [(1, 2), (2, 2)]

    async def test_cancel_returns_the_outputs_so_far_and_the_model_stays_usable(self):
        model = await OnnxModel.load(str(FIXTURE))
        cancelled = []

        def stop_after_three(done, total):
            if done == 3:
                cancelled.append(asyncio.ensure_future(model.cancel()))

        mapped = await model.map(self.feeds(50), progress_callback=stop_after_three)

        assert await cancelled[0] is True
        assert 3 <= len(mapped) <= 4  # after the current input: at most one more ran while cancel() was scheduled
        assert len(await model.map(self.feeds(4))) == 4 and (await model.run(self.feeds(1)[0]))["y"].shape == (1, 3)

    async def test_cancel_with_nothing_running_returns_false(self):
        model = await OnnxModel.load(str(FIXTURE))

        assert await model.cancel() is False

    async def test_one_map_at_a_time_and_a_valid_on_error(self):
        model = await OnnxModel.load(str(FIXTURE))
        running = asyncio.ensure_future(model.map(self.feeds(5)))
        await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="Another map"):
            await model.map(self.feeds(1))
        assert len(await running) == 5
        with pytest.raises(ValueError, match="on_error"):
            await model.map(self.feeds(1), on_error="ignore")

    async def test_an_error_in_the_iterable_ends_the_run(self):
        model = await OnnxModel.load(str(FIXTURE))

        def generate():
            yield self.feeds(1)[0]
            raise OSError("cannot read the image")

        with pytest.raises(OSError, match="cannot read"):
            await model.map(generate(), on_error="return")
        assert len(await model.map(self.feeds(2))) == 2  # and the model is free again


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
        monkeypatch.setattr("lmlite._bridge.ensure_runtime", MagicMock())
        return module, handle

    async def test_url_and_bytes_reach_the_browser_and_results_come_back_as_numpy(self, js):
        module, handle = js

        model = await OnnxModel.load("https://example.test/m.onnx", backend="webgpu")
        module.lmLite.loadOnnxModel.assert_awaited_once_with("https://example.test/m.onnx", "webgpu", None, True, True)

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
            "https://example.test/files/models/lmlite/scale-add/model.onnx", "wasm", None, False, True
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

    async def test_map_makes_one_bridge_call_per_input_and_skips_bad_ones_before_the_bridge(self, js):
        _, handle = js
        model = await OnnxModel.load("https://example.test/m.onnx")
        good = {"x": np.ones((1, 3), np.float32), "b": B}

        out = await model.map([good, {"x": np.ones((1, 3), np.int64), "b": B}, good], on_error="return")

        assert [type(item).__name__ for item in out] == ["dict", "TypeError", "dict"]
        assert handle.run.await_count == 2  # the bad input never crossed the bridge

    async def test_input_errors_are_raised_before_crossing_the_bridge(self, js):
        _, handle = js
        model = await OnnxModel.load("https://example.test/m.onnx")

        with pytest.raises(TypeError, match="Input 'x' must be float32"):
            await model.run({"x": np.ones((1, 3), np.int64), "b": B})
        handle.run.assert_not_awaited()
