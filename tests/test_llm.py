# Copyright 2026 Emmanuel Decitre
# SPDX-License-Identifier: Apache-2.0

import os
import json
import tempfile
import types
import zipfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestVersion:
    """Test version module."""

    def test_version_exists(self):
        """Test that version is defined."""
        from lmlite.version import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__) > 0


class TestLLMDefaults:
    """Test LLM class defaults and configuration."""

    def test_defaults(self):
        from lmlite import LLM

        assert LLM.DEFAULTS["generator_model"] is None
        assert LLM.DEFAULTS["embedding_model"] is None
        assert LLM.DEFAULTS["classification_model"] is None
        assert LLM.DEFAULTS["backend"] == "wasm"
        assert LLM.DEFAULTS["embedding_backend"] is None
        assert LLM.DEFAULTS["dtype"] is None
        assert LLM.DEFAULTS["dtype_fallbacks"] is None
        assert LLM.DEFAULTS["embedding_dtype"] is None
        assert LLM.DEFAULTS["max_new_tokens"] == 50
        assert LLM.DEFAULTS["return_full_text"] is True
        assert LLM.DEFAULTS["temperature"] == 0.7
        assert LLM.DEFAULTS["top_k"] == 50
        assert LLM.DEFAULTS["do_sample"] is True
        assert LLM.DEFAULTS["local_models_path"] == "/drive/models"
        assert LLM.DEFAULTS["auto_detect_local"] is True


class TestClearCache:
    async def test_clear_cache_calls_js_runtime(self):
        from lmlite import clear_cache

        import sys

        mock_js_module = types.ModuleType("js")

        mock_js_module.lmLite = types.SimpleNamespace()
        mock_js_module.lmLite.clearModelCache = AsyncMock(
            return_value={
                "deleted_caches": ["transformers-cache"],
                "deleted_databases": ["transformers_cache"],
            }
        )

        with patch("lmlite.llm.ensure_runtime"):
            sys.modules["js"] = mock_js_module
            try:
                result = await clear_cache()
                assert result["deleted_caches"] == ["transformers-cache"]
                assert result["deleted_databases"] == ["transformers_cache"]
                mock_js_module.lmLite.clearModelCache.assert_awaited_once()
            finally:
                if "js" in sys.modules:
                    del sys.modules["js"]

    async def test_clear_cache_raises_if_js_symbol_missing(self):
        from lmlite import clear_cache

        import sys

        mock_js_module = types.ModuleType("js")

        mock_js_module.lmLite = types.SimpleNamespace()

        with patch("lmlite.llm.ensure_runtime"):
            sys.modules["js"] = mock_js_module
            try:
                try:
                    await clear_cache()
                    raise AssertionError("Expected RuntimeError when clearModelCache is missing")
                except RuntimeError as exc:
                    assert "clearModelCache" in str(exc)
            finally:
                if "js" in sys.modules:
                    del sys.modules["js"]


class TestCachedModels:
    @pytest.fixture
    def js(self, monkeypatch):
        import sys

        js = types.ModuleType("js")

        js.lmLite = types.SimpleNamespace()
        monkeypatch.setitem(sys.modules, "js", js)
        monkeypatch.setattr("lmlite.llm.ensure_runtime", MagicMock())
        return js

    async def test_cached_models_returns_list(self, js):
        from lmlite import cached_models

        entries = [
            {"model_id": "Xenova/gpt2", "revision": "main", "files": ["config.json"], "size": 10},
            {"model_id": None, "revision": None, "files": ["https://example.com/x"], "size": None},
        ]
        js.lmLite.getCachedModels = AsyncMock(return_value=entries)

        assert await cached_models() == entries
        js.lmLite.getCachedModels.assert_awaited_once_with()

    async def test_runtime_without_namespace_asks_for_restart(self, js):
        from lmlite import cached_models

        del js.lmLite  # a kernel still holding a runtime from before globalThis.lmLite
        with pytest.raises(RuntimeError, match="restart the kernel"):
            await cached_models()

    async def test_cached_models_passes_through_none(self, js):
        from lmlite import cached_models

        js.lmLite.getCachedModels = AsyncMock(return_value=None)

        assert await cached_models() is None

    async def test_cached_models_requires_current_runtime(self, js):
        from lmlite import cached_models

        with pytest.raises(RuntimeError, match="getCachedModels"):
            await cached_models()

    async def test_clear_cache_forwards_model(self, js):
        from lmlite import clear_cache

        payload = {"model_id": "gpt2", "deleted_files": ["https://huggingface.co/gpt2/resolve/main/config.json"]}
        js.lmLite.clearModelCache = AsyncMock(return_value=payload)

        assert await clear_cache(model="gpt2") == payload
        js.lmLite.clearModelCache.assert_awaited_once_with("gpt2")

    async def test_clear_cache_forwards_none_by_default(self, js):
        from lmlite import clear_cache

        js.lmLite.clearModelCache = AsyncMock(return_value={"deleted_caches": [], "deleted_databases": []})

        await clear_cache()
        js.lmLite.clearModelCache.assert_awaited_once_with(None)


class TestLLMDispose:
    async def test_dispose_forwards_to_js(self):
        from lmlite import LLM

        instance = MagicMock()
        instance.dispose = AsyncMock()
        llm = LLM(instance)
        await llm.dispose()
        instance.dispose.assert_awaited_once_with()

    async def test_dispose_propagates_cleanup_errors(self):
        from lmlite import LLM

        instance = MagicMock()
        instance.dispose = AsyncMock(side_effect=RuntimeError("session release failed"))
        with pytest.raises(RuntimeError, match="session release failed"):
            await LLM(instance).dispose()


class TestLLMGenerate:
    """Test LLM text generation."""

    async def test_generate(self):
        """Test generate method calls JS instance."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.generate = AsyncMock(return_value="Generated text")

        llm = LLM(mock_js_instance)
        result = await llm.generate("Test prompt")

        assert result == "Generated text"
        mock_js_instance.generate.assert_awaited_once_with("Test prompt")


class TestLLMEmbed:
    """Test LLM embedding generation."""

    async def test_embed(self):
        """Test embed method calls JS instance and returns list."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        # JS returns a JS array-like object
        mock_vec = [0.1, 0.2, 0.3]
        mock_js_instance.embed = AsyncMock(return_value=mock_vec)

        llm = LLM(mock_js_instance)
        result = await llm.embed("Test text")

        assert result == [0.1, 0.2, 0.3]
        mock_js_instance.embed.assert_awaited_once_with("Test text")


class TestLLMClassify:
    """Test LLM text classification."""

    async def test_classify_default_top_k(self):
        from lmlite import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.classify = AsyncMock(return_value=[{"label": "POSITIVE", "score": 0.99}])

        llm = LLM(mock_js_instance)
        result = await llm.classify("Great movie")

        assert result[0]["label"] == "POSITIVE"
        mock_js_instance.classify.assert_awaited_once_with("Great movie")

    async def test_classify_with_top_k(self):
        from lmlite import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.classify = AsyncMock(return_value=[{"label": "POSITIVE", "score": 0.99}])

        llm = LLM(mock_js_instance)
        await llm.classify("Great movie", top_k=2)

        mock_js_instance.classify.assert_awaited_once_with("Great movie", 2)

    async def test_classify_converts_nested_proxy_items(self):
        from lmlite import LLM

        class JsProxyLabel:
            def __init__(self, payload):
                self.payload = payload

            def to_py(self):
                return self.payload

        proxy_item = JsProxyLabel({"label": "POSITIVE", "score": 0.99})

        mock_js_instance = MagicMock()
        mock_js_instance.classify = AsyncMock(return_value=[proxy_item])
        llm = LLM(mock_js_instance)

        import sys

        mock_pyodide_ffi = types.ModuleType("pyodide.ffi")
        mock_pyodide_ffi.to_py = MagicMock(side_effect=lambda value: value)
        pyodide_backup = sys.modules.pop("pyodide", None)
        pyodide_ffi_backup = sys.modules.pop("pyodide.ffi", None)
        sys.modules["pyodide.ffi"] = mock_pyodide_ffi

        try:
            result = await llm.classify("Great movie")
            assert result[0]["label"] == "POSITIVE"
            assert result[0]["score"] == 0.99
        finally:
            if "pyodide.ffi" in sys.modules:
                del sys.modules["pyodide.ffi"]
            if pyodide_backup:
                sys.modules["pyodide"] = pyodide_backup
            if pyodide_ffi_backup:
                sys.modules["pyodide.ffi"] = pyodide_ffi_backup


class TestLLMSimilaritySearch:
    """Test LLM similarity search."""

    async def test_similarity_search(self):
        """Test similarity search with mocked JS function."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        # Mock embeddings
        mock_js_instance.embed = AsyncMock(
            side_effect=[
                [1.0, 0.0],  # query
                [1.0, 0.0],  # doc1 - perfect match
                [0.0, 1.0],  # doc2 - orthogonal
                [0.7, 0.7],  # doc3 - partial match
            ]
        )

        llm = LLM(mock_js_instance)

        # Mock the JS cosineSimilarityBatch function
        mock_similarities = [1.0, 0.0, 0.7]

        import sys

        mock_js_module = MagicMock()
        mock_js_module.lmLite.cosineSimilarityBatch = MagicMock(return_value=mock_similarities)
        sys.modules["js"] = mock_js_module

        # Mock to_js to just return the input (simulating successful conversion)
        mock_pyodide_module = MagicMock()
        mock_pyodide_module.ffi = MagicMock()
        mock_pyodide_module.ffi.to_js = MagicMock(side_effect=lambda x: x)

        try:
            # Mock pyodide.ffi module
            sys.modules["pyodide"] = mock_pyodide_module
            sys.modules["pyodide.ffi"] = mock_pyodide_module.ffi

            results = await llm.similarity_search("query", ["doc1", "doc2", "doc3"])

            # Results should be sorted by similarity (descending)
            assert len(results) == 3
            assert results[0] == ("doc1", 1.0)
            assert results[1] == ("doc3", 0.7)
            assert results[2] == ("doc2", 0.0)

            # Verify cosineSimilarityBatch was called
            mock_js_module.lmLite.cosineSimilarityBatch.assert_called_once()
        finally:
            if "js" in sys.modules:
                del sys.modules["js"]
            if "pyodide" in sys.modules:
                del sys.modules["pyodide"]
            if "pyodide.ffi" in sys.modules:
                del sys.modules["pyodide.ffi"]


class TestLLMExportModels:
    """Test LLM.export_models()."""

    async def test_export_models_already_exists(self, capsys):
        """Test that export skips if zip already exists."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        llm = LLM(mock_js_instance, models=["gpt2"])

        with (
            patch("os.path.exists", side_effect=lambda p: "model-Xenova-gpt2@" in p),
            patch("lmlite.llm._get_latest_revision", return_value="bf2c7f02e0b826c60d03af341171bde20893da66"),
        ):
            await llm.export_models()

        out = capsys.readouterr().out
        assert "already exported" in out.lower()

    async def test_export_models_string_result(self, capsys):
        """Test handling string result from JS side."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.exportModelFiles = AsyncMock(return_value="Export complete")
        llm = LLM(mock_js_instance, models=["gpt2"])

        with patch("os.path.exists", return_value=False):
            await llm.export_models()

        out = capsys.readouterr().out
        assert "Export complete" in out

    async def test_export_models_with_js_files(self, capsys):
        """Test exporting model files when JS returns file objects."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.model = "gpt2"
        mock_file1 = MagicMock()
        mock_file1.filename = "config.json"
        mock_file1.data.to_py = MagicMock(return_value=b'{"key": "value"}')
        mock_result.files = [mock_file1]
        mock_js_instance.exportModelFiles = AsyncMock(return_value=mock_result)
        llm = LLM(mock_js_instance, models=["gpt2"])

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("os.path.exists", return_value=False), patch("os.makedirs"):
                original_init = zipfile.ZipFile.__init__

                def patched_init(self, file, *args, **kwargs):
                    if isinstance(file, str) and "/drive/models" in file:
                        file = os.path.join(tmpdir, os.path.basename(file))
                    return original_init(self, file, *args, **kwargs)

                with patch.object(zipfile.ZipFile, "__init__", patched_init):
                    await llm.export_models()

        out = capsys.readouterr().out
        assert "Successfully exported" in out
        assert "1 files" in out

    async def test_export_models_no_models(self, capsys):
        """Test export_models when no models are configured."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        llm = LLM(mock_js_instance)

        await llm.export_models()

        out = capsys.readouterr().out
        assert "no models" in out.lower()


class TestLLMChat:
    """Test LLM chat functionality."""

    async def test_chat_single_message(self):
        """Test chat with single user message."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.chat = AsyncMock(return_value="Hi there!")

        llm = LLM(mock_js_instance)
        messages = [{"role": "user", "content": "Hello"}]
        result = await llm.chat(messages)

        assert result == "Hi there!"
        mock_js_instance.chat.assert_awaited_once_with(json.dumps(messages))

    async def test_chat_conversation_history(self):
        """Test chat with conversation history."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.chat = AsyncMock(return_value="I'm doing well!")

        llm = LLM(mock_js_instance)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]
        result = await llm.chat(messages)

        assert "I'm doing well" in result
        mock_js_instance.chat.assert_awaited_once_with(json.dumps(messages))

    async def test_chat_custom_role(self):
        """Test chat with custom role (not user/assistant)."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.chat = AsyncMock(return_value="Response")

        llm = LLM(mock_js_instance)
        messages = [{"role": "custom", "content": "test"}]
        result = await llm.chat(messages)

        assert result == "Response"

    async def test_chat_pyodide_uses_json_bridge(self, monkeypatch):
        import sys

        from lmlite import LLM

        ffi = types.ModuleType("pyodide.ffi")
        ffi.to_js = MagicMock(side_effect=AssertionError("Chat must not require FFI conversion"))
        js = types.ModuleType("js")
        js.lmLite = types.SimpleNamespace()
        js.Object = MagicMock()
        monkeypatch.setitem(sys.modules, "pyodide.ffi", ffi)
        monkeypatch.setitem(sys.modules, "js", js)
        instance = MagicMock(chat=AsyncMock(return_value="Answer"))
        messages = [{"role": "user", "content": "Hello"}]

        assert await LLM(instance).chat(messages) == "Answer"
        ffi.to_js.assert_not_called()
        instance.chat.assert_awaited_once_with(json.dumps(messages))

    @pytest.mark.parametrize("compatibility_module", [True, False])
    async def test_chat_xeus_uses_json_with_or_without_pyodide_shim(self, monkeypatch, compatibility_module):
        import sys

        from lmlite import LLM

        ffi = types.ModuleType("pyodide.ffi")

        def to_js(value):
            raise AssertionError("Chat must not call the Xeus compatibility converter")

        ffi.to_js = MagicMock(side_effect=to_js)
        monkeypatch.setitem(sys.modules, "pyodide.ffi", ffi if compatibility_module else None)
        pjs = types.ModuleType("pjs")
        pjs.to_js = MagicMock(side_effect=to_js)
        monkeypatch.setitem(sys.modules, "pjs", pjs)
        instance = MagicMock(chat=AsyncMock(return_value="Answer"))
        messages = [{"role": "user", "content": 'Bonjour \u00e9\n"quoted" <|im_start|>'}]

        assert await LLM(instance).chat(messages) == "Answer"
        pjs.to_js.assert_not_called()
        ffi.to_js.assert_not_called()
        instance.chat.assert_awaited_once_with(json.dumps(messages))
        assert json.loads(instance.chat.call_args.args[0]) == messages

    async def test_chat_propagates_errors(self):
        from lmlite import LLM

        instance = MagicMock(chat=AsyncMock(side_effect=RuntimeError("template error")))
        with pytest.raises(RuntimeError, match="template error"):
            await LLM(instance).chat([{"role": "user", "content": "Hello"}])


class TestLLMStreaming:
    @staticmethod
    def _instance(*chunks):
        stream = MagicMock(read=AsyncMock(side_effect=chunks))
        return MagicMock(chatStream=AsyncMock(return_value=stream), generateStream=AsyncMock(return_value=stream))

    async def test_chat_stream_yields_chunks_until_end(self):
        from lmlite import LLM

        instance = self._instance(" Hel", "lo", "")
        messages = [{"role": "user", "content": "Hi"}]
        stream = LLM(instance).chat_stream(messages)
        instance.chatStream.assert_not_called()  # lazy until iterated
        assert [chunk async for chunk in stream] == [" Hel", "lo"]
        instance.chatStream.assert_awaited_once_with(json.dumps(messages))
        instance.cancel.assert_not_called()

    async def test_generate_stream(self):
        from lmlite import LLM

        instance = self._instance("a", "")
        assert [chunk async for chunk in LLM(instance).generate_stream("p")] == ["a"]
        instance.generateStream.assert_awaited_once_with("p")

    async def test_closing_stream_early_cancels(self):
        from lmlite import LLM

        instance = self._instance("a", "b", "")
        stream = LLM(instance).chat_stream([{"role": "user", "content": "Hi"}])
        assert await anext(stream) == "a"
        await stream.aclose()
        instance.cancel.assert_called_once_with()

    async def test_stream_error_propagates(self):
        from lmlite import LLM

        instance = self._instance("a", RuntimeError("generation failed"))
        with pytest.raises(RuntimeError, match="generation failed"):
            [chunk async for chunk in LLM(instance).chat_stream([{"role": "user", "content": "Hi"}])]

    async def test_chat_tokens_parses_json_records(self):
        from lmlite import LLM

        record = {"text": " Hi", "prob": 0.6, "alternatives": [{"text": " Hello", "prob": 0.3}]}
        stream = MagicMock(read=AsyncMock(side_effect=[json.dumps(record), ""]))
        instance = MagicMock(chatTokens=AsyncMock(return_value=stream), generateTokens=AsyncMock(return_value=stream))
        messages = [{"role": "user", "content": "Hi"}]
        assert [t async for t in LLM(instance).chat_tokens(messages, alternatives=2)] == [record]
        instance.chatTokens.assert_awaited_once_with(json.dumps(messages), 2)
        stream.read.side_effect = [json.dumps(record), ""]
        assert [t async for t in LLM(instance).generate_tokens("p")] == [record]
        instance.generateTokens.assert_awaited_once_with("p", 1)

    @pytest.mark.parametrize("alternatives", [-1, 1.5, True, "2"])
    def test_tokens_reject_invalid_alternatives(self, alternatives):
        from lmlite import LLM

        with pytest.raises(ValueError, match="alternatives"):
            LLM(MagicMock()).chat_tokens([], alternatives=alternatives)
        with pytest.raises(ValueError, match="alternatives"):
            LLM(MagicMock()).generate_tokens("p", alternatives=alternatives)

    async def test_cancel(self):
        from lmlite import LLM

        instance = MagicMock(cancel=AsyncMock(return_value=True))
        assert await LLM(instance).cancel() is True
        instance.cancel.assert_awaited_once_with()


class TestLLMAutoDetect:
    """Test auto-detection and cache injection of local models."""

    async def test_auto_detect_zip_cache_injection(self):
        """Test that zip files are injected into browser cache without filesystem extraction."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.init = AsyncMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "xenova-gpt2.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("Xenova/gpt2/config.json", "{}")
                zf.writestr("Xenova/gpt2/vocab.json", "{}")
                zf.writestr("Xenova/gpt2/onnx/model_quantized.onnx", b"onnx")

            import sys

            mock_js_module = MagicMock()
            mock_js_module.lmLite.createLLM = MagicMock(return_value=mock_js_instance)
            mock_js_module.lmLite.isModelCached = AsyncMock(return_value=False)
            mock_js_module.lmLite.loadModelToCache = AsyncMock()

            with patch("lmlite.llm.ensure_runtime"):
                sys.modules["js"] = mock_js_module
                try:
                    await LLM.create(local_models_path=tmpdir, generator_model="gpt2")

                    assert mock_js_module.lmLite.loadModelToCache.await_count == 3
                    files_arg = {}
                    for call in mock_js_module.lmLite.loadModelToCache.await_args_list:
                        assert call.args[0] == "gpt2"
                        assert len(call.args[1]) == 1
                        files_arg.update(call.args[1])
                    assert "config.json" in files_arg
                    assert "vocab.json" in files_arg
                    assert files_arg["onnx/model_quantized.onnx"] == b"onnx"

                    # No filesystem extraction
                    assert not os.path.exists(os.path.join(tmpdir, "Xenova"))

                    # Config passed to createLLM has no use_local_models / emfs_model_uri
                    config = mock_js_module.lmLite.createLLM.call_args[0][0]
                    if isinstance(config, dict):
                        assert "use_local_models" not in config
                        assert "emfs_model_uri" not in config
                finally:
                    if "js" in sys.modules:
                        del sys.modules["js"]

    async def test_auto_detect_zip_read_progress(self, capsys):
        """Test read progress reaches completion and excludes unsafe archive paths."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.init = AsyncMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "xenova-gpt2.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("Xenova/gpt2/config.json", b"{}")
                zf.writestr("Xenova/gpt2/onnx/model.onnx", b"weights")
                zf.writestr("Xenova/gpt2/../outside.bin", b"ignored" * 100)

            import sys

            mock_js_module = MagicMock()
            mock_js_module.lmLite.createLLM = MagicMock(return_value=mock_js_instance)
            mock_js_module.lmLite.isModelCached = AsyncMock(return_value=False)
            mock_js_module.lmLite.loadModelToCache = AsyncMock()

            with patch("lmlite.llm.ensure_runtime"):
                sys.modules["js"] = mock_js_module
                try:
                    await LLM.create(local_models_path=tmpdir, generator_model="gpt2")

                    progress = capsys.readouterr().err
                    assert "100%" in progress
                    assert "Opening local archive for gpt2" in progress
                    assert "Caching gpt2: onnx/model.onnx" in progress
                    assert "Models ready: 100%" in progress

                    files = {}
                    for call in mock_js_module.lmLite.loadModelToCache.await_args_list:
                        files.update(call.args[1])
                    assert set(files) == {"config.json", "onnx/model.onnx"}
                finally:
                    if "js" in sys.modules:
                        del sys.modules["js"]

    async def test_auto_detect_skips_injection_when_already_cached(self):
        """Test that zip injection is skipped when model is already in browser cache."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.init = AsyncMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "xenova-gpt2.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("Xenova/gpt2/config.json", "{}")

            import sys

            mock_js_module = MagicMock()
            mock_js_module.lmLite.createLLM = MagicMock(return_value=mock_js_instance)
            mock_js_module.lmLite.isModelCached = AsyncMock(return_value=True)
            mock_js_module.lmLite.loadModelToCache = AsyncMock()

            with patch("lmlite.llm.ensure_runtime"):
                sys.modules["js"] = mock_js_module
                try:
                    await LLM.create(local_models_path=tmpdir, generator_model="gpt2")

                    mock_js_module.lmLite.isModelCached.assert_awaited_once_with("gpt2")
                    mock_js_module.lmLite.loadModelToCache.assert_not_awaited()
                finally:
                    if "js" in sys.modules:
                        del sys.modules["js"]

    async def test_auto_detect_without_is_model_cached_symbol(self):
        """Test that zip injection still works when js.lmLite.isModelCached is unavailable."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.init = AsyncMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "xenova-gpt2.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("Xenova/gpt2/config.json", "{}")

            import sys

            mock_js_module = types.ModuleType("js")

            mock_js_module.lmLite = types.SimpleNamespace()
            mock_js_module.lmLite.createLLM = MagicMock(return_value=mock_js_instance)
            mock_js_module.lmLite.loadModelToCache = AsyncMock()

            with patch("lmlite.llm.ensure_runtime"):
                sys.modules["js"] = mock_js_module
                try:
                    await LLM.create(local_models_path=tmpdir, generator_model="gpt2")

                    mock_js_module.lmLite.loadModelToCache.assert_awaited_once()
                finally:
                    if "js" in sys.modules:
                        del sys.modules["js"]

    async def test_auto_detect_full_model_path(self):
        """Test that explicit model paths are passed through without Xenova prefixing."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.init = AsyncMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "model-acme-gpt2@main.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("acme/gpt2/config.json", "{}")

            import sys

            mock_js_module = MagicMock()
            mock_js_module.lmLite.createLLM = MagicMock(return_value=mock_js_instance)
            mock_js_module.lmLite.isModelCached = AsyncMock(return_value=False)
            mock_js_module.lmLite.loadModelToCache = AsyncMock()

            with patch("lmlite.llm.ensure_runtime"):
                sys.modules["js"] = mock_js_module
                try:
                    await LLM.create(local_models_path=tmpdir, generator_model="acme/gpt2")

                    mock_js_module.lmLite.isModelCached.assert_awaited_once_with("acme/gpt2")
                    mock_js_module.lmLite.loadModelToCache.assert_awaited_once()
                    call_args = mock_js_module.lmLite.loadModelToCache.call_args[0]
                    assert call_args[0] == "acme/gpt2"
                finally:
                    if "js" in sys.modules:
                        del sys.modules["js"]

    async def test_auto_detect_message_when_not_local_nor_cached(self, capsys):
        """Test user-facing message when model is absent locally and in browser cache."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.init = AsyncMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            import sys

            mock_js_module = MagicMock()
            mock_js_module.lmLite.createLLM = MagicMock(return_value=mock_js_instance)
            mock_js_module.lmLite.isModelCached = AsyncMock(return_value=False)
            mock_js_module.lmLite.loadModelToCache = AsyncMock()

            with patch("lmlite.llm.ensure_runtime"):
                sys.modules["js"] = mock_js_module
                try:
                    await LLM.create(local_models_path=tmpdir, generator_model="gpt2")
                    out = capsys.readouterr().out
                    expected_message = (
                        f"No Xenova/gpt2 model found in {tmpdir}/ nor in browser cache, will download from 🤗"
                    )
                    assert expected_message in out
                finally:
                    if "js" in sys.modules:
                        del sys.modules["js"]

    async def test_auto_detect_message_when_found_in_browser_cache(self, capsys):
        """Test user-facing message when model is already in browser cache."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.init = AsyncMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            import sys

            mock_js_module = MagicMock()
            mock_js_module.lmLite.createLLM = MagicMock(return_value=mock_js_instance)
            mock_js_module.lmLite.isModelCached = AsyncMock(return_value=True)
            mock_js_module.lmLite.loadModelToCache = AsyncMock()

            with patch("lmlite.llm.ensure_runtime"):
                sys.modules["js"] = mock_js_module
                try:
                    await LLM.create(local_models_path=tmpdir, generator_model="gpt2")
                    out = capsys.readouterr().out
                    assert "Xenova/gpt2 model found in browser cache" in out
                finally:
                    if "js" in sys.modules:
                        del sys.modules["js"]

    async def test_auto_detect_checks_all_configured_models(self, capsys):
        """Test generator and embedding models are both checked and reported."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.init = AsyncMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "xenova-gpt2.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("Xenova/gpt2/config.json", "{}")

            import sys

            mock_js_module = MagicMock()
            mock_js_module.lmLite.createLLM = MagicMock(return_value=mock_js_instance)
            mock_js_module.lmLite.isModelCached = AsyncMock(side_effect=[True, False])
            mock_js_module.lmLite.loadModelToCache = AsyncMock()

            with patch("lmlite.llm.ensure_runtime"):
                sys.modules["js"] = mock_js_module
                try:
                    await LLM.create(
                        local_models_path=tmpdir,
                        generator_model="gpt2",
                        embedding_model="all-mpnet-base-v2",
                    )
                    out = capsys.readouterr().out
                    assert "Xenova/gpt2 model found in browser cache" in out
                    assert f"Skipping local zip injection from {zip_path}" in out
                    assert (
                        f"No Xenova/all-mpnet-base-v2 model found in {tmpdir}/ nor in browser cache, "
                        "will download from 🤗" in out
                    )

                    checked_models = [call.args[0] for call in mock_js_module.lmLite.isModelCached.await_args_list]
                    assert checked_models == ["gpt2", "all-mpnet-base-v2"]
                    mock_js_module.lmLite.loadModelToCache.assert_not_awaited()
                finally:
                    if "js" in sys.modules:
                        del sys.modules["js"]

    async def test_auto_detect_requires_async_is_model_cached(self):
        """Test that isModelCached must be async when provided."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.init = AsyncMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            import sys

            mock_js_module = types.ModuleType("js")

            mock_js_module.lmLite = types.SimpleNamespace()
            mock_js_module.lmLite.createLLM = MagicMock(return_value=mock_js_instance)
            mock_js_module.lmLite.isModelCached = MagicMock(return_value=False)

            with patch("lmlite.llm.ensure_runtime"):
                sys.modules["js"] = mock_js_module
                try:
                    try:
                        await LLM.create(local_models_path=tmpdir, generator_model="gpt2")
                        raise AssertionError("Expected TypeError for non-async isModelCached")
                    except TypeError as exc:
                        assert "await" in str(exc).lower()
                finally:
                    if "js" in sys.modules:
                        del sys.modules["js"]

    async def test_auto_detect_with_js_proxy_like_object(self):
        """Test runtime symbol lookup for js-proxy-like objects (no __dict__)."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.init = AsyncMock()

        class JsProxyLike:
            __slots__ = ("createLLM", "isModelCached", "loadModelToCache")

        js_proxy_like = JsProxyLike()
        js_proxy_like.createLLM = MagicMock(return_value=mock_js_instance)
        js_proxy_like.isModelCached = AsyncMock(return_value=False)
        js_proxy_like.loadModelToCache = AsyncMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "xenova-gpt2.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("Xenova/gpt2/config.json", "{}")

            import sys

            with patch("lmlite.llm.ensure_runtime"):
                sys.modules["js"] = types.SimpleNamespace(lmLite=js_proxy_like)
                try:
                    await LLM.create(local_models_path=tmpdir, generator_model="gpt2")
                    js_proxy_like.loadModelToCache.assert_awaited_once()
                finally:
                    if "js" in sys.modules:
                        del sys.modules["js"]


class TestProgressRenderer:
    @pytest.mark.parametrize("platform", ["emscripten", "wasi"])
    def test_browser_uses_notebook_widgets_without_monitor_thread(self, monkeypatch, platform):
        import sys

        from tqdm import tqdm as text_tqdm

        from lmlite.llm import _loading_progress

        notebook = types.ModuleType("tqdm.notebook")
        notebook.IProgress = object()
        notebook.tqdm = MagicMock(monitor_interval=10)
        monkeypatch.setitem(sys.modules, "tqdm.notebook", notebook)
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.setattr(text_tqdm, "monitor_interval", 10)
        with patch("tqdm.auto.tqdm") as auto:
            progress = _loading_progress()

        assert text_tqdm.monitor_interval == 0
        assert notebook.tqdm.monitor_interval == 0
        notebook.tqdm.assert_called_once_with(
            desc="Preparing models", total=None, unit="B", unit_scale=True, mininterval=0.1, bar_format="{desc}"
        )
        assert progress is notebook.tqdm.return_value
        auto.assert_not_called()

    def test_browser_reports_missing_widget_support_and_uses_threadless_text(self, monkeypatch):
        import sys

        from lmlite.llm import _loading_progress

        notebook = types.ModuleType("tqdm.notebook")
        notebook.IProgress = None
        notebook.tqdm = MagicMock()
        monkeypatch.setitem(sys.modules, "tqdm.notebook", notebook)
        monkeypatch.setattr(sys, "platform", "emscripten")
        with patch("tqdm.tqdm") as text:
            with pytest.warns(RuntimeWarning, match="Notebook progress widgets are unavailable"):
                progress = _loading_progress()

        assert text.monitor_interval == 0
        assert progress is text.return_value
        notebook.tqdm.assert_not_called()

    def test_native_python_keeps_auto_renderer_and_monitor_settings(self, monkeypatch):
        import sys

        from tqdm import tqdm as text_tqdm

        from lmlite.llm import _loading_progress

        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(text_tqdm, "monitor_interval", 10)
        with patch("tqdm.auto.tqdm") as auto:
            progress = _loading_progress()

        assert progress is auto.return_value
        assert text_tqdm.monitor_interval == 10


class TestLLMLoadingProgress:
    @pytest.fixture
    def loading(self, monkeypatch):
        import sys

        bar = MagicMock(n=0, bar_format="{desc}")

        def reset(total=None):
            bar.n = 0
            if total is not None:
                bar.total = total

        def update(amount):
            bar.n += amount

        bar.reset.side_effect = reset
        bar.update.side_effect = update
        bar.__enter__.return_value = bar
        monkeypatch.setattr("lmlite.llm._loading_progress", MagicMock(return_value=bar))
        monkeypatch.setattr("lmlite.llm.ensure_runtime", MagicMock())

        instance = MagicMock()
        instance.init = AsyncMock()
        js = types.ModuleType("js")
        js.lmLite = types.SimpleNamespace()
        js.lmLite.createLLM = MagicMock(return_value=instance)
        js.lmLite.isModelCached = AsyncMock(return_value=False)
        js.lmLite.loadModelToCache = AsyncMock()
        monkeypatch.setitem(sys.modules, "js", js)

        ffi = types.ModuleType("pyodide.ffi")
        ffi.to_js = MagicMock(side_effect=lambda value: value)
        ffi.create_proxy = MagicMock(side_effect=lambda callback: MagicMock(side_effect=callback))
        monkeypatch.setitem(sys.modules, "pyodide.ffi", ffi)
        return bar, instance, js, ffi

    @pytest.mark.parametrize("return_full_text", [True, False])
    async def test_return_full_text_is_forwarded_to_js(self, loading, return_full_text):
        from lmlite import LLM

        _, instance, js, ffi = loading
        await LLM.create(
            generator_model="gpt2",
            return_full_text=return_full_text,
            auto_detect_local=False,
        )

        config = js.lmLite.createLLM.call_args.args[0]
        assert config["return_full_text"] is return_full_text
        ffi.to_js.assert_called_once_with(config)
        instance.init.assert_awaited_once()

    @pytest.mark.parametrize("embedding_dtype", [None, "q8"])
    @pytest.mark.parametrize("embedding_backend", [None, "wasm", "webgpu"])
    async def test_embedding_settings_are_forwarded_to_js(self, loading, embedding_dtype, embedding_backend):
        from lmlite import LLM

        _, instance, js, ffi = loading
        await LLM.create(
            embedding_model="all-MiniLM-L6-v2",
            embedding_dtype=embedding_dtype,
            embedding_backend=embedding_backend,
            dtype="fp32",
            auto_detect_local=False,
        )

        config = js.lmLite.createLLM.call_args.args[0]
        assert config["embedding_dtype"] == embedding_dtype
        assert config["embedding_backend"] == embedding_backend
        assert config["dtype"] == "fp32"
        assert config["embedding_model"] == "all-MiniLM-L6-v2"
        ffi.to_js.assert_called_once_with(config)
        instance.init.assert_awaited_once()

    async def test_hosted_model_bypasses_zip_and_hub_revision_lookup(self, loading, tmp_path, monkeypatch):
        import sys

        from lmlite import LLM

        _, instance, js, _ = loading
        monkeypatch.setattr(sys, "platform", "emscripten")
        js.lmLite.getHostedModelBase = AsyncMock(return_value="https://example.test/demo/models/")
        with patch("lmlite.llm.model_zip_candidates") as candidates:
            await LLM.create(generator_model="gpt2", local_models_path=str(tmp_path))

        candidates.assert_not_called()
        js.lmLite.isModelCached.assert_not_awaited()
        js.lmLite.loadModelToCache.assert_not_awaited()
        config = js.lmLite.createLLM.call_args.args[0]
        assert config["hosted_model_base"] == "https://example.test/demo/models/"
        assert config["hosted_models"] == ["Xenova/gpt2"]
        instance.init.assert_awaited_once()

    async def test_explicit_hosted_url_and_nonhosted_model_fallback(self, loading, tmp_path):
        from lmlite import LLM

        _, _, js, _ = loading
        js.lmLite.getHostedModelBase = AsyncMock(side_effect=["https://example.test/models/", None])
        await LLM.create(
            generator_model="gpt2@main",
            embedding_model="acme/embed@v1",
            hosted_models_url="https://example.test/models/",
            local_models_path=str(tmp_path),
        )

        config = js.lmLite.createLLM.call_args.args[0]
        assert config["hosted_models"] == ["Xenova/gpt2"]
        assert js.lmLite.getHostedModelBase.await_count == 2
        js.lmLite.isModelCached.assert_awaited_once_with("acme/embed@v1")

    async def test_hosted_manifest_error_is_not_silently_ignored(self, loading):
        from lmlite import LLM

        bar, instance, js, _ = loading
        js.lmLite.getHostedModelBase = AsyncMock(side_effect=RuntimeError("manifest request failed"))
        with pytest.raises(RuntimeError, match="manifest request failed"):
            await LLM.create(generator_model="gpt2", hosted_models_url="https://example.test/models/")

        instance.init.assert_not_awaited()
        bar.__exit__.assert_called_once()

    async def test_explicit_hosted_url_requires_current_runtime(self, loading):
        from lmlite import LLM

        _, instance, _, _ = loading
        with pytest.raises(RuntimeError, match="JS runtime missing getHostedModelBase"):
            await LLM.create(generator_model="gpt2", hosted_models_url="https://example.test/models/")

        instance.init.assert_not_awaited()

    async def test_download_updates_and_initialization_status(self, loading):
        from lmlite import LLM

        bar, instance, _, _ = loading
        snapshots = []

        async def init(callback):
            callback("acme/model", "progress", "model.onnx", 10, 100)
            snapshots.append((bar.n, bar.total))
            callback("acme/model", "progress", "model.onnx", 40, 100)
            snapshots.append((bar.n, bar.total))
            callback("acme/model", "progress", "tokenizer.json", 2, 20)
            snapshots.append((bar.n, bar.total))
            callback("acme/model", "warming up", "", 0, 0)
            snapshots.append((bar.n, bar.total))

        instance.init.side_effect = init
        await LLM.create(generator_model="acme/model", auto_detect_local=False)

        assert snapshots == [(10, 100), (40, 100), (2, 20), (2, 20)]
        assert (bar.n, bar.total) == (1, 1)
        assert "{percentage" in bar.bar_format
        assert bar.set_description.call_args.args == ("Models ready",)
        instance.init.call_args.args[0].destroy.assert_called_once()
        bar.__exit__.assert_called_once()

    async def test_progress_and_proxy_closed_on_failure(self, loading):
        from lmlite import LLM

        bar, instance, _, _ = loading
        instance.init.side_effect = RuntimeError("WebGPU unavailable")

        with pytest.raises(RuntimeError, match="WebGPU unavailable"):
            await LLM.create(generator_model="acme/model", auto_detect_local=False)

        instance.init.call_args.args[0].destroy.assert_called_once()
        bar.__exit__.assert_called_once()
        assert not any(call.args == ("Models ready",) for call in bar.set_description.call_args_list)

    async def test_xeus_callback_reports_download_progress(self, loading, monkeypatch):
        import sys

        from lmlite import LLM

        bar, instance, _, _ = loading
        monkeypatch.setitem(sys.modules, "pyodide.ffi", None)
        pyjs = types.ModuleType("pyjs")
        handle = MagicMock()
        pyjs.create_callable = MagicMock(side_effect=lambda value: (MagicMock(side_effect=value), handle))
        monkeypatch.setitem(sys.modules, "pyjs", pyjs)
        snapshots = []

        async def init(callback):
            handle.delete.assert_not_called()
            callback("acme/model", "progress", "model.onnx", 50, 100)
            snapshots.append((bar.n, bar.total))
            callback("acme/model", "warming up", "", 0, 0)
            assert bar.set_description.call_args.args == ("Warming up acme/model",)
            assert bar.bar_format == "{desc}"

        instance.init.side_effect = init
        await LLM.create(generator_model="acme/model", auto_detect_local=False)

        assert snapshots == [(50, 100)]
        pyjs.create_callable.assert_called_once()
        assert instance.init.call_args.args[0] is not pyjs.create_callable.call_args.args[0]
        handle.delete.assert_called_once()
        assert bar.set_description.call_args.args == ("Models ready",)

    @pytest.mark.parametrize("failure_stage", ["create", "init", "callback"])
    async def test_xeus_callback_handle_closed_on_failure(self, loading, monkeypatch, failure_stage):
        import sys

        from lmlite import LLM

        bar, instance, js, _ = loading
        monkeypatch.setitem(sys.modules, "pyodide.ffi", None)
        pyjs = types.ModuleType("pyjs")
        handle = MagicMock()
        pyjs.create_callable = MagicMock(side_effect=lambda value: (MagicMock(side_effect=value), handle))
        monkeypatch.setitem(sys.modules, "pyjs", pyjs)
        error = RuntimeError("Loading failed")
        if failure_stage == "create":
            js.lmLite.createLLM.side_effect = error
        elif failure_stage == "init":
            instance.init.side_effect = error
        else:
            bar.update.side_effect = error

            async def init(callback):
                callback("acme/model", "progress", "model.onnx", 50, 100)

            instance.init.side_effect = init

        with pytest.raises(RuntimeError, match="Loading failed"):
            await LLM.create(generator_model="acme/model", auto_detect_local=False)

        handle.delete.assert_called_once()
        bar.__exit__.assert_called_once()
        assert not any(call.args == ("Models ready",) for call in bar.set_description.call_args_list)

    async def test_download_retry_resets_progress(self, loading):
        from lmlite import LLM

        bar, instance, _, _ = loading
        snapshots = []

        async def init(callback):
            callback("acme/model", "progress", "model.onnx", 50, 100)
            callback("acme/model", "progress", "model.onnx", 5, 100)
            snapshots.append((bar.n, bar.total))

        instance.init.side_effect = init
        await LLM.create(generator_model="acme/model", auto_detect_local=False)

        assert snapshots == [(5, 100)]
        assert all(call.args[0] >= 0 for call in bar.update.call_args_list)

    async def test_download_sets_widget_total_and_handles_late_size(self, loading):
        from lmlite import LLM

        bar, instance, _, _ = loading

        async def init(callback):
            callback("acme/model", "progress", "model.onnx", 5, 0)
            callback("acme/model", "progress", "model.onnx", 20, 100)
            assert (bar.n, bar.total) == (20, 100)
            bar.reset.assert_any_call(total=100)
            callback("acme/model", "progress", "model.onnx", 100, 100)
            assert (bar.n, bar.total) == (100, 100)

        instance.init.side_effect = init
        await LLM.create(generator_model="acme/model", auto_detect_local=False)

    async def test_zip_reads_update_in_chunks(self, loading, tmp_path):
        from lmlite import LLM

        bar, _, js, _ = loading
        weights = b"x" * (1024 * 1024 + 7)
        with zipfile.ZipFile(tmp_path / "model-Xenova-gpt2@main.zip", "w") as archive:
            archive.writestr("Xenova/gpt2/onnx/model.onnx", weights)
            archive.writestr("Xenova/gpt2/../ignored", b"ignored")
            archive.writestr("Xenova/gpt2/onnx/", b"")

        await LLM.create(generator_model="gpt2@main", local_models_path=str(tmp_path))

        assert [call.args[0] for call in bar.update.call_args_list] == [1024 * 1024, 7, 1]
        bar.reset.assert_any_call(total=len(weights))
        js.lmLite.loadModelToCache.assert_awaited_once_with("gpt2@main", {"onnx/model.onnx": weights})

    async def test_zip_files_are_cached_before_reading_next_file(self, loading, tmp_path, monkeypatch):
        from lmlite import LLM

        bar, _, js, _ = loading
        zip_path = tmp_path / "model-Xenova-gpt2@main.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("Xenova/gpt2/config.json", "{}")
            archive.writestr("Xenova/gpt2/onnx/model.onnx", b"weights")

        events = []
        original_open = zipfile.ZipFile.open

        def open_member(archive, name, *args, **kwargs):
            events.append(("read", name))
            return original_open(archive, name, *args, **kwargs)

        async def cache_file(model, files):
            assert len(files) == 1
            filename, data = next(iter(files.items()))
            assert isinstance(data, bytearray)
            events.append(("cache", filename))
            assert bar.set_description.call_args.args == (f"Caching {model}: {filename}",)

        monkeypatch.setattr(zipfile.ZipFile, "open", open_member)
        js.lmLite.loadModelToCache.side_effect = cache_file

        await LLM.create(generator_model="gpt2@main", local_models_path=str(tmp_path))

        assert events == [
            ("read", "Xenova/gpt2/config.json"),
            ("cache", "config.json"),
            ("read", "Xenova/gpt2/onnx/model.onnx"),
            ("cache", "onnx/model.onnx"),
        ]

    async def test_opening_status_is_visible_before_archive_access(self, loading, tmp_path, monkeypatch):
        from lmlite import LLM

        bar, instance, _, _ = loading
        zip_path = tmp_path / "model-Xenova-gpt2@main.zip"
        zip_path.write_bytes(b"corrupt archive")

        def open_archive(*args, **kwargs):
            assert bar.set_description.call_args.args == ("Opening local archive for gpt2@main",)
            assert bar.bar_format == "{desc}"
            raise zipfile.BadZipFile("invalid archive")

        monkeypatch.setattr(zipfile, "ZipFile", open_archive)

        with pytest.raises(zipfile.BadZipFile, match="invalid archive"):
            await LLM.create(generator_model="gpt2@main", local_models_path=str(tmp_path))

        instance.init.assert_not_awaited()
        bar.__exit__.assert_called_once()

    async def test_status_preserves_download_bytes_and_hides_counter(self, loading):
        from lmlite import LLM

        bar, instance, _, _ = loading

        async def init(callback):
            callback("acme/model", "progress", "model.onnx", 100, 100)
            assert bar.bar_format is None
            callback("acme/model", "warming up", "", 0, 0)
            assert (bar.n, bar.total) == (100, 100)
            assert bar.bar_format == "{desc}"

        instance.init.side_effect = init
        await LLM.create(generator_model="acme/model", auto_detect_local=False)

        assert (bar.n, bar.total) == (1, 1)

    async def test_cache_failure_stops_before_reading_next_file(self, loading, tmp_path):
        from lmlite import LLM

        bar, instance, js, _ = loading
        with zipfile.ZipFile(tmp_path / "model-Xenova-gpt2@main.zip", "w") as archive:
            archive.writestr("config.json", "{}")
            archive.writestr("model.onnx", b"weights")
        js.lmLite.loadModelToCache.side_effect = RuntimeError("cache quota exceeded")

        with pytest.raises(RuntimeError, match="cache quota exceeded"):
            await LLM.create(generator_model="gpt2@main", local_models_path=str(tmp_path))

        js.lmLite.loadModelToCache.assert_awaited_once()
        assert bar.n == 2
        instance.init.assert_not_awaited()
        bar.__exit__.assert_called_once()

    async def test_empty_zip(self, loading, tmp_path):
        from lmlite import LLM

        bar, _, js, _ = loading
        with zipfile.ZipFile(tmp_path / "model-Xenova-gpt2@main.zip", "w") as archive:
            archive.writestr("Xenova/gpt2/", b"")

        await LLM.create(generator_model="gpt2@main", local_models_path=str(tmp_path))

        bar.reset.assert_any_call(total=0)
        bar.update.assert_called_once_with(1)
        js.lmLite.loadModelToCache.assert_not_awaited()

    @pytest.mark.parametrize("cached", [False, True])
    async def test_no_zip_read_progress_without_injection(self, loading, tmp_path, cached):
        from lmlite import LLM

        bar, _, js, _ = loading
        js.lmLite.isModelCached.return_value = cached
        if cached:
            with zipfile.ZipFile(tmp_path / "model-Xenova-gpt2@main.zip", "w") as archive:
                archive.writestr("config.json", "{}")

        await LLM.create(generator_model="gpt2@main", local_models_path=str(tmp_path))

        bar.update.assert_called_once_with(1)
        js.lmLite.loadModelToCache.assert_not_awaited()
        assert not any(call.args[0].startswith("Reading") for call in bar.set_description.call_args_list)

    async def test_progress_callback_receives_loading_events(self, loading):
        from lmlite import LLM

        _, instance, js, _ = loading

        async def init(callback):
            callback("Xenova/gpt2", "loading", "", 0, 0)
            callback("Xenova/gpt2", "progress", "onnx/model.onnx", 5, 10)
            callback("", "ready", "", 0, 0)

        instance.init.side_effect = init
        events = []
        await LLM.create(generator_model="gpt2", auto_detect_local=False, progress_callback=events.append)

        assert [event["status"] for event in events] == ["loading", "progress", "ready"]
        assert events[1] == {
            "model": "Xenova/gpt2",
            "status": "progress",
            "file": "onnx/model.onnx",
            "loaded": 5,
            "total": 10,
        }
        assert "progress_callback" not in js.lmLite.createLLM.call_args.args[0]

    async def test_progress_callback_receives_errors(self, loading):
        from lmlite import LLM

        _, instance, _, _ = loading
        instance.init.side_effect = RuntimeError("Failed to load text-generation model")
        events = []
        with pytest.raises(RuntimeError, match="Failed to load"):
            await LLM.create(generator_model="gpt2", auto_detect_local=False, progress_callback=events.append)

        assert events[-1]["status"] == "error"
        assert "Failed to load" in events[-1]["error"]


class TestModelInfo:
    @pytest.fixture
    def js(self, monkeypatch):
        import sys

        js = types.ModuleType("js")

        js.lmLite = types.SimpleNamespace()
        monkeypatch.setitem(sys.modules, "js", js)
        monkeypatch.setattr("lmlite.llm.ensure_runtime", MagicMock())
        return js

    async def test_model_info_returns_js_metadata(self, js):
        from lmlite import LLM, ModelInfo

        payload = {
            "model_id": "HuggingFaceTB/SmolLM2-360M-Instruct",
            "revision": "main",
            "task": "text-generation",
            "backend": "wasm",
            "dtype": "q4",
            "cache_status": "partial",
            "download_bytes": None,
            "files": [{"name": "config.json", "size": 10, "cached": True}],
            "error": None,
        }
        js.lmLite.getModelInfo = AsyncMock(return_value=payload)

        info = await LLM.model_info("HuggingFaceTB/SmolLM2-360M-Instruct", backend="wasm", dtype="q4")

        assert info == ModelInfo(**payload)
        assert info.download_bytes is None
        js.lmLite.getModelInfo.assert_awaited_once_with("HuggingFaceTB/SmolLM2-360M-Instruct", None, "wasm", "q4")

    async def test_model_info_requires_current_runtime(self, js):
        from lmlite import LLM

        with pytest.raises(RuntimeError, match="getModelInfo"):
            await LLM.model_info("gpt2")

    @pytest.mark.parametrize("tag", ["feature-extraction", None])
    async def test_model_task_returns_hub_tag(self, js, tag):
        from lmlite import LLM

        js.lmLite.getModelTask = AsyncMock(return_value=tag)

        assert await LLM.model_task("all-MiniLM-L6-v2") == tag
        js.lmLite.getModelTask.assert_awaited_once_with("all-MiniLM-L6-v2")


class TestModelZipNaming:
    async def test_model_zip_name_includes_default_revision(self):
        from lmlite.llm import model_zip_name

        with patch(
            "lmlite.llm._get_latest_revision",
            new_callable=AsyncMock,
            return_value="bf2c7f02e0b826c60d03af341171bde20893da66",
        ):
            assert await model_zip_name("gpt2") == "model-Xenova-gpt2@bf2c7f.zip"

    async def test_model_zip_name_keeps_explicit_revision(self):
        from lmlite.llm import model_zip_name

        assert await model_zip_name("acme/gpt2@v1") == "model-acme-gpt2@v1.zip"

    async def test_model_zip_name_shortens_full_sha_revision(self):
        from lmlite.llm import model_zip_name

        sha = "1234567890abcdef1234567890abcdef12345678"
        assert await model_zip_name(f"acme/gpt2@{sha}") == "model-acme-gpt2@123456.zip"


class TestLLMSimilaritySearchWithK:
    """Test similarity search with k parameter."""

    async def test_similarity_search_with_k(self):
        """Test similarity search returns top k results."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        # Mock embeddings - need more for multiple calls
        mock_js_instance.embed = AsyncMock(
            side_effect=[
                # First search call
                [1.0, 0.0],  # query
                [1.0, 0.0],  # doc1 - perfect match
                [0.0, 1.0],  # doc2 - orthogonal
                [0.7, 0.7],  # doc3 - partial match
                # Second search call
                [1.0, 0.0],  # query
                [1.0, 0.0],  # doc1
                [0.0, 1.0],  # doc2
                [0.7, 0.7],  # doc3
            ]
        )

        llm = LLM(mock_js_instance)

        # Mock the JS cosineSimilarityBatch function
        mock_similarities = [1.0, 0.0, 0.7]

        import sys

        mock_js_module = MagicMock()
        mock_js_module.lmLite.cosineSimilarityBatch = MagicMock(return_value=mock_similarities)
        sys.modules["js"] = mock_js_module

        # Mock to_js to just return the input (simulating successful conversion)
        mock_pyodide_module = MagicMock()
        mock_pyodide_module.ffi = MagicMock()
        mock_pyodide_module.ffi.to_js = MagicMock(side_effect=lambda x: x)

        try:
            # Mock pyodide.ffi module
            sys.modules["pyodide"] = mock_pyodide_module
            sys.modules["pyodide.ffi"] = mock_pyodide_module.ffi

            # Test with k=2
            results = await llm.similarity_search("query", ["doc1", "doc2", "doc3"], k=2)

            # Should only return top 2
            assert len(results) == 2
            assert results[0] == ("doc1", 1.0)
            assert results[1] == ("doc3", 0.7)

            # Test with k=1
            results = await llm.similarity_search("query", ["doc1", "doc2", "doc3"], k=1)
            assert len(results) == 1
            assert results[0] == ("doc1", 1.0)

        finally:
            if "js" in sys.modules:
                del sys.modules["js"]
            if "pyodide" in sys.modules:
                del sys.modules["pyodide"]
            if "pyodide.ffi" in sys.modules:
                del sys.modules["pyodide.ffi"]


class TestConversionFallbacks:
    """Test fallback dict conversion when pyodide/pjs not available."""

    async def test_create_with_pjs_fallback(self):
        """Test LLM.create with pjs module (xeus-python fallback)."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.init = AsyncMock()

        import sys

        mock_js_module = types.ModuleType("js")

        mock_js_module.lmLite = types.SimpleNamespace()
        mock_js_module.lmLite.createLLM = MagicMock(return_value=mock_js_instance)

        # Remove pyodide, use pjs
        pyodide_backup = sys.modules.pop("pyodide", None)
        pyodide_ffi_backup = sys.modules.pop("pyodide.ffi", None)

        mock_pjs_module = MagicMock()
        mock_pjs_module.to_js = MagicMock(side_effect=lambda x: x)

        sys.modules["js"] = mock_js_module
        sys.modules["pjs"] = mock_pjs_module

        try:
            with patch("lmlite.llm.ensure_runtime"):
                llm = await LLM.create(generator_model="gpt2")
                assert llm is not None
                mock_js_module.lmLite.createLLM.assert_called_once()
        finally:
            if "js" in sys.modules:
                del sys.modules["js"]
            if "pjs" in sys.modules:
                del sys.modules["pjs"]
            if pyodide_backup:
                sys.modules["pyodide"] = pyodide_backup
            if pyodide_ffi_backup:
                sys.modules["pyodide.ffi"] = pyodide_ffi_backup

    async def test_create_with_no_conversion(self):
        """Test LLM.create when no conversion modules available (fallback to dict)."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.init = AsyncMock()

        import sys

        mock_js_module = types.ModuleType("js")

        mock_js_module.lmLite = types.SimpleNamespace()
        mock_js_module.lmLite.createLLM = MagicMock(return_value=mock_js_instance)

        # Backup and remove modules
        pyodide_backup = sys.modules.pop("pyodide", None)
        pyodide_ffi_backup = sys.modules.pop("pyodide.ffi", None)
        pjs_backup = sys.modules.pop("pjs", None)

        sys.modules["js"] = mock_js_module

        try:
            with patch("lmlite.llm.ensure_runtime"):
                llm = await LLM.create(
                    generator_model="gpt2",
                    classification_model="toxic-bert",
                    backend="webgpu",
                )
                assert llm is not None
                # Verify createLLM was called with the config dict directly
                config = mock_js_module.lmLite.createLLM.call_args[0][0]
                assert isinstance(config, dict)
                assert config["backend"] == "webgpu"
                assert config["classification_model"] == "toxic-bert"
        finally:
            if "js" in sys.modules:
                del sys.modules["js"]
            if pyodide_backup:
                sys.modules["pyodide"] = pyodide_backup
            if pyodide_ffi_backup:
                sys.modules["pyodide.ffi"] = pyodide_ffi_backup
            if pjs_backup:
                sys.modules["pjs"] = pjs_backup

    async def test_similarity_search_with_pjs_fallback(self):
        """Test similarity search with pjs module fallback."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        # Mock embeddings
        mock_js_instance.embed = AsyncMock(
            side_effect=[
                [1.0, 0.0],  # query
                [1.0, 0.0],  # doc1
                [0.0, 1.0],  # doc2
            ]
        )

        llm = LLM(mock_js_instance)

        import sys

        mock_js_module = MagicMock()
        mock_js_module.lmLite.cosineSimilarityBatch = MagicMock(return_value=[1.0, 0.0])

        # Remove pyodide, use pjs
        pyodide_backup = sys.modules.pop("pyodide", None)
        pyodide_ffi_backup = sys.modules.pop("pyodide.ffi", None)

        mock_pjs_module = MagicMock()
        mock_pjs_module.to_js = MagicMock(side_effect=lambda x: x)

        sys.modules["js"] = mock_js_module
        sys.modules["pjs"] = mock_pjs_module

        try:
            results = await llm.similarity_search("query", ["doc1", "doc2"])
            assert len(results) == 2
            assert results[0] == ("doc1", 1.0)
            assert results[1] == ("doc2", 0.0)
            mock_pjs_module.to_js.assert_called()
        finally:
            if "js" in sys.modules:
                del sys.modules["js"]
            if "pjs" in sys.modules:
                del sys.modules["pjs"]
            if pyodide_backup:
                sys.modules["pyodide"] = pyodide_backup
            if pyodide_ffi_backup:
                sys.modules["pyodide.ffi"] = pyodide_ffi_backup


class TestCreateWithPyodide:
    """Test LLM creation with Pyodide conversion."""

    async def test_create_with_pyodide_to_js(self):
        """Test LLM.create with pyodide.ffi.to_js conversion."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        mock_js_instance.init = AsyncMock()

        import sys

        mock_js_module = types.ModuleType("js")

        mock_js_module.lmLite = types.SimpleNamespace()
        mock_js_module.lmLite.createLLM = MagicMock(return_value=mock_js_instance)

        mock_pyodide_ffi = MagicMock()
        mock_pyodide_ffi.to_js = MagicMock(side_effect=lambda x: x)

        sys.modules["js"] = mock_js_module
        sys.modules["pyodide.ffi"] = mock_pyodide_ffi

        try:
            with patch("lmlite.llm.ensure_runtime"):
                llm = await LLM.create(generator_model="gpt2")
                assert llm is not None
                mock_pyodide_ffi.to_js.assert_called_once()
        finally:
            if "js" in sys.modules:
                del sys.modules["js"]
            if "pyodide.ffi" in sys.modules:
                del sys.modules["pyodide.ffi"]


class TestSimilaritySearchNoConversion:
    """Test similarity search with no conversion modules."""

    async def test_similarity_search_no_conversion_modules(self):
        """Test similarity search fallback when no conversion modules available."""
        from lmlite import LLM

        mock_js_instance = MagicMock()
        # Mock embeddings
        mock_js_instance.embed = AsyncMock(
            side_effect=[
                [1.0, 0.0],  # query
                [1.0, 0.0],  # doc1
                [0.0, 1.0],  # doc2
            ]
        )

        llm = LLM(mock_js_instance)

        import sys

        mock_js_module = MagicMock()
        mock_js_module.lmLite.cosineSimilarityBatch = MagicMock(return_value=[1.0, 0.0])

        # Remove all conversion modules
        pyodide_backup = sys.modules.pop("pyodide", None)
        pyodide_ffi_backup = sys.modules.pop("pyodide.ffi", None)
        pjs_backup = sys.modules.pop("pjs", None)

        sys.modules["js"] = mock_js_module

        try:
            results = await llm.similarity_search("query", ["doc1", "doc2"])
            assert len(results) == 2
            assert results[0] == ("doc1", 1.0)
            assert results[1] == ("doc2", 0.0)
        finally:
            if "js" in sys.modules:
                del sys.modules["js"]
            if pyodide_backup:
                sys.modules["pyodide"] = pyodide_backup
            if pyodide_ffi_backup:
                sys.modules["pyodide.ffi"] = pyodide_ffi_backup
            if pjs_backup:
                sys.modules["pjs"] = pjs_backup


class TestLLMPipeline:
    """Test the generic Transformers.js task pipeline interface."""

    async def test_pipeline_loads_task_model_with_shared_options(self, monkeypatch):
        import sys

        from lmlite import LLM, Pipeline

        mock_js_instance = MagicMock()
        mock_js_instance.init = AsyncMock()
        mock_js_module = types.ModuleType("js")
        mock_js_module.lmLite = types.SimpleNamespace()
        mock_js_module.lmLite.createLLM = MagicMock(return_value=mock_js_instance)
        mock_js_module.lmLite.isModelCached = AsyncMock(return_value=True)
        monkeypatch.setitem(sys.modules, "js", mock_js_module)

        with patch("lmlite.llm.ensure_runtime"):
            pipe = await LLM.pipeline("translation", model="opus-mt-en-fr", backend="webgpu", dtype="fp32")

        assert isinstance(pipe, Pipeline)
        assert (pipe.task, pipe.model) == ("translation", "opus-mt-en-fr")
        config = mock_js_module.lmLite.createLLM.call_args.args[0]
        assert config["task"] == "translation"
        assert config["task_model"] == "opus-mt-en-fr"
        assert config["backend"] == "webgpu"
        assert config["dtype"] == "fp32"
        mock_js_module.lmLite.isModelCached.assert_awaited_once_with("opus-mt-en-fr")
        assert pipe.llm._models == ["opus-mt-en-fr"]

    async def test_unsupported_task_lists_supported_tasks(self):
        from lmlite import LLM

        with pytest.raises(ValueError, match="Unsupported task 'image-to-text'.*zero-shot-classification"):
            await LLM.pipeline("image-to-text", model="m")
        with pytest.raises(ValueError, match="requires a model"):
            await LLM.pipeline("translation", model=None)
        with pytest.raises(TypeError, match="generator_model"):
            await LLM.pipeline("translation", model="m", generator_model="gpt2")

    async def test_zero_shot_forwards_labels_as_json_options(self):
        from lmlite import Pipeline

        llm = MagicMock()
        output = {"sequence": "text", "labels": ["technology", "sports"], "scores": [0.9, 0.1]}
        llm.js.runTask = AsyncMock(return_value=output)
        pipe = Pipeline(llm, "zero-shot-classification", "m")

        result = await pipe("text", candidate_labels=["technology", "sports"], multi_label=False)

        assert result == output
        inputs, options = llm.js.runTask.await_args.args
        assert inputs == "text"
        assert json.loads(options) == {"candidate_labels": ["technology", "sports"], "multi_label": False}

    async def test_text_batches_are_sent_as_lists(self):
        from lmlite import Pipeline

        llm = MagicMock()
        llm.js.runTask = AsyncMock(return_value=[{"translation_text": "a"}, {"translation_text": "b"}])
        result = await Pipeline(llm, "translation", "m")(("Hello", "World"))

        assert result == [{"translation_text": "a"}, {"translation_text": "b"}]
        assert llm.js.runTask.await_args.args == (["Hello", "World"], "{}")

    async def test_dispose_and_export_delegate_to_llm(self):
        from lmlite import Pipeline

        llm = MagicMock()
        llm.dispose = AsyncMock()
        llm.export_models = AsyncMock()
        pipe = Pipeline(llm, "translation", "m")
        await pipe.dispose()
        await pipe.export_models()
        llm.dispose.assert_awaited_once_with()
        llm.export_models.assert_awaited_once_with()
        assert repr(pipe) == "Pipeline(task='translation', model='m')"


def _wav_bytes(samples, rate=16000, channels=1, width=2):
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        scale = 2 ** (8 * width - 1) - 1
        wav.writeframes(b"".join(int(s * scale).to_bytes(width, "little", signed=True) for s in samples))
    return buffer.getvalue()


class TestAudioInput:
    """Test speech recognition input normalization."""

    async def test_asr_decodes_wav_bytes_to_samples(self):
        from lmlite import Pipeline

        llm = MagicMock()
        llm.js.runTask = AsyncMock(return_value={"text": "hello"})
        result = await Pipeline(llm, "automatic-speech-recognition", "m")(
            _wav_bytes([0.0, 0.5, -0.5]), return_timestamps=True
        )

        assert result == {"text": "hello"}
        samples, options = llm.js.runTask.await_args.args
        assert samples == pytest.approx([0.0, 0.5, -0.5], abs=1e-4)
        assert json.loads(options) == {"return_timestamps": True}

    def test_wav_file_stereo_and_resampling(self, tmp_path):
        from lmlite.llm import _audio_input

        path = tmp_path / "a.wav"
        # Stereo channels average to mono; 8 kHz is upsampled to 16 kHz.
        path.write_bytes(_wav_bytes([0.5, -0.5, 1.0, 0.0], rate=8000, channels=2))
        assert _audio_input(str(path)) == pytest.approx([0.0, 1 / 6, 1 / 3, 0.5], abs=1e-4)

    @pytest.mark.parametrize("width", [1, 3, 4])
    def test_wav_sample_widths(self, width):
        from lmlite.llm import _read_wav

        if width == 1:
            import io
            import wave

            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(1)
                wav.setframerate(16000)
                wav.writeframes(bytes([128, 192, 64]))
            data = buffer.getvalue()
        else:
            data = _wav_bytes([0.0, 0.5, -0.5], width=width)
        assert _read_wav(data) == pytest.approx([0.0, 0.5, -0.5], abs=1e-2)

    def test_urls_and_samples_pass_through(self):
        from lmlite.llm import _audio_input

        assert _audio_input("https://example.test/a.wav") == "https://example.test/a.wav"
        assert _audio_input((0, 1)) == [0.0, 1.0]

    def test_invalid_audio_reports_supported_inputs(self):
        from lmlite.llm import _audio_input

        with pytest.raises(FileNotFoundError, match="PCM WAV"):
            _audio_input("missing.wav")
        with pytest.raises(ValueError, match="PCM WAV"):
            _audio_input(b"not a wav file")


class TestTokenizer:
    @pytest.fixture
    def js(self, monkeypatch):
        import sys

        js = types.ModuleType("js")
        handle = types.SimpleNamespace(
            vocab_size=3,
            encode=MagicMock(side_effect=lambda text, special: json.dumps([len(text)] + ([1] if special else []))),
            decode=MagicMock(return_value="ab"),
        )
        js.lmLite = types.SimpleNamespace(loadTokenizer=AsyncMock(return_value=handle))
        monkeypatch.setitem(sys.modules, "js", js)
        monkeypatch.setattr("lmlite.llm.ensure_runtime", MagicMock())
        return js

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"model": "org/m", "tokenizer_json": "t.json"}, {"model": "org/m", "tokenizer_config": "c.json"}],
    )
    async def test_argument_rules_fail_before_loading(self, js, kwargs):
        from lmlite import LLM

        with pytest.raises(ValueError):
            await LLM.tokenizer(**kwargs)
        js.lmLite.loadTokenizer.assert_not_awaited()

    async def test_encode_is_sync_and_returns_plain_ints(self, js):
        from lmlite import LLM, Tokenizer

        tok = await LLM.tokenizer(tokenizer_json="files/models/org/m/tokenizer.json")

        js.lmLite.loadTokenizer.assert_awaited_once_with(None, "files/models/org/m/tokenizer.json", None, None)
        assert isinstance(tok, Tokenizer) and tok.vocab_size == 3
        ids = tok.encode("abc")
        assert ids == [3] and all(type(i) is int for i in ids)
        assert tok.encode("abc", add_special_tokens=True) == [3, 1]
        assert tok.decode([1, 2]) == "ab"
        tok._js.decode.assert_called_once_with("[1, 2]")
        with pytest.raises(TypeError):
            tok.encode(b"abc")

    async def test_close_is_idempotent_and_blocks_use(self, js):
        from lmlite import LLM

        tok = await LLM.tokenizer("org/m")
        tok.close()
        tok.close()
        with pytest.raises(RuntimeError, match="closed"):
            tok.encode("a")

    async def test_bridge_without_sync_results_fails_at_load_time(self, js):
        from lmlite import LLM

        js.lmLite.loadTokenizer.return_value.encode = MagicMock(return_value=object())  # e.g. a Promise proxy
        with pytest.raises(RuntimeError, match="synchronously"):
            await LLM.tokenizer("org/m")
