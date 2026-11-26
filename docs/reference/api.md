# API reference

Generated from the docstrings. For how to use these, see the [guides](../guides/index.md); for what each task accepts and returns, the [pipeline task table](pipeline-tasks.md).

## Functions

```{eval-rst}
.. autofunction:: lmlite.pipeline

.. autofunction:: lmlite.tokenizer

.. autofunction:: lmlite.model_info

.. autofunction:: lmlite.model_task

.. autofunction:: lmlite.hosted_base

.. autofunction:: lmlite.cached_models

.. autofunction:: lmlite.clear_cache
```

## Language models

```{eval-rst}
.. autoclass:: lmlite.LLM
   :members:
   :inherited-members:
```

## Embeddings and classification

```{eval-rst}
.. autoclass:: lmlite.Embedder
   :members:
   :inherited-members:

.. autoclass:: lmlite.Classifier
   :members:
   :inherited-members:
```

## Task pipelines

```{eval-rst}
.. autoclass:: lmlite.Pipeline
   :members:
   :special-members: __call__

.. autoclass:: lmlite.Speech
   :members:

.. autoclass:: lmlite.Tokenizer
   :members:

.. autoclass:: lmlite.ModelInfo

.. autodata:: lmlite.SUPPORTED_TASKS
   :no-value:
```

## Bare ONNX models

```{eval-rst}
.. autoclass:: lmlite.OnnxModel
   :members:
```

## Notebook widgets

```{eval-rst}
.. autoclass:: lmlite.audio.AudioRecorder
   :members: audio, wait, request, clear

.. autoclass:: lmlite.camera.CameraCapture
   :members: image, wait, request, clear

.. autoexception:: lmlite.camera.CaptureCancelled

.. autoclass:: lmlite.widgets.CacheManager
   :members:
```
