import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import './llm.js';

vi.mock('https://cdn.jsdelivr.net/npm/@huggingface/transformers', () => ({
  pipeline: vi.fn(),
  env: {},
}));

const {
  BrowserLLM,
  createLLM,
  cosineSimilarity,
  cosineSimilarityBatch,
  isModelCached,
  loadModelToCache,
  clearModelCache,
  getHostedModelBase,
} = globalThis;

describe('Hosted model discovery', () => {
  afterEach(() => vi.unstubAllGlobals());

  it.each([
    ['https://example.test/demo/extensions/kernel/worker.js', 'https://example.test/demo/files/models/'],
    ['https://example.test/demo/pyodide/extensions/kernel/worker.js', 'https://example.test/demo/pyodide/files/models/'],
    ['https://example.test/demo/lab/index.html', 'https://example.test/demo/files/models/'],
  ])('finds the models directory from %s', async (location, expected) => {
    vi.stubGlobal('location', new URL(location));
    const fetch = vi.fn(async () => new Response(JSON.stringify({ models: ['Xenova/gpt2'] })));
    vi.stubGlobal('fetch', fetch);

    expect(await getHostedModelBase('gpt2')).toBe(expected);
    expect(String(fetch.mock.calls[0][0])).toBe(`${expected}manifest.json`);
  });

  it('supports an explicit base URL without a discoverable worker URL', async () => {
    vi.stubGlobal('location', new URL('blob:https://example.test/worker'));
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ models: ['Xenova/gpt2'] }))));

    expect(await getHostedModelBase('gpt2@main', 'https://example.test/models')).toBe('https://example.test/models/');
  });

  it('does not map an explicit Hub revision to an unversioned hosted tree', async () => {
    const fetch = vi.fn();
    vi.stubGlobal('fetch', fetch);
    expect(await getHostedModelBase('gpt2@v1', 'https://example.test/models/')).toBeNull();
    expect(fetch).not.toHaveBeenCalled();
  });

  it('falls back when the auto-discovered manifest does not exist', async () => {
    vi.stubGlobal('location', new URL('https://example.test/lab/index.html'));
    vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 404 })));
    expect(await getHostedModelBase('gpt2')).toBeNull();
  });

  it('does not ignore failure at an explicit manifest URL', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 404 })));
    await expect(getHostedModelBase('gpt2', 'https://example.test/models/')).rejects.toThrow('manifest request failed');
  });

  it('falls back for models not listed in the manifest', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ models: ['org/other'] }))));
    expect(await getHostedModelBase('gpt2', 'https://example.test/models/')).toBeNull();
  });

  it.each([{}, { models: ['../model'] }, { models: [5] }])('rejects invalid manifests %j', async (manifest) => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(manifest))));
    await expect(getHostedModelBase('gpt2', 'https://example.test/models/')).rejects.toThrow('Invalid hosted model manifest');
  });
});

describe('BrowserLLM Constructor', () => {
  it('should initialize with minimal config', () => {
    const config = { generator_model: 'gpt2' };
    const llm = new BrowserLLM(config);

    expect(llm.generatorModel).toBe('gpt2');
    expect(llm.embeddingModel).toBeUndefined();
    expect(llm.backend).toBe('wasm');
  });

  it('should set classification_model when provided', () => {
    const config = {
      classification_model: 'distilbert-base-uncased-finetuned-sst-2-english'
    };
    const llm = new BrowserLLM(config);

    expect(llm.classificationModel).toBe('distilbert-base-uncased-finetuned-sst-2-english');
    expect(llm.generatorModel).toBeUndefined();
  });

  it('should normalize backend to lowercase', () => {
    const config = { backend: 'WEBGPU', generator_model: 'gpt2' };
    const llm = new BrowserLLM(config);

    expect(llm.backend).toBe('webgpu');
  });

  it('should set default values for generation parameters', () => {
    const config = { generator_model: 'gpt2' };
    const llm = new BrowserLLM(config);

    expect(llm.max_new_tokens).toBe(50);
    expect(llm.temperature).toBe(0.7);
    expect(llm.top_k).toBe(50);
    expect(llm.do_sample).toBe(true);
  });

  it('should store explicit dtype options', () => {
    const config = {
      generator_model: 'gpt2',
      dtype: 'fp32',
      dtype_fallbacks: ['fp32', 'int8'],
    };
    const llm = new BrowserLLM(config);

    expect(llm.dtype).toBe('fp32');
    expect(llm.dtypeFallbacks).toEqual(['fp32', 'int8']);
  });

  it('should allow overriding default parameters', () => {
    const config = {
      generator_model: 'gpt2',
      max_new_tokens: 100,
      temperature: 0.5,
      top_k: 30
    };
    const llm = new BrowserLLM(config);

    expect(llm.max_new_tokens).toBe(100);
    expect(llm.temperature).toBe(0.5);
    expect(llm.top_k).toBe(30);
  });
});

describe('BrowserLLM dtype handling', () => {
  it('should honor explicit dtype override', async () => {
    const llm = new BrowserLLM({ generator_model: 'gpt2', dtype: 'fp32' });
    const pipelineFn = vi.fn(async () => 'pipeline');

    await llm._loadPipelineWithDtypeFallback(
      pipelineFn,
      'text-generation',
      'Xenova/gpt2',
      { device: 'wasm' },
      ['q8', 'int8', 'fp32']
    );

    expect(pipelineFn).toHaveBeenCalledTimes(1);
    expect(pipelineFn).toHaveBeenCalledWith(
      'text-generation',
      'Xenova/gpt2',
      { device: 'wasm', dtype: 'fp32' }
    );
  });

  it('should use configured dtype fallback order', async () => {
    const llm = new BrowserLLM({
      generator_model: 'gpt2',
      dtype_fallbacks: ['int8', 'fp32'],
    });
    const pipelineFn = vi
      .fn()
      .mockRejectedValueOnce(new Error("Can't create a session. ERROR_CODE: 1"))
      .mockResolvedValueOnce('pipeline');

    await llm._loadPipelineWithDtypeFallback(
      pipelineFn,
      'text-generation',
      'Xenova/gpt2',
      { device: 'wasm' },
      ['q8', 'fp16']
    );

    expect(pipelineFn).toHaveBeenNthCalledWith(
      1,
      'text-generation',
      'Xenova/gpt2',
      { device: 'wasm', dtype: 'int8' }
    );
    expect(pipelineFn).toHaveBeenNthCalledWith(
      2,
      'text-generation',
      'Xenova/gpt2',
      { device: 'wasm', dtype: 'fp32' }
    );
  });
});

describe('BrowserLLM initialization progress', () => {
  it('uses individual hosted files for selected models and preserves remote fallback for others', async () => {
    const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers');
    pipeline.mockImplementation(async (_task, _model, options) => {
      options.progress_callback({ status: 'progress', file: 'model.onnx', loaded: 1, total: 2 });
      return vi.fn(async () => []);
    });
    const llm = new BrowserLLM({
      generator_model: 'gpt2',
      embedding_model: 'org/embedding',
      hosted_model_base: 'https://example.test/demo/models/',
      hosted_models: ['Xenova/gpt2'],
      dtype: 'fp32',
    });
    pipeline.mockClear();

    await llm.init(vi.fn());

    expect(env.allowLocalModels).toBe(true);
    expect(env.localModelPath).toBe('https://example.test/demo/models/');
    expect(pipeline).toHaveBeenNthCalledWith(1, 'text-generation', 'Xenova/gpt2',
      expect.objectContaining({ local_files_only: true, dtype: 'fp32' }));
    expect(pipeline.mock.calls[1][2].local_files_only).toBeUndefined();

    await new BrowserLLM({}).init();
    expect(env.allowLocalModels).toBe(false);
  });

  it('forwards download progress and warm-up status to the notebook callback', async () => {
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers');
    const generator = vi.fn(async () => []);
    pipeline.mockImplementation(async (_task, _model, options) => {
      options.progress_callback({
        status: 'progress', file: 'onnx/model.onnx', loaded: 25, total: 100,
      });
      options.progress_callback({ status: 'ready' });
      return generator;
    });
    const callback = vi.fn();
    const llm = new BrowserLLM({ generator_model: 'gpt2' });

    await llm.init(callback);

    expect(callback).toHaveBeenCalledWith(
      'Xenova/gpt2', 'progress', 'onnx/model.onnx', 25, 100
    );
    expect(callback).toHaveBeenCalledWith('Xenova/gpt2', 'initializing', '', 0, 0);
    expect(callback).toHaveBeenCalledWith('Xenova/gpt2', 'warming up', '', 0, 0);
    expect(generator).toHaveBeenCalledWith('test', { max_new_tokens: 1 });
    expect(callback).toHaveBeenLastCalledWith('', 'ready', '', 0, 0);
  });

  it.each(['embedding_model', 'classification_model'])(
    'forwards progress for %s',
    async (modelKey) => {
      const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers');
      pipeline.mockImplementation(async (_task, _model, options) => {
        options.progress_callback({
          status: 'progress', file: 'model.onnx', loaded: 5, total: 10,
        });
        return vi.fn();
      });
      const callback = vi.fn();

      await new BrowserLLM({ [modelKey]: 'test-model' }).init(callback);

      expect(callback).toHaveBeenCalledWith('Xenova/test-model', 'progress', 'model.onnx', 5, 10);
    }
  );

  it('still initializes without a progress callback', async () => {
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers');
    pipeline.mockImplementation(async (_task, _model, options) => {
      options.progress_callback({ status: 'downloading', file: 'model.onnx' });
      return vi.fn(async () => []);
    });

    await expect(new BrowserLLM({ generator_model: 'gpt2' }).init()).resolves.toBeUndefined();
  });
});

describe('BrowserLLM embedding dtype pinning', () => {
  it.each([
    ['wasm', 'webgpu', null],
    ['WEBGPU', 'wasm', 'q8'],
  ])('loads embeddings on %s independently of backend=%s', async (embedding_backend, backend, embedding_dtype) => {
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers');
    pipeline.mockImplementation(async () => vi.fn(async () => []));
    pipeline.mockClear();
    const llm = new BrowserLLM({
      generator_model: 'gpt2',
      embedding_model: 'org/embedding',
      classification_model: 'org/classifier',
      backend,
      embedding_backend,
      embedding_dtype,
      dtype: 'fp32',
      hosted_model_base: 'https://example.test/models/',
      hosted_models: ['org/embedding'],
    });

    await llm.init();

    const embeddingCall = pipeline.mock.calls.find(c => c[0] === 'feature-extraction');
    expect(embeddingCall[2]).toEqual(expect.objectContaining({
      device: embedding_backend.toLowerCase(),
      local_files_only: true,
      progress_callback: expect.any(Function),
    }));
    expect(embeddingCall[2].dtype).toBe(embedding_dtype ?? undefined);
    for (const call of pipeline.mock.calls.filter(c => c[0] !== 'feature-extraction')) {
      expect(call[2]).toEqual(expect.objectContaining({ device: backend, dtype: 'fp32' }));
    }
  });

  it.each(['cpu', ''])('rejects invalid embedding backend %j before loading models', async (embedding_backend) => {
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers');
    pipeline.mockClear();

    await expect(new BrowserLLM({ embedding_model: 'org/embedding', embedding_backend }).init())
      .rejects.toThrow('Invalid embedding_backend');
    expect(pipeline).not.toHaveBeenCalled();
  });

  it('does not fall back to WASM when WebGPU embedding loading fails', async () => {
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers');
    pipeline.mockRejectedValue(new Error('WebGPU unavailable'));
    pipeline.mockClear();

    await expect(new BrowserLLM({
      embedding_model: 'org/embedding', embedding_backend: 'webgpu',
    }).init()).rejects.toThrow("Failed to load embedding model 'org/embedding': WebGPU unavailable");
    expect(pipeline).toHaveBeenCalledTimes(1);
    expect(pipeline.mock.calls[0][2].device).toBe('webgpu');
  });

  it('surfaces a pinned dtype failure without retrying', async () => {
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers');
    pipeline.mockRejectedValue(new Error('Could not locate file model_quantized.onnx'));
    pipeline.mockClear();

    await expect(new BrowserLLM({
      embedding_model: 'org/embedding',
      embedding_dtype: 'q8',
      dtype_fallbacks: ['fp32'],
    }).init()).rejects.toThrow(
      "Failed to load embedding model 'org/embedding': Could not locate file model_quantized.onnx"
    );

    expect(pipeline).toHaveBeenCalledTimes(1);
    expect(pipeline).toHaveBeenCalledWith(
      'feature-extraction', 'org/embedding', expect.objectContaining({ dtype: 'q8' })
    );
  });

  it('loads feature-extraction with no dtype by default', async () => {
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers');
    pipeline.mockImplementation(async () => vi.fn(async () => ({ data: [0] })));
    pipeline.mockClear();

    await new BrowserLLM({ embedding_model: 'org/embedding' }).init();

    const call = pipeline.mock.calls.find((c) => c[0] === 'feature-extraction');
    expect(call).toBeDefined();
    expect(call[2].dtype).toBeUndefined();
  });

  it('pins the feature-extraction dtype when embedding_dtype is set', async () => {
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers');
    const embedder = vi.fn(async () => ({ data: [0] }));
    pipeline.mockImplementation(async () => embedder);
    pipeline.mockClear();

    await new BrowserLLM({ embedding_model: 'org/embedding', embedding_dtype: 'q8' }).init();

    const call = pipeline.mock.calls.find((c) => c[0] === 'feature-extraction');
    expect(call).toBeDefined();
    expect(call[2]).toEqual(expect.objectContaining({ dtype: 'q8' }));
    expect(embedder).not.toHaveBeenCalled();
  });

  it('embedding_dtype is NOT overridden by the generator dtype / dtype_fallbacks', async () => {
    // dtype and dtype_fallbacks target the generator/classifier pipelines. They
    // must never leak into the embedding pin — the embedding dtype must stay q8.
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers');
    pipeline.mockImplementation(async () => vi.fn(async () => ({ data: [0] })));
    pipeline.mockClear();

    await new BrowserLLM({
      embedding_model: 'org/embedding',
      embedding_dtype: 'q8',
      dtype: 'fp32',
      dtype_fallbacks: ['int8', 'fp16'],
    }).init();

    const call = pipeline.mock.calls.find((c) => c[0] === 'feature-extraction');
    expect(call).toBeDefined();
    expect(call[2].dtype).toBe('q8');
  });

  it('does not pass a backend/device to the embedding pipeline (preserves prior behavior)', async () => {
    // The embedding pipeline keeps Transformers.js's default device selection;
    // `backend` applies only to the generator/classifier pipelines.
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers');
    pipeline.mockImplementation(async () => vi.fn(async () => ({ data: [0] })));
    pipeline.mockClear();

    await new BrowserLLM({
      embedding_model: 'org/embedding',
      embedding_dtype: 'q8',
      backend: 'webgpu',
    }).init();

    const call = pipeline.mock.calls.find((c) => c[0] === 'feature-extraction');
    expect(call).toBeDefined();
    expect(call[2].device).toBeUndefined();
  });
});

describe('BrowserLLM.dispose()', () => {
  it('releases all loaded pipelines and is safe to repeat', async () => {
    const llm = new BrowserLLM({});
    const pipelines = ['generator', 'embedder', 'classifier'].map(name => {
      const pipeline = { dispose: vi.fn(async () => {}) };
      llm[name] = pipeline;
      return pipeline;
    });
    await llm.dispose();
    await llm.dispose();
    for (const pipeline of pipelines) {
      expect(pipeline.dispose).toHaveBeenCalledTimes(1);
    }
    expect(llm.generator).toBeNull();
    expect(llm.embedder).toBeNull();
    expect(llm.classifier).toBeNull();
    await expect(llm.generate('hello')).rejects.toThrow('Generator model not initialized');
    await expect(llm.embed('hello')).rejects.toThrow('No embedding model configured');
    await expect(llm.classify('hello')).rejects.toThrow('No classification model configured');
  });

  it('attempts every cleanup and retains failed pipelines for retry', async () => {
    const llm = new BrowserLLM({});
    const generator = { dispose: vi.fn().mockRejectedValueOnce(new Error('session release failed'))
      .mockResolvedValueOnce(undefined) };
    const embedder = { dispose: vi.fn(async () => {}) };
    const classifier = { dispose: vi.fn().mockRejectedValueOnce(new Error('GPU release failed'))
      .mockResolvedValueOnce(undefined) };
    Object.assign(llm, { generator, embedder, classifier });

    await expect(llm.dispose()).rejects.toThrow(
      'Failed to dispose generator: session release failed; Failed to dispose classifier: GPU release failed'
    );
    expect(llm.generator).toBe(generator);
    expect(llm.classifier).toBe(classifier);
    expect(llm.embedder).toBeNull();
    expect(embedder.dispose).toHaveBeenCalledTimes(1);
    await llm.dispose();
    expect(llm.generator).toBeNull();
    expect(llm.classifier).toBeNull();
    expect(embedder.dispose).toHaveBeenCalledTimes(1);
  });

  it('does not clear cached downloads', async () => {
    const deleteCache = vi.fn();
    vi.stubGlobal('caches', { delete: deleteCache });
    try {
      await new BrowserLLM({}).dispose();
      expect(deleteCache).not.toHaveBeenCalled();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe('BrowserLLM.classify()', () => {
  let llm;

  beforeEach(() => {
    const config = { classification_model: 'distilbert' };
    llm = new BrowserLLM(config);
  });

  it('should throw error if classifier not initialized', async () => {
    await expect(llm.classify('test')).rejects.toThrow('No classification model configured');
  });

  it('should call classifier with correct parameters', async () => {
    const mockResult = [{ label: 'POSITIVE', score: 0.99 }];
    llm.classifier = vi.fn(async () => mockResult);

    const result = await llm.classify('I love this!');

    expect(llm.classifier).toHaveBeenCalledWith('I love this!');
    expect(result).toEqual(mockResult);
  });

  it('should pass top_k parameter to classifier', async () => {
    const mockResult = [
      { label: 'POSITIVE', score: 0.99 },
      { label: 'NEGATIVE', score: 0.01 }
    ];
    llm.classifier = vi.fn(async () => mockResult);

    const result = await llm.classify('mixed sentiment', 2);

    expect(llm.classifier).toHaveBeenCalledWith('mixed sentiment', { top_k: 2 });
    expect(result).toHaveLength(2);
  });
});

describe('BrowserLLM task pipelines', () => {
  it('loads a supported task with the shared backend, dtype and hosted options', async () => {
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers');
    const taskPipeline = vi.fn();
    pipeline.mockReset();
    pipeline.mockResolvedValue(taskPipeline);
    const llm = new BrowserLLM({
      task: 'translation',
      task_model: 'opus-mt-en-fr',
      backend: 'webgpu',
      dtype: 'fp32',
      hosted_model_base: 'https://example.test/models/',
      hosted_models: ['Xenova/opus-mt-en-fr'],
    });

    await llm.init();

    expect(pipeline).toHaveBeenCalledTimes(1);
    expect(pipeline).toHaveBeenCalledWith('translation', 'Xenova/opus-mt-en-fr',
      expect.objectContaining({ device: 'webgpu', dtype: 'fp32', local_files_only: true }));
    expect(llm.taskPipeline).toBe(taskPipeline);
  });

  it('rejects unsupported tasks and missing models before loading', async () => {
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers');
    pipeline.mockReset();
    await expect(new BrowserLLM({ task: 'image-to-text', task_model: 'm' }).init())
      .rejects.toThrow("Unsupported task 'image-to-text'. Supported pipeline tasks are: zero-shot-classification");
    await expect(new BrowserLLM({ task: 'translation' }).init()).rejects.toThrow("requires a model");
    expect(pipeline).not.toHaveBeenCalled();
  });

  it('reports the task and model when loading fails', async () => {
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers');
    pipeline.mockReset();
    pipeline.mockRejectedValue(new Error('Unsupported model type: foo'));
    await expect(new BrowserLLM({ task: 'translation', task_model: 'org/foo' }).init())
      .rejects.toThrow("Failed to load translation model 'org/foo': Unsupported model type: foo");
  });

  it('passes candidate_labels positionally for zero-shot classification', async () => {
    const llm = new BrowserLLM({ task: 'zero-shot-classification' });
    const output = { sequence: 'text', labels: ['technology', 'sports'], scores: [0.9, 0.1] };
    llm.taskPipeline = vi.fn(async () => output);

    const result = await llm.runTask('text', JSON.stringify({ candidate_labels: ['technology', 'sports'], multi_label: true }));

    expect(llm.taskPipeline).toHaveBeenCalledWith('text', ['technology', 'sports'], { multi_label: true });
    expect(result).toBe(output);
    await expect(llm.runTask('text', '{}')).rejects.toThrow('requires candidate_labels');
  });

  it('unwraps a single translation and keeps batched results', async () => {
    const llm = new BrowserLLM({ task: 'translation' });
    llm.taskPipeline = vi.fn(async (inputs) => [inputs].flat().map(text => ({ translation_text: `fr:${text}` })));

    expect(await llm.runTask('Hello', '{"max_new_tokens": 8}')).toEqual({ translation_text: 'fr:Hello' });
    expect(llm.taskPipeline).toHaveBeenCalledWith('Hello', { max_new_tokens: 8 });
    expect(await llm.runTask(['a', 'b'])).toEqual([{ translation_text: 'fr:a' }, { translation_text: 'fr:b' }]);
  });

  it('converts raw audio samples to Float32Array and keeps URLs', async () => {
    const llm = new BrowserLLM({ task: 'automatic-speech-recognition' });
    llm.taskPipeline = vi.fn(async () => ({ text: 'hello' }));

    await llm.runTask([0, 0.5, -0.5], '{"return_timestamps": true}');
    const [audio, options] = llm.taskPipeline.mock.calls[0];
    expect(audio).toBeInstanceOf(Float32Array);
    expect(Array.from(audio)).toEqual([0, 0.5, -0.5]);
    expect(options).toEqual({ return_timestamps: true });

    vi.stubGlobal('AudioContext', class {});
    await llm.runTask('https://example.test/a.wav');
    expect(llm.taskPipeline.mock.calls[1][0]).toBe('https://example.test/a.wav');
    vi.unstubAllGlobals();
  });

  it('fetches and decodes WAV URLs when AudioContext is unavailable (kernel workers)', async () => {
    // 8 kHz stereo 16-bit PCM: frames (0.5, -0.5) and (1.0, 0.0) average to 0 and 0.5.
    const pcm = [16384, -16384, 32767, 0];
    const buffer = new ArrayBuffer(44 + pcm.length * 2);
    const view = new DataView(buffer);
    const ascii = (offset, text) => [...text].forEach((ch, i) => view.setUint8(offset + i, ch.charCodeAt(0)));
    ascii(0, 'RIFF'); view.setUint32(4, buffer.byteLength - 8, true); ascii(8, 'WAVE');
    ascii(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 2, true);
    view.setUint32(24, 8000, true); view.setUint32(28, 32000, true); view.setUint16(32, 4, true); view.setUint16(34, 16, true);
    ascii(36, 'data'); view.setUint32(40, pcm.length * 2, true);
    pcm.forEach((value, i) => view.setInt16(44 + i * 2, value, true));
    const fetch = vi.fn(async () => new Response(buffer));
    vi.stubGlobal('fetch', fetch);
    const llm = new BrowserLLM({ task: 'automatic-speech-recognition' });
    llm.taskPipeline = vi.fn(async () => ({ text: 'hello' }));

    try {
      await llm.runTask('https://example.test/a.wav');
      expect(fetch).toHaveBeenCalledWith('https://example.test/a.wav');
      const audio = llm.taskPipeline.mock.calls[0][0];
      expect(audio).toBeInstanceOf(Float32Array);
      expect(Array.from(audio).map(x => Number(x.toFixed(3)))).toEqual([0, 0.167, 0.333, 0.5]);

      fetch.mockResolvedValueOnce(new Response('ID3 mp3 data'));
      await expect(llm.runTask('https://example.test/a.mp3')).rejects.toThrow('Audio must be a WAV file');
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('wraps inference errors and rejects calls without a loaded pipeline', async () => {
    const llm = new BrowserLLM({ task: 'translation' });
    await expect(llm.runTask('Hello')).rejects.toThrow('translation pipeline not initialized');
    await expect(new BrowserLLM({}).runTask('Hello')).rejects.toThrow('No pipeline task configured');
    llm.taskPipeline = vi.fn(async () => { throw new Error('out of memory'); });
    await expect(llm.runTask('Hello')).rejects.toThrow('translation failed: out of memory');
  });

  it('disposes the task pipeline', async () => {
    const llm = new BrowserLLM({ task: 'translation' });
    const taskPipeline = { dispose: vi.fn(async () => {}) };
    llm.taskPipeline = taskPipeline;
    await llm.dispose();
    expect(taskPipeline.dispose).toHaveBeenCalledTimes(1);
    expect(llm.taskPipeline).toBeNull();
  });
});

describe('BrowserLLM.generate()', () => {
  let llm;

  beforeEach(() => {
    const config = { generator_model: 'gpt2', max_new_tokens: 100 };
    llm = new BrowserLLM(config);
  });

  it('should throw error if generator not initialized', async () => {
    await expect(llm.generate('test')).rejects.toThrow('Generator model not initialized. Call init() first.');
  });

  it('should call generator with prompt and max_new_tokens', async () => {
    const mockResult = [{ generated_text: 'Hello world' }];
    llm.generator = vi.fn(async () => mockResult);

    const result = await llm.generate('Hello');

    expect(llm.generator).toHaveBeenCalledWith('Hello', { max_new_tokens: 100, return_full_text: true, temperature: 0.7, top_k: 50, do_sample: true });
    expect(result).toBe('Hello world');
  });

  it('should request only newly generated text when return_full_text is false', async () => {
    llm = new BrowserLLM({ generator_model: 'gpt2', return_full_text: false });
    llm.generator = vi.fn(async () => [{ generated_text: 'world' }]);

    expect(await llm.generate('Hello')).toBe('world');
    expect(llm.generator).toHaveBeenCalledWith('Hello', {
      max_new_tokens: 50, return_full_text: false, temperature: 0.7, top_k: 50, do_sample: true
    });
  });
});

describe('BrowserLLM.chat()', () => {
  it('parses JSON messages from either Python browser bridge', async () => {
    const llm = new BrowserLLM({ generator_model: 'org/chat-model' });
    const messages = [{ role: 'user', content: 'Bonjour \u00e9\n"quoted" <|im_start|>' }];
    llm.generator = vi.fn(async (input) => [{
      generated_text: [...input, { role: 'assistant', content: 'Answer' }]
    }]);
    llm.generator.tokenizer = { chat_template: 'template' };
    expect(await llm.chat(JSON.stringify(messages))).toBe('Answer');
    expect(llm.generator.mock.calls[0][0]).toEqual(messages);
  });

  it.each(['{invalid', 'null', '{}', '[]'])('rejects invalid serialized messages %s', async (input) => {
    const llm = new BrowserLLM({});
    llm.generator = vi.fn();
    await expect(llm.chat(input)).rejects.toThrow();
    expect(llm.generator).not.toHaveBeenCalled();
  });

  it('uses structured chat input when the loaded tokenizer has a template', async () => {
    const llm = new BrowserLLM({ generator_model: 'org/chat-model', return_full_text: true });
    const messages = [
      { role: 'SYSTEM', content: 'Be helpful.' },
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi' },
      { role: 'user', content: 'Explain embeddings' },
    ];
    llm.generator = vi.fn(async (input) => [{
      generated_text: [...input, { role: 'assistant', content: ' Embeddings represent meaning. Assistant: is a label. ' }]
    }]);
    llm.generator.tokenizer = { chat_template: 'model-specific template' };

    expect(await llm.chat(messages)).toBe('Embeddings represent meaning. Assistant: is a label.');
    expect(llm.generator).toHaveBeenCalledWith(
      [{ role: 'system', content: 'Be helpful.' }, ...messages.slice(1)],
      { max_new_tokens: 50, return_full_text: false, temperature: 0.7, top_k: 50, do_sample: true }
    );
    expect(llm.return_full_text).toBe(true);
    expect(messages[0].role).toBe('SYSTEM');
  });

  it.each([true, false])('keeps generic formatting without a template, full text=%s', async (return_full_text) => {
    const llm = new BrowserLLM({ generator_model: 'gpt2', return_full_text });
    llm.generator = vi.fn(async () => [{ generated_text: ' Answer with Assistant: inside. ' }]);
    llm.generator.tokenizer = { chat_template: null };

    expect(await llm.chat([
      { role: 'custom', content: 'Instructions' },
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi' },
      { role: 'user', content: 'Continue' },
    ])).toBe('Answer with Assistant: inside.');
    expect(llm.generator.mock.calls[0][0]).toBe(
      'custom: Instructions\nUser: Hello\nAssistant: Hi\nUser: Continue\nAssistant:'
    );
    expect(llm.generator.mock.calls[0][1].return_full_text).toBe(false);
  });

  it('propagates template errors without retrying generic formatting', async () => {
    const llm = new BrowserLLM({ generator_model: 'org/chat' });
    llm.generator = vi.fn(async () => { throw new Error('Unsupported system role'); });
    llm.generator.tokenizer = { chat_template: 'template' };
    await expect(llm.chat([{ role: 'system', content: 'Hello' }])).rejects.toThrow('Unsupported system role');
    expect(llm.generator).toHaveBeenCalledTimes(1);
  });

  it.each([[], null, [{}], [{ role: '', content: 'x' }], [{ role: 'user', content: 3 }]])(
    'rejects invalid messages %j', async (messages) => {
      const llm = new BrowserLLM({});
      llm.generator = vi.fn();
      await expect(llm.chat(messages)).rejects.toThrow();
      expect(llm.generator).not.toHaveBeenCalled();
    }
  );

  it('rejects missing pipelines and malformed chat output', async () => {
    const llm = new BrowserLLM({});
    await expect(llm.chat([{ role: 'user', content: 'Hi' }])).rejects.toThrow('Generator model not initialized');
    llm.generator = vi.fn(async () => [{ generated_text: 'unexpected string' }]);
    llm.generator.tokenizer = { chat_template: 'template' };
    await expect(llm.chat([{ role: 'user', content: 'Hi' }])).rejects.toThrow('invalid assistant message');
  });
});

describe('BrowserLLM.embed()', () => {
  let llm;

  beforeEach(() => {
    const config = { embedding_model: 'all-MiniLM-L6-v2' };
    llm = new BrowserLLM(config);
  });

  it('should throw error if embedder not initialized', async () => {
    await expect(llm.embed('test')).rejects.toThrow('No embedding model configured');
  });

  it('should request a mean-pooled, L2-normalized sentence embedding', async () => {
    const mockEmbedding = [0.1, 0.2, 0.3];
    llm.embedder = vi.fn(async () => ({ data: mockEmbedding }));

    const result = await llm.embed('hello');

    // Must ask the pipeline to pool over tokens and normalize, otherwise it
    // returns the raw [1, tokens, hidden] last_hidden_state (see embed()).
    expect(llm.embedder).toHaveBeenCalledWith('hello', { pooling: 'mean', normalize: true });
    expect(result).toEqual(mockEmbedding);
  });

  it('pools to a flat sentence vector instead of the per-token tensor', async () => {
    // Mirror the real pipeline: without pooling it returns the unpooled
    // [1, tokens, hidden] last_hidden_state (flattened length = tokens*hidden);
    // with pooling it returns a single hidden-dim sentence vector. The mock
    // branches on the options so this test fails if embed() drops them -
    // independently of the toHaveBeenCalledWith assertion above.
    const HIDDEN = 4;
    const TOKENS = 3;
    const sentenceVector = Float32Array.from([0.5, -0.25, 0.75, 0.125]); // HIDDEN
    const tokenTensor = Float32Array.from(
      Array.from({ length: TOKENS * HIDDEN }, (_, i) => i / 16), // float32-exact
    );
    llm.embedder = vi.fn(async (_text, options) =>
      options?.pooling === 'mean'
        ? { data: sentenceVector }
        : { data: tokenTensor },
    );

    const result = await llm.embed('a longer phrase with several tokens');

    // Fixed-dim pooled vector, not the tokens*hidden flattened tensor.
    expect(result).toEqual([0.5, -0.25, 0.75, 0.125]);
    expect(result).toHaveLength(HIDDEN);
    expect(result).not.toHaveLength(TOKENS * HIDDEN);
  });
});

describe('Helper functions', () => {
  it('should create BrowserLLM instances', () => {
    const llm = createLLM({ generator_model: 'gpt2' });

    expect(llm).toBeInstanceOf(BrowserLLM);
    expect(llm.generatorModel).toBe('gpt2');
  });

  it('should compute cosine similarity', () => {
    expect(cosineSimilarity([1, 0, 0], [1, 0, 0])).toBe(1);
    expect(cosineSimilarity([1, 0, 0], [0, 1, 0])).toBe(0);
  });

  it('should compute batch cosine similarity', () => {
    const scores = cosineSimilarityBatch([1, 0, 0], [
      [1, 0, 0],
      [0, 1, 0]
    ]);

    expect(scores).toEqual([1, 0]);
  });
});

describe('isModelCached()', () => {
  it('should return false when no cache exists', async () => {
    globalThis.caches = { keys: vi.fn(async () => []) };

    const result = await isModelCached('gpt2');

    expect(result).toBe(false);
  });

  it('should return false when model is not in cache', async () => {
    const mockCache = { keys: vi.fn(async () => [{ url: 'https://huggingface.co/Xenova/other-model/resolve/main/config.json' }]) };
    globalThis.caches = {
      keys: vi.fn(async () => ['transformers-cache']),
      open: vi.fn(async () => mockCache),
    };

    const result = await isModelCached('gpt2');

    expect(result).toBe(false);
  });

  it('should return true when model files are in cache', async () => {
    const mockCache = { keys: vi.fn(async () => [{ url: 'https://huggingface.co/Xenova/gpt2/resolve/main/config.json' }]) };
    globalThis.caches = {
      keys: vi.fn(async () => ['transformers-cache']),
      open: vi.fn(async () => mockCache),
    };

    const result = await isModelCached('gpt2');

    expect(result).toBe(true);
  });

  it('should match full model IDs without Xenova fallback', async () => {
    const mockCache = { keys: vi.fn(async () => [{ url: 'https://huggingface.co/acme/gpt2/resolve/main/config.json' }]) };
    globalThis.caches = {
      keys: vi.fn(async () => ['transformers-cache']),
      open: vi.fn(async () => mockCache),
    };

    const result = await isModelCached('acme/gpt2');

    expect(result).toBe(true);
  });
});

describe('loadModelToCache()', () => {
  it('should be exposed globally', () => {
    expect(typeof globalThis.loadModelToCache).toBe('function');
  });

  it('should write files to the Cache API', async () => {
    const mockCache = { put: vi.fn() };
    const mockCaches = {
      keys: vi.fn(async () => ['transformers-cache']),
      open: vi.fn(async () => mockCache),
    };
    globalThis.caches = mockCaches;

    const files = {
      'config.json': new Uint8Array([1, 2, 3]),
      'vocab.json': new Uint8Array([4, 5, 6]),
    };

    await globalThis.loadModelToCache('gpt2', files);

    expect(mockCaches.open).toHaveBeenCalledWith('transformers-cache');
    expect(mockCache.put).toHaveBeenCalledTimes(2);

    const [url0] = mockCache.put.mock.calls[0];
    expect(url0).toContain('Xenova/gpt2');
    expect(url0).toContain('config.json');
  });

  it('should keep explicit model paths unchanged', async () => {
    const mockCache = { put: vi.fn() };
    const mockCaches = {
      keys: vi.fn(async () => ['transformers-cache']),
      open: vi.fn(async () => mockCache),
    };
    globalThis.caches = mockCaches;

    await globalThis.loadModelToCache('acme/gpt2', {
      'config.json': new Uint8Array([1, 2, 3]),
    });

    const [url0] = mockCache.put.mock.calls[0];
    expect(url0).toContain('acme/gpt2');
    expect(url0).not.toContain('Xenova/acme/gpt2');
  });

  it('should preserve nested model paths in Cache API URLs', async () => {
    const mockCache = { put: vi.fn() };
    globalThis.caches = {
      keys: vi.fn(async () => ['transformers-cache']),
      open: vi.fn(async () => mockCache),
    };

    await globalThis.loadModelToCache('gpt2', {
      'onnx/model_quantized.onnx': new Uint8Array([1, 2, 3]),
    });

    expect(mockCache.put).toHaveBeenCalledWith(
      'https://huggingface.co/Xenova/gpt2/resolve/main/onnx/model_quantized.onnx',
      expect.any(Response)
    );
  });
});

describe('Hosted model export', () => {
  afterEach(() => vi.unstubAllGlobals());

  it.each([
    'https://example.test/demo/models/Xenova/gpt2/onnx/model.onnx',
    'https://huggingface.co/Xenova/gpt2/resolve/main/onnx/model.onnx',
  ])('preserves nested asset paths for %s', async (url) => {
    const cache = { match: vi.fn(async () => new Response(new Uint8Array([1, 2, 3]))) };
    vi.stubGlobal('caches', { open: vi.fn(async () => cache) });
    const llm = new BrowserLLM({});

    const result = await llm._exportViaPython('Xenova/gpt2', 'transformers-cache', [{ url }]);

    expect(result.files[0].filename).toBe('onnx/model.onnx');
    expect(result.files[0].data).toEqual(new Uint8Array([1, 2, 3]));
  });
});

describe('clearModelCache()', () => {
  it('should be exposed globally', () => {
    expect(typeof clearModelCache).toBe('function');
  });

  it('should delete matching Cache API and IndexedDB entries', async () => {
    globalThis.caches = {
      keys: vi.fn(async () => ['transformers-cache', 'app-cache']),
      delete: vi.fn(async (name) => name === 'transformers-cache'),
    };
    globalThis.indexedDB = {
      databases: vi.fn(async () => [{ name: 'transformers-cache' }, { name: 'app-db' }]),
      deleteDatabase: vi.fn((name) => {
        const request = {};
        queueMicrotask(() => {
          if (request.onsuccess) {
            request.onsuccess({ target: { result: undefined } });
          }
        });
        return request;
      }),
    };

    const result = await clearModelCache();

    expect(globalThis.caches.delete).toHaveBeenCalledWith('transformers-cache');
    expect(globalThis.caches.delete).not.toHaveBeenCalledWith('app-cache');
    expect(globalThis.indexedDB.deleteDatabase).toHaveBeenCalledWith('transformers-cache');
    expect(result.deleted_caches).toEqual(['transformers-cache']);
    expect(result.deleted_databases).toContain('transformers-cache');
  });
});
