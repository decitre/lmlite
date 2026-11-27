import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import './lmlite.js';

// A browser with WebGPU, unless a test says otherwise.
beforeEach(() => { vi.stubGlobal('navigator', { gpu: { requestAdapter: async () => ({}) } }); });
afterEach(() => { vi.unstubAllGlobals(); });

vi.mock('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2', () => {
  class PreTrainedTokenizer {
    constructor(json, config) { this.json = json; this.config = config; }
    get_vocab() { return new Map(this.json.model.vocab.map((token, id) => [token, id])); }
    encode(text, { add_special_tokens = true } = {}) { return [...text].map(c => c.charCodeAt(0)).concat(add_special_tokens ? [0] : []); }
    decode(ids) { return String.fromCharCode(...ids); }
  }
  class BertTokenizer extends PreTrainedTokenizer {}
  return {
  PreTrainedTokenizer,
  BertTokenizer,
  pipeline: vi.fn(),
  RawImage: class {
    constructor(data, width, height, channels) { Object.assign(this, { data, width, height, channels }); }
    static fromBlob = vi.fn(async (blob) => ({ blob }));
    static read = vi.fn(async (url) => ({ url }));
  },
  env: {
    fetch: vi.fn(async () => new Response('{}')),
    backends: { onnx: { versions: { web: '1.2.3' }, wasm: { wasmPaths: { mjs: 'a.mjs', wasm: 'a.wasm' } } } },
  },
  InterruptableStoppingCriteria: class {
    interrupted = false;
    interrupt() { this.interrupted = true; }
  },
  // Emits each streamed token as one chunk, like TextStreamer's callback_function.
  TextStreamer: class {
    constructor(tokenizer, options) { this.options = options; }
    put(text) { this.options.callback_function(text); }
  },
  ModelRegistry: {
    get_available_dtypes: vi.fn(),
    is_pipeline_cached_files: vi.fn(),
    get_file_metadata: vi.fn(),
  },
  };
});

vi.mock('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.2.3/dist/ort.webgpu.min.mjs', () => {
  class Tensor {
    constructor(type, data, dims) { Object.assign(this, { type, data, dims }); }
    async getData() { return this.data; }
  }
  const session = {
    inputNames: ['x'], outputNames: ['y'],
    inputMetadata: [{ name: 'x', type: 'float32', shape: ['batch', 3] }],
    outputMetadata: [{ name: 'y', type: 'float32', shape: ['batch', 3] }],
    run: vi.fn(async ({ x }) => ({ y: new Tensor('float32', x.data.map(v => v * 2), x.dims) })),
    release: vi.fn(async () => {}),
  };
  return { env: { wasm: {} }, Tensor, InferenceSession: { create: vi.fn(async () => session) }, session };
});

const {
  BrowserRuntime,
  createRuntime,
  cosineSimilarity,
  cosineSimilarityBatch,
  isModelCached,
  loadModelToCache,
  loadOnnxModel,
  AudioRecorder,
  encodeWav,
  mixChannels,
  clearModelCache,
  getCachedModels,
  getHostedModelBase,
  getModelInfo,
  getModelTask,
  loadTokenizer,
} = globalThis.lmLite;

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

  describe('when the manifest request does not give a usable answer', () => {
    const listed = () => new Response(JSON.stringify({ models: ['Xenova/gpt2'] }));

    beforeEach(() => {
      vi.useFakeTimers({ toFake: ['setTimeout'] });
      vi.stubGlobal('location', new URL('https://example.test/lab/index.html'));
    });

    afterEach(() => {
      vi.useRealTimers();
      vi.restoreAllMocks();
    });

    // The retries wait between attempts, so run the timers, with the rejection handled as soon as it happens.
    async function settle(promise) {
      const outcome = promise.then((value) => ({ value }), (error) => ({ error }));
      await vi.runAllTimersAsync();
      return await outcome;
    }

    it('tries again after a 5xx and uses the manifest that follows', async () => {
      const fetch = vi.fn()
        .mockResolvedValueOnce(new Response('', { status: 503 }))
        .mockResolvedValueOnce(new Response('', { status: 503 }))
        .mockResolvedValueOnce(listed());
      vi.stubGlobal('fetch', fetch);

      expect((await settle(getHostedModelBase('gpt2'))).value).toBe('https://example.test/files/models/');
      expect(fetch).toHaveBeenCalledTimes(3);
    });

    it.each([
      ['a 503', () => new Response('', { status: 503 })],
      ['a network error', () => Promise.reject(new TypeError('NetworkError when attempting to fetch resource.'))],
    ])('treats %s on the auto-discovered manifest as nothing hosted, with a warning', async (name, answer) => {
      const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
      const fetch = vi.fn(async () => answer());
      vi.stubGlobal('fetch', fetch);

      expect((await settle(getHostedModelBase('gpt2'))).value).toBeNull();
      expect(fetch).toHaveBeenCalledTimes(3);
      expect(warn.mock.calls[0][0]).toContain('loading from the Hub');
    });

    it('does not retry a 404, the normal answer of a site that hosts no models', async () => {
      const fetch = vi.fn(async () => new Response('', { status: 404 }));
      vi.stubGlobal('fetch', fetch);

      expect((await settle(getHostedModelBase('gpt2'))).value).toBeNull();
      expect(fetch).toHaveBeenCalledTimes(1);
    });

    it('still reports an invalid manifest that did arrive, even when auto-discovered', async () => {
      vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ models: [5] }))));

      expect((await settle(getHostedModelBase('gpt2'))).error.message).toContain('Invalid hosted model manifest');
    });

    it('keeps raising for an explicit manifest URL that stays unavailable, after the same retries', async () => {
      const fetch = vi.fn(async () => new Response('', { status: 503 }));
      vi.stubGlobal('fetch', fetch);

      expect((await settle(getHostedModelBase('gpt2', 'https://example.test/models/'))).error.message).toContain('manifest request failed: 503');
      expect(fetch).toHaveBeenCalledTimes(3);
    });

    it('lets an explicit manifest URL recover from one network error', async () => {
      const fetch = vi.fn()
        .mockRejectedValueOnce(new TypeError('NetworkError'))
        .mockResolvedValueOnce(listed());
      vi.stubGlobal('fetch', fetch);

      expect((await settle(getHostedModelBase('gpt2', 'https://example.test/models/'))).value).toBe('https://example.test/models/');
    });
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

describe('the names before the runtime was renamed', () => {
  it('still point at the runtime', () => {
    expect(globalThis.lmLite.createLLM).toBe(globalThis.lmLite.createRuntime);
    expect(globalThis.lmLite.BrowserLLM).toBe(globalThis.lmLite.BrowserRuntime);
  });
});

describe('BrowserRuntime Constructor', () => {
  it('should initialize with minimal config', () => {
    const config = { generator_model: 'gpt2' };
    const llm = new BrowserRuntime(config);

    expect(llm.generatorModel).toBe('gpt2');
    expect(llm.embeddingModel).toBeUndefined();
    expect(llm.backend).toBe('wasm');
  });

  it('should set classification_model when provided', () => {
    const config = {
      classification_model: 'distilbert-base-uncased-finetuned-sst-2-english'
    };
    const llm = new BrowserRuntime(config);

    expect(llm.classificationModel).toBe('distilbert-base-uncased-finetuned-sst-2-english');
    expect(llm.generatorModel).toBeUndefined();
  });

  it('should normalize backend to lowercase', () => {
    const config = { backend: 'WEBGPU', generator_model: 'gpt2' };
    const llm = new BrowserRuntime(config);

    expect(llm.backend).toBe('webgpu');
  });

  it('should set default values for generation parameters', () => {
    const config = { generator_model: 'gpt2' };
    const llm = new BrowserRuntime(config);

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
    const llm = new BrowserRuntime(config);

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
    const llm = new BrowserRuntime(config);

    expect(llm.max_new_tokens).toBe(100);
    expect(llm.temperature).toBe(0.5);
    expect(llm.top_k).toBe(30);
  });
});

describe('BrowserRuntime dtype handling', () => {
  it('should honor explicit dtype override', async () => {
    const llm = new BrowserRuntime({ generator_model: 'gpt2', dtype: 'fp32' });
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
    const llm = new BrowserRuntime({
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

describe('BrowserRuntime initialization progress', () => {
  it('uses individual hosted files for selected models and preserves remote fallback for others', async () => {
    const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
    pipeline.mockImplementation(async (_task, _model, options) => {
      options.progress_callback({ status: 'progress', file: 'model.onnx', loaded: 1, total: 2 });
      return vi.fn(async () => []);
    });
    const llm = new BrowserRuntime({
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

    await new BrowserRuntime({}).init();
    expect(env.allowLocalModels).toBe(false);
  });

  it('serves Hub requests for hosted models from the hosted files', async () => {
    const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
    pipeline.mockImplementation(async () => vi.fn(async () => []));
    const original = vi.fn(async () => new Response('{}'));
    env.fetch = original;
    await new BrowserRuntime({
      generator_model: 'gpt2',
      hosted_model_base: 'https://example.test/demo/models/',
      hosted_models: ['Xenova/gpt2'],
    }).init();

    await env.fetch('https://huggingface.co/Xenova/gpt2/resolve/main/tokenizer_config.json', { method: 'GET' });
    await env.fetch('https://huggingface.co/org/other/resolve/main/tokenizer_config.json');

    expect(original.mock.calls).toEqual([
      ['https://example.test/demo/models/Xenova/gpt2/tokenizer_config.json', { method: 'GET' }],
      ['https://huggingface.co/org/other/resolve/main/tokenizer_config.json', undefined],
    ]);
  });

  it('forwards download progress and warm-up status to the notebook callback', async () => {
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
    const generator = vi.fn(async () => []);
    pipeline.mockImplementation(async (_task, _model, options) => {
      options.progress_callback({
        status: 'progress', file: 'onnx/model.onnx', loaded: 25, total: 100,
      });
      options.progress_callback({ status: 'ready' });
      return generator;
    });
    const callback = vi.fn();
    const llm = new BrowserRuntime({ generator_model: 'gpt2' });

    await llm.init(callback);

    expect(callback).toHaveBeenCalledWith(
      'Xenova/gpt2', 'progress', 'onnx/model.onnx', 25, 100
    );
    expect(callback).toHaveBeenCalledWith('Xenova/gpt2', 'initializing', '', 0, 0);
    expect(callback).toHaveBeenCalledWith('Xenova/gpt2', 'warming up', '', 0, 0);
    expect(generator).toHaveBeenCalledWith('test', { max_new_tokens: 1 });
    expect(callback).toHaveBeenLastCalledWith('', 'ready', '', 0, 0);
  });

  it('yields a macrotask before each generation step so kernels can resume between tokens', async () => {
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
    const events = [];
    const forward = vi.fn(async function () { events.push(this === model ? 'forward' : 'unbound'); });
    const model = { forward };
    const generator = Object.assign(vi.fn(async () => []), { model });
    pipeline.mockImplementation(async () => generator);
    const llm = new BrowserRuntime({ generator_model: 'gpt2' });
    await llm.init();

    setTimeout(() => events.push('macrotask'), 0);
    await llm.generator.model.forward('inputs');
    expect(events).toEqual(['macrotask', 'forward']);
    expect(forward).toHaveBeenCalledWith('inputs');
  });

  it.each(['embedding_model', 'classification_model'])(
    'forwards progress for %s',
    async (modelKey) => {
      const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
      pipeline.mockImplementation(async (_task, _model, options) => {
        options.progress_callback({
          status: 'progress', file: 'model.onnx', loaded: 5, total: 10,
        });
        return vi.fn();
      });
      const callback = vi.fn();

      await new BrowserRuntime({ [modelKey]: 'test-model' }).init(callback);

      expect(callback).toHaveBeenCalledWith('Xenova/test-model', 'progress', 'model.onnx', 5, 10);
    }
  );

  it('still initializes without a progress callback', async () => {
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
    pipeline.mockImplementation(async (_task, _model, options) => {
      options.progress_callback({ status: 'downloading', file: 'model.onnx' });
      return vi.fn(async () => []);
    });

    await expect(new BrowserRuntime({ generator_model: 'gpt2' }).init()).resolves.toBeUndefined();
  });
});

describe('BrowserRuntime embedding dtype pinning', () => {
  it.each([
    ['wasm', 'webgpu', null],
    ['WEBGPU', 'wasm', 'q8'],
  ])('loads embeddings on %s independently of backend=%s', async (embedding_backend, backend, embedding_dtype) => {
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
    pipeline.mockImplementation(async () => vi.fn(async () => []));
    pipeline.mockClear();
    const llm = new BrowserRuntime({
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
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
    pipeline.mockClear();

    await expect(new BrowserRuntime({ embedding_model: 'org/embedding', embedding_backend }).init())
      .rejects.toThrow('Invalid embedding_backend');
    expect(pipeline).not.toHaveBeenCalled();
  });

  it('does not fall back to WASM when WebGPU embedding loading fails with backend_fallback false', async () => {
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
    pipeline.mockRejectedValue(new Error('WebGPU unavailable'));
    pipeline.mockClear();

    await expect(new BrowserRuntime({
      embedding_model: 'org/embedding', embedding_backend: 'webgpu', backend_fallback: false,
    }).init()).rejects.toThrow("Failed to load embedding model 'org/embedding': WebGPU unavailable");
    expect(pipeline).toHaveBeenCalledTimes(1);
    expect(pipeline.mock.calls[0][2].device).toBe('webgpu');
  });

  describe('backend fallback without WebGPU', () => {
    const noGpu = () => vi.stubGlobal('navigator', { gpu: { requestAdapter: async () => null } });
    const load = async (config) => {
      const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
      pipeline.mockReset();
      pipeline.mockResolvedValue(vi.fn());
      const runtime = new BrowserRuntime(config);
      await runtime.init();
      return { pipeline, runtime };
    };

    it('uses wasm for every model when there is no adapter', async () => {
      noGpu();
      const { pipeline, runtime } = await load({
        classification_model: 'org/cls', embedding_model: 'org/emb', embedding_backend: 'webgpu', backend: 'webgpu',
      });
      expect(runtime.backend).toBe('wasm');
      expect(pipeline.mock.calls.map(c => c[2].device)).toEqual(['wasm', 'wasm']);
    });

    it('uses wasm when navigator.gpu is missing', async () => {
      vi.stubGlobal('navigator', {});
      expect((await load({ classification_model: 'org/cls', backend: 'webgpu' })).runtime.backend).toBe('wasm');
    });

    it('keeps webgpu with backend_fallback false, and the load fails as before', async () => {
      noGpu();
      const { pipeline, runtime } = await load({ classification_model: 'org/cls', backend: 'webgpu', backend_fallback: false });
      expect(runtime.backend).toBe('webgpu');
      expect(pipeline.mock.calls[0][2].device).toBe('webgpu');
    });

    it('keeps webgpu when there is an adapter', async () => {
      expect((await load({ classification_model: 'org/cls', backend: 'webgpu' })).runtime.backend).toBe('webgpu');
    });

    describe('after a WebGPU load failure', () => {
      const tooManyBuffers = 'no available backend found. ERR: [webgpu] Error: Too many storage buffers in shader. Current: 10, Max is 9';
      const failOnWebgpu = async (message) => {
        const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
        pipeline.mockReset();
        pipeline.mockImplementation(async (task, model, options) => {
          if (options.device === 'webgpu') throw new Error(message);
          return vi.fn(async () => []);
        });
        return pipeline;
      };

      it('retries a classifier once on wasm, keeps the dtype order, and reports why', async () => {
        const pipeline = await failOnWebgpu(tooManyBuffers);
        const runtime = new BrowserRuntime({ classification_model: 'org/cls', backend: 'webgpu', dtype: 'q4' });
        await runtime.init();

        expect(pipeline.mock.calls.map(c => [c[2].device, c[2].dtype])).toEqual([['webgpu', 'q4'], ['wasm', 'q4']]);
        expect(runtime.backend).toBe('wasm');
        expect(runtime.fallbackReason).toContain('Too many storage buffers');
      });

      it('frees a generator that loaded but failed to warm up before the retry', async () => {
        const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
        const dispose = vi.fn(async () => {});
        const order = [];
        pipeline.mockReset();
        pipeline.mockImplementation(async (task, model, options) => {
          order.push(`load ${options.device}`);
          const generator = vi.fn(async () => {
            if (options.device === 'webgpu') throw new Error('Device lost');
          });
          generator.model = {};
          generator.dispose = async () => { order.push('dispose'); await dispose(); };
          return generator;
        });
        const runtime = new BrowserRuntime({ generator_model: 'org/gen', backend: 'webgpu', dtype: 'q4' });
        await runtime.init();

        expect(order).toEqual(['load webgpu', 'dispose', 'load wasm']);
        expect(runtime.backend).toBe('wasm');
      });

      it('retries an embedder on wasm with its pinned dtype', async () => {
        const pipeline = await failOnWebgpu(tooManyBuffers);
        const runtime = new BrowserRuntime({
          embedding_model: 'org/emb', embedding_backend: 'webgpu', embedding_dtype: 'q8',
        });
        await runtime.init();

        expect(pipeline.mock.calls.map(c => [c[2].device, c[2].dtype])).toEqual([['webgpu', 'q8'], ['wasm', 'q8']]);
        expect(runtime.embeddingBackend).toBe('wasm');
      });

      it('loads the next model on the backend the first one ended up on', async () => {
        const pipeline = await failOnWebgpu(tooManyBuffers);
        await new BrowserRuntime({ classification_model: 'org/cls', task: 'translation', task_model: 'org/t', backend: 'webgpu' }).init();

        expect(pipeline.mock.calls.map(c => c[2].device)).toEqual(['webgpu', 'wasm', 'wasm']);
      });

      it.each([
        ['out of memory', 'no available backend found. ERR: [webgpu] RangeError: Out of memory'],
        ['a dropped download', 'no available backend found. ERR: [webgpu] TypeError: Failed to fetch'],
        ['a missing file', 'Could not locate file: model.onnx'],
      ])('does not retry %s', async (name, message) => {
        const pipeline = await failOnWebgpu(message);
        await expect(new BrowserRuntime({ classification_model: 'org/cls', backend: 'webgpu', dtype: 'q4' }).init())
          .rejects.toThrow(message);
        expect(pipeline.mock.calls.map(c => c[2].device)).toEqual(['webgpu']);
      });

      it('does not retry with backend_fallback false', async () => {
        const pipeline = await failOnWebgpu(tooManyBuffers);
        await expect(new BrowserRuntime({
          classification_model: 'org/cls', backend: 'webgpu', dtype: 'q4', backend_fallback: false,
        }).init()).rejects.toThrow('Too many storage buffers');
        expect(pipeline).toHaveBeenCalledTimes(1);
      });

      it('names both failures when the retry fails too', async () => {
        const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
        pipeline.mockReset();
        pipeline.mockImplementation(async (task, model, options) => {
          throw new Error(options.device === 'webgpu' ? tooManyBuffers : 'unsupported operator');
        });
        await expect(new BrowserRuntime({ classification_model: 'org/cls', backend: 'webgpu', dtype: 'q4' }).init())
          .rejects.toThrow(/unsupported operator \(WebGPU had failed first: .*Too many storage buffers/);
      });
    });

    it('leaves wasm and an unset embedding device alone', async () => {
      noGpu();
      const { pipeline } = await load({ embedding_model: 'org/emb', backend: 'wasm' });
      expect(pipeline.mock.calls[0][2].device).toBeUndefined();
    });
  });

  describe('the message when a generator or task model fails', () => {
    const failing = async (options, mock) => {
      const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
      pipeline.mockReset();
      mock(pipeline);
      return new BrowserRuntime(options).init().then(() => null, (error) => error.message);
    };

    it('says a WASM out-of-memory error is memory, and keeps the runtime message', async () => {
      const message = await failing({ generator_model: 'distilgpt2', backend: 'wasm' }, (pipeline) =>
        pipeline.mockRejectedValue(new Error('no available backend found. ERR: [wasm] RangeError: Out of memory')));

      expect(message).toContain("Failed to load text-generation model 'distilgpt2': no available backend found. ERR: [wasm] RangeError: Out of memory");
      expect(message).toContain('ran out of memory');
      expect(message).toContain("backend 'wasm'");
      expect(message).toContain('Transformers.js 4.3.2');
      expect(message).not.toContain('may not be supported');
    });

    it('names the warm-up stage when the model loaded but could not generate', async () => {
      const message = await failing({ generator_model: 'gpt2' }, (pipeline) =>
        pipeline.mockResolvedValue(Object.assign(vi.fn().mockRejectedValue(new Error('session run failed')), { model: {} })));

      expect(message).toContain("Failed to warm up text-generation model 'gpt2': session run failed");
      expect(message).toContain('loaded but could not generate one token');
    });

    it('lists the dtypes tried when none is available, and keeps the compatibility advice', async () => {
      const message = await failing({ generator_model: 'org/model', dtype_fallbacks: ['q4', 'fp32'] }, (pipeline) =>
        pipeline.mockRejectedValue(new Error('Could not locate file model.onnx')));

      expect(message).toContain("Failed to load text-generation model 'org/model': Could not locate file model.onnx");
      expect(message).toMatch(/dtype \w+( then \w+)+/);
      expect(message).toContain('may not be supported');
    });

    it.each([
      ['Safari', 'Load failed'],
      ['Chrome', 'Failed to fetch'],
      ['Firefox', 'NetworkError when attempting to fetch resource.'],
    ])('says a failed fetch in %s is a fetch, not an unsupported model', async (_browser, text) => {
      const message = await failing({ generator_model: 'org/model', backend: 'webgpu' }, (pipeline) =>
        pipeline.mockRejectedValue(new TypeError(text)));

      expect(message).toContain(`Failed to load text-generation model 'org/model': ${text}`);
      expect(message).toContain('A file could not be fetched');
      expect(message).toContain("fails only with backend 'webgpu'");
      expect(message).not.toContain('may not be supported');
      expect(message).toContain("backend 'webgpu'");
    });

    it('suggests the other backend only after a WebGPU failure', async () => {
      const message = await failing({ generator_model: 'org/model', backend: 'wasm' }, (pipeline) =>
        pipeline.mockRejectedValue(new TypeError('Failed to fetch')));

      expect(message).toContain('A file could not be fetched');
      expect(message).not.toContain('fails only with backend');
    });

    it('says a fetch that fails during the warm-up is a fetch too', async () => {
      const message = await failing({ generator_model: 'gpt2', backend: 'webgpu' }, (pipeline) =>
        pipeline.mockResolvedValue(Object.assign(vi.fn().mockRejectedValue(new TypeError('Load failed')), { model: {} })));

      expect(message).toContain("Failed to warm up text-generation model 'gpt2': Load failed");
      expect(message).toContain('A file could not be fetched');
      expect(message).not.toContain('could not generate one token');
    });

    it('treats a message that is both out of memory and a fetch failure as memory', async () => {
      const message = await failing({ generator_model: 'org/model' }, (pipeline) =>
        pipeline.mockRejectedValue(new Error('Failed to fetch: out of memory')));

      expect(message).toContain('ran out of memory');
      expect(message).not.toContain('A file could not be fetched');
    });

    it('gives a task model a fetch failure message too', async () => {
      const message = await failing({ task: 'translation', task_model: 'org/foo', backend: 'webgpu' }, (pipeline) =>
        pipeline.mockRejectedValue(new TypeError('Load failed')));

      expect(message).toContain("Failed to load translation model 'org/foo': Load failed");
      expect(message).toContain('A file could not be fetched');
    });

    it('gives a task model the same message', async () => {
      const message = await failing({ task: 'translation', task_model: 'org/foo' }, (pipeline) =>
        pipeline.mockRejectedValue(new Error('failed to allocate a buffer of size 1733430199')));

      expect(message).toContain("Failed to load translation model 'org/foo': failed to allocate a buffer of size 1733430199");
      expect(message).toContain('ran out of memory');
    });
  });

  it('surfaces a pinned dtype failure without retrying', async () => {
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
    pipeline.mockRejectedValue(new Error('Could not locate file model_quantized.onnx'));
    pipeline.mockClear();

    await expect(new BrowserRuntime({
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
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
    pipeline.mockImplementation(async () => vi.fn(async () => ({ data: [0] })));
    pipeline.mockClear();

    await new BrowserRuntime({ embedding_model: 'org/embedding' }).init();

    const call = pipeline.mock.calls.find((c) => c[0] === 'feature-extraction');
    expect(call).toBeDefined();
    expect(call[2].dtype).toBeUndefined();
  });

  it('pins the feature-extraction dtype when embedding_dtype is set', async () => {
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
    const embedder = vi.fn(async () => ({ data: [0] }));
    pipeline.mockImplementation(async () => embedder);
    pipeline.mockClear();

    await new BrowserRuntime({ embedding_model: 'org/embedding', embedding_dtype: 'q8' }).init();

    const call = pipeline.mock.calls.find((c) => c[0] === 'feature-extraction');
    expect(call).toBeDefined();
    expect(call[2]).toEqual(expect.objectContaining({ dtype: 'q8' }));
    expect(embedder).not.toHaveBeenCalled();
  });

  it('embedding_dtype is NOT overridden by the generator dtype / dtype_fallbacks', async () => {
    // dtype and dtype_fallbacks target the generator/classifier pipelines. They
    // must never leak into the embedding pin — the embedding dtype must stay q8.
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
    pipeline.mockImplementation(async () => vi.fn(async () => ({ data: [0] })));
    pipeline.mockClear();

    await new BrowserRuntime({
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
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
    pipeline.mockImplementation(async () => vi.fn(async () => ({ data: [0] })));
    pipeline.mockClear();

    await new BrowserRuntime({
      embedding_model: 'org/embedding',
      embedding_dtype: 'q8',
      backend: 'webgpu',
    }).init();

    const call = pipeline.mock.calls.find((c) => c[0] === 'feature-extraction');
    expect(call).toBeDefined();
    expect(call[2].device).toBeUndefined();
  });
});

describe('BrowserRuntime.dispose()', () => {
  it('releases all loaded pipelines and is safe to repeat', async () => {
    const llm = new BrowserRuntime({});
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
    const llm = new BrowserRuntime({});
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
      await new BrowserRuntime({}).dispose();
      expect(deleteCache).not.toHaveBeenCalled();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe('BrowserRuntime.classify()', () => {
  let llm;

  beforeEach(() => {
    const config = { classification_model: 'distilbert' };
    llm = new BrowserRuntime(config);
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

describe('BrowserRuntime task pipelines', () => {
  it('loads a supported task with the shared backend, dtype and hosted options', async () => {
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
    const taskPipeline = vi.fn();
    pipeline.mockReset();
    pipeline.mockResolvedValue(taskPipeline);
    const llm = new BrowserRuntime({
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
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
    pipeline.mockReset();
    await expect(new BrowserRuntime({ task: 'image-to-text', task_model: 'm' }).init())
      .rejects.toThrow("Unsupported task 'image-to-text'. Supported pipeline tasks are: zero-shot-classification");
    await expect(new BrowserRuntime({ task: 'translation' }).init()).rejects.toThrow("requires a model");
    expect(pipeline).not.toHaveBeenCalled();
  });

  it('reports the task and model when loading fails', async () => {
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
    pipeline.mockReset();
    pipeline.mockRejectedValue(new Error('Unsupported model type: foo'));
    await expect(new BrowserRuntime({ task: 'translation', task_model: 'org/foo' }).init())
      .rejects.toThrow("Failed to load translation model 'org/foo': Unsupported model type: foo");
  });

  it('passes candidate_labels positionally for zero-shot classification', async () => {
    const llm = new BrowserRuntime({ task: 'zero-shot-classification' });
    const output = { sequence: 'text', labels: ['technology', 'sports'], scores: [0.9, 0.1] };
    llm.taskPipeline = vi.fn(async () => output);

    const result = await llm.runTask('text', JSON.stringify({ candidate_labels: ['technology', 'sports'], multi_label: true }));

    expect(llm.taskPipeline).toHaveBeenCalledWith('text', ['technology', 'sports'], { multi_label: true });
    expect(result).toBe(output);
    await expect(llm.runTask('text', '{}')).rejects.toThrow('requires candidate_labels');
  });

  it('passes candidate_labels positionally for zero-shot image classification', async () => {
    const llm = new BrowserRuntime({ task: 'zero-shot-image-classification' });
    llm.taskPipeline = vi.fn(async () => [{ label: 'food', score: 1 }]);

    await llm.runTask('https://example.test/a.jpg', JSON.stringify({ candidate_labels: ['food', 'face'] }));

    expect(llm.taskPipeline).toHaveBeenCalledWith({ url: 'https://example.test/a.jpg' }, ['food', 'face'], {});
    await expect(llm.runTask('https://example.test/a.jpg', '{}')).rejects.toThrow('requires candidate_labels');
  });

  it('decodes encoded bytes, raw pixels and batches into RawImage inputs', async () => {
    const llm = new BrowserRuntime({ task: 'object-detection' });
    llm.taskPipeline = vi.fn(async () => []);
    const bytes = new Uint8Array([1, 2, 3]);

    await llm.runTask(bytes, '{"threshold": 0.5, "percentage": false}');
    expect(llm.taskPipeline.mock.calls[0][0].blob).toBeInstanceOf(Blob);
    expect(llm.taskPipeline.mock.calls[0][1]).toEqual({ threshold: 0.5, percentage: false });

    await llm.runTask(['raw', new Uint8Array(24), 3, 2, 4]);
    expect(llm.taskPipeline.mock.calls[1][0]).toMatchObject({ width: 3, height: 2, channels: 4 });

    await llm.runTask([bytes, 'https://example.test/a.jpg']);
    expect(llm.taskPipeline.mock.calls[2][0]).toHaveLength(2);
  });

  it('reports undecodable image bytes', async () => {
    const { RawImage } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
    RawImage.fromBlob.mockRejectedValueOnce(new Error('bad data'));
    const llm = new BrowserRuntime({ task: 'image-classification' });
    llm.taskPipeline = vi.fn();

    await expect(llm.runTask(new Uint8Array([0]))).rejects.toThrow('The image could not be decoded: bad data');
    expect(llm.taskPipeline).not.toHaveBeenCalled();
  });

  it('returns one plain vector per image for image feature extraction', async () => {
    const llm = new BrowserRuntime({ task: 'image-feature-extraction' });
    // [images, tokens, dim] without pooling: keep the first token; [images, dim] when pooled.
    llm.taskPipeline = vi.fn(async (inputs, options) => ({
      tolist: () => {
        const vectors = [[0.5, 1.5], [2, 3]].slice(0, Array.isArray(inputs) ? 2 : 1);
        return options.pool ? vectors : vectors.map(vector => [vector, [9, 9]]);
      },
    }));

    expect(await llm.runTask('https://example.test/a.jpg')).toEqual([0.5, 1.5]);
    expect(await llm.runTask(['https://example.test/a.jpg', 'https://example.test/b.jpg'])).toEqual([[0.5, 1.5], [2, 3]]);
    expect(await llm.runTask('https://example.test/a.jpg', '{"pool": true}')).toEqual([0.5, 1.5]);
  });

  it('unwraps a single translation and keeps batched results', async () => {
    const llm = new BrowserRuntime({ task: 'translation' });
    llm.taskPipeline = vi.fn(async (inputs) => [inputs].flat().map(text => ({ translation_text: `fr:${text}` })));

    expect(await llm.runTask('Hello', '{"max_new_tokens": 8}')).toEqual({ translation_text: 'fr:Hello' });
    expect(llm.taskPipeline).toHaveBeenCalledWith('Hello', { max_new_tokens: 8 });
    expect(await llm.runTask(['a', 'b'])).toEqual([{ translation_text: 'fr:a' }, { translation_text: 'fr:b' }]);
  });

  it('converts raw audio samples to Float32Array and keeps URLs', async () => {
    const llm = new BrowserRuntime({ task: 'automatic-speech-recognition' });
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

  it('gives audio classification the same audio input as speech recognition, one clip per call', async () => {
    const llm = new BrowserRuntime({ task: 'audio-classification' });
    llm.taskPipeline = vi.fn(async (clip) => [{ label: 'Violin', score: clip.length / 10 }]);

    expect(await llm.runTask([0, 0.5, -0.5], '{"top_k": 3}')).toEqual([{ label: 'Violin', score: 0.3 }]);
    const [audio, options] = llm.taskPipeline.mock.calls[0];
    expect(audio).toBeInstanceOf(Float32Array);
    expect(options).toEqual({ top_k: 3 });

    const { results } = await llm.runTasks([new Float32Array(2), new Float32Array(4)], 8);
    expect(results).toEqual([[{ label: 'Violin', score: 0.2 }], [{ label: 'Violin', score: 0.4 }]]);
    expect(llm.taskPipeline).toHaveBeenCalledTimes(3);
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
    const llm = new BrowserRuntime({ task: 'automatic-speech-recognition' });
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
    const llm = new BrowserRuntime({ task: 'translation' });
    await expect(llm.runTask('Hello')).rejects.toThrow('translation pipeline not initialized');
    await expect(new BrowserRuntime({}).runTask('Hello')).rejects.toThrow('No pipeline task configured');
    llm.taskPipeline = vi.fn(async () => { throw new Error('out of memory'); });
    await expect(llm.runTask('Hello')).rejects.toThrow('translation failed: out of memory');
  });

  it('disposes the task pipeline', async () => {
    const llm = new BrowserRuntime({ task: 'translation' });
    const taskPipeline = { dispose: vi.fn(async () => {}) };
    llm.taskPipeline = taskPipeline;
    await llm.dispose();
    expect(taskPipeline.dispose).toHaveBeenCalledTimes(1);
    expect(llm.taskPipeline).toBeNull();
  });
});

describe('BrowserRuntime.generate()', () => {
  let llm;

  beforeEach(() => {
    const config = { generator_model: 'gpt2', max_new_tokens: 100 };
    llm = new BrowserRuntime(config);
  });

  it('should throw error if generator not initialized', async () => {
    await expect(llm.generate('test')).rejects.toThrow('Generator model not initialized. Call init() first.');
  });

  it('should call generator with prompt and max_new_tokens', async () => {
    const mockResult = [{ generated_text: 'Hello world' }];
    llm.generator = vi.fn(async () => mockResult);

    const result = await llm.generate('Hello');

    expect(llm.generator).toHaveBeenCalledWith('Hello', { max_new_tokens: 100, return_full_text: true, temperature: 0.7, top_k: 50, do_sample: true, stopping_criteria: expect.anything() });
    expect(result).toBe('Hello world');
  });

  it('should request only newly generated text when return_full_text is false', async () => {
    llm = new BrowserRuntime({ generator_model: 'gpt2', return_full_text: false });
    llm.generator = vi.fn(async () => [{ generated_text: 'world' }]);

    expect(await llm.generate('Hello')).toBe('world');
    expect(llm.generator).toHaveBeenCalledWith('Hello', {
      max_new_tokens: 50, return_full_text: false, temperature: 0.7, top_k: 50, do_sample: true, stopping_criteria: expect.anything()
    });
  });
});

describe('BrowserRuntime.chat()', () => {
  it('parses JSON messages from either Python browser bridge', async () => {
    const llm = new BrowserRuntime({ generator_model: 'org/chat-model' });
    const messages = [{ role: 'user', content: 'Bonjour \u00e9\n"quoted" <|im_start|>' }];
    llm.generator = vi.fn(async (input) => [{
      generated_text: [...input, { role: 'assistant', content: 'Answer' }]
    }]);
    llm.generator.tokenizer = { chat_template: 'template' };
    expect(await llm.chat(JSON.stringify(messages))).toBe('Answer');
    expect(llm.generator.mock.calls[0][0]).toEqual(messages);
  });

  it.each(['{invalid', 'null', '{}', '[]'])('rejects invalid serialized messages %s', async (input) => {
    const llm = new BrowserRuntime({});
    llm.generator = vi.fn();
    await expect(llm.chat(input)).rejects.toThrow();
    expect(llm.generator).not.toHaveBeenCalled();
  });

  it('uses structured chat input when the loaded tokenizer has a template', async () => {
    const llm = new BrowserRuntime({ generator_model: 'org/chat-model', return_full_text: true });
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
      { max_new_tokens: 50, return_full_text: false, temperature: 0.7, top_k: 50, do_sample: true, stopping_criteria: expect.anything() }
    );
    expect(llm.return_full_text).toBe(true);
    expect(messages[0].role).toBe('SYSTEM');
  });

  it.each([true, false])('keeps generic formatting without a template, full text=%s', async (return_full_text) => {
    const llm = new BrowserRuntime({ generator_model: 'gpt2', return_full_text });
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
    const llm = new BrowserRuntime({ generator_model: 'org/chat' });
    llm.generator = vi.fn(async () => { throw new Error('Unsupported system role'); });
    llm.generator.tokenizer = { chat_template: 'template' };
    await expect(llm.chat([{ role: 'system', content: 'Hello' }])).rejects.toThrow('Unsupported system role');
    expect(llm.generator).toHaveBeenCalledTimes(1);
  });

  it.each([[], null, [{}], [{ role: '', content: 'x' }], [{ role: 'user', content: 3 }]])(
    'rejects invalid messages %j', async (messages) => {
      const llm = new BrowserRuntime({});
      llm.generator = vi.fn();
      await expect(llm.chat(messages)).rejects.toThrow();
      expect(llm.generator).not.toHaveBeenCalled();
    }
  );

  it('rejects missing pipelines and malformed chat output', async () => {
    const llm = new BrowserRuntime({});
    await expect(llm.chat([{ role: 'user', content: 'Hi' }])).rejects.toThrow('Generator model not initialized');
    llm.generator = vi.fn(async () => [{ generated_text: 'unexpected string' }]);
    llm.generator.tokenizer = { chat_template: 'template' };
    await expect(llm.chat([{ role: 'user', content: 'Hi' }])).rejects.toThrow('invalid assistant message');
  });
});

describe('BrowserRuntime streaming and cancel()', () => {
  // Emits one token per macrotask and honours the stopping criteria like model.generate().
  function streamingGenerator(tokens) {
    const generator = vi.fn(async (input, { streamer, stopping_criteria }) => {
      let text = '';
      for (const token of tokens) {
        await new Promise((resolve) => setTimeout(resolve, 0));
        if (stopping_criteria.interrupted) {
          break;
        }
        streamer?.put(token);
        text += token;
      }
      return [{ generated_text: Array.isArray(input) ? [...input, { role: 'assistant', content: text }] : text }];
    });
    generator.tokenizer = { chat_template: 'template' };
    return generator;
  }

  async function readAll(stream) {
    const chunks = [];
    for (let chunk; (chunk = await stream.read());) {
      chunks.push(chunk);
    }
    return chunks;
  }

  const messages = [{ role: 'user', content: 'Hi' }];

  it('streams chunks that join to the chat() answer', async () => {
    const llm = new BrowserRuntime({ generator_model: 'org/chat' });
    llm.generator = streamingGenerator([' Hello', ' there', '! ']);
    const stream = await llm.chatStream(JSON.stringify(messages));
    const chunks = await readAll(stream);
    expect(chunks).toEqual([' Hello', ' there', '! ']);
    expect(await stream.read()).toBe('');
    expect(chunks.join('').trim()).toBe(await llm.chat(messages));
    expect(llm.generator.mock.calls[0][1].return_full_text).toBe(false);
  });

  it('streams plain text generation without the prompt', async () => {
    const llm = new BrowserRuntime({ generator_model: 'gpt2', return_full_text: true });
    llm.generator = streamingGenerator(['a', 'b']);
    expect(await readAll(await llm.generateStream('prompt'))).toEqual(['a', 'b']);
    expect(llm.generator.mock.calls[0][1].return_full_text).toBe(false);
  });

  it('cancel() ends the stream early and leaves the instance reusable', async () => {
    const llm = new BrowserRuntime({ generator_model: 'org/chat' });
    llm.generator = streamingGenerator(['a', 'b', 'c', 'd']);
    expect(await llm.cancel()).toBe(false);
    const stream = await llm.chatStream(messages);
    expect(await stream.read()).toBe('a');
    expect(await llm.cancel()).toBe(true);
    expect((await readAll(stream)).length).toBeLessThan(3);
    expect(await llm.chat(messages)).toBe('abcd');
  });

  it('a new generation waits for a cancelled one instead of failing', async () => {
    const llm = new BrowserRuntime({ generator_model: 'org/chat' });
    llm.generator = streamingGenerator(['a', 'b', 'c']);
    const first = llm.chat(messages);
    await new Promise((resolve) => setTimeout(resolve, 0));
    llm.cancel();  // fire-and-forget, as Python does when a stream is closed early
    expect(await llm.chat(messages)).toBe('abc');
    expect((await first).length).toBeLessThan(3);
  });

  it('cancel() right after starting is not lost while the runtime loads', async () => {
    const llm = new BrowserRuntime({ generator_model: 'org/chat' });
    llm.generator = streamingGenerator(['a', 'b', 'c']);
    const first = llm.chat(messages);
    expect(await llm.cancel()).toBe(true);
    expect((await first).length).toBeLessThan(3);
    expect(await llm.chat(messages)).toBe('abc');
  });

  it('rejects concurrent generations and surfaces stream errors from read()', async () => {
    const llm = new BrowserRuntime({ generator_model: 'org/chat' });
    llm.generator = streamingGenerator(['a', 'b']);
    const first = llm.chat(messages);
    await new Promise((resolve) => setTimeout(resolve, 0));
    const second = await llm.chatStream(messages);
    await expect(second.read()).rejects.toThrow('Another generation is running');
    await expect(second.read()).rejects.toThrow('Another generation is running');
    expect(await first).toBe('ab');
  });

  it('rejects streaming without a generator', async () => {
    const llm = new BrowserRuntime({});
    await expect(llm.generateStream('x')).rejects.toThrow('Generator model not initialized');
    await expect(llm.chatStream(messages)).rejects.toThrow('Generator model not initialized');
  });
});

describe('BrowserRuntime token records', () => {
  // A vocabulary small enough to check probabilities by hand; 'é' spans two byte tokens.
  const vocab = ['<end>', ' Hello', ' world', '!', '\xC3', '\xA9'];
  const tokenizer = {
    decode: (ids, { skip_special_tokens = false } = {}) => {
      const text = ids.filter(id => !(skip_special_tokens && id === 0)).map(id => vocab[id]).join('');
      return text.replace('\xC3\xA9', 'é').replace(/[\xC3\xA9]$/, '�');
    },
  };

  // Each step feeds `logits` through the processors, picks the argmax, and streams it, like model.generate().
  function scriptedGenerator(steps) {
    const generator = vi.fn(async (input, { streamer, logits_processor, stopping_criteria }) => {
      streamer.put([[7n, 8n]]);  // the prompt
      for (const logits of steps) {
        await new Promise((resolve) => setTimeout(resolve, 0));
        if (stopping_criteria.interrupted) {
          break;
        }
        const processed = logits_processor.reduce((l, processor) => processor([], l), { data: Float32Array.from(logits) });
        const id = processed.data.indexOf(Math.max(...processed.data));
        streamer.put([[BigInt(id)]]);
      }
      streamer.end();
      return [{ generated_text: [...input, { role: 'assistant', content: '' }] }];
    });
    generator.tokenizer = { ...tokenizer, chat_template: 'template' };
    return generator;
  }

  async function records(stream) {
    const out = [];
    for (let chunk; (chunk = await stream.read());) {
      out.push(JSON.parse(chunk));
    }
    return out;
  }

  const messages = [{ role: 'user', content: 'Hi' }];
  const ln = Math.log;

  it('reports each token with its probability and the runner-up', async () => {
    const llm = new BrowserRuntime({ generator_model: 'org/chat' });
    llm.generator = scriptedGenerator([
      [-Infinity, ln(6), ln(3), ln(1), -Infinity, -Infinity],  // ' Hello' 0.6, runner-up ' world' 0.3
      [ln(1), ln(1), ln(8), -Infinity, -Infinity, -Infinity],  // ' world' 0.8
      [ln(9), ln(1), -Infinity, -Infinity, -Infinity, -Infinity],  // <end> is skipped
    ]);
    const tokens = await records(await llm.chatTokens(JSON.stringify(messages)));
    expect(tokens.map(t => t.text)).toEqual([' Hello', ' world']);
    expect(tokens[0].prob).toBeCloseTo(0.6);
    expect(tokens[0].alternatives).toHaveLength(1);
    expect(tokens[0].alternatives[0].text).toBe(' world');
    expect(tokens[0].alternatives[0].prob).toBeCloseTo(0.3);
    expect(tokens[1].prob).toBeCloseTo(0.8);
    expect(tokens[1].alternatives.map(a => a.text)).toEqual(['<end>']);  // ties keep vocabulary order
  });

  it('honours the number of alternatives and keeps split characters whole', async () => {
    const llm = new BrowserRuntime({ generator_model: 'gpt2' });
    llm.generator = scriptedGenerator([
      [-Infinity, -Infinity, ln(1), ln(2), ln(4), ln(3)],
      [-Infinity, ln(1), ln(2), ln(3), ln(1), ln(4)],
    ]);
    const tokens = await records(await llm.generateTokens('prompt', 2));
    expect(tokens.map(t => t.text)).toEqual(['', 'é']);
    expect(tokens[0].alternatives.map(a => a.text)).toEqual(['\uFFFD', '!']);  // a lone partial character decodes as U+FFFD
    expect(tokens.map(t => t.alternatives.length)).toEqual([2, 2]);
    expect(llm.generator.mock.calls[0][1].return_full_text).toBe(false);
    expect((await records(await llm.generateTokens('prompt', 0)))[1].alternatives).toEqual([]);
  });

  it('can be cancelled like text streams', async () => {
    const llm = new BrowserRuntime({ generator_model: 'org/chat' });
    const step = [-Infinity, ln(2), ln(1), -Infinity, -Infinity, -Infinity];
    llm.generator = scriptedGenerator([step, step, step, step]);
    const stream = await llm.chatTokens(messages);
    await stream.read();
    expect(await llm.cancel()).toBe(true);
    expect((await records(stream)).length).toBeLessThan(3);
  });
});

describe('BrowserRuntime.embed()', () => {
  let llm;

  beforeEach(() => {
    const config = { embedding_model: 'all-MiniLM-L6-v2' };
    llm = new BrowserRuntime(config);
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
  it('should create BrowserRuntime instances', () => {
    const llm = createRuntime({ generator_model: 'gpt2' });

    expect(llm).toBeInstanceOf(BrowserRuntime);
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
    expect(typeof globalThis.lmLite.loadModelToCache).toBe('function');
    expect(globalThis.loadModelToCache).toBeUndefined();  // only the lmLite namespace is global
    expect(globalThis.createRuntime).toBeUndefined();
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

    await globalThis.lmLite.loadModelToCache('gpt2', files);

    expect(mockCaches.open).toHaveBeenCalledWith('transformers-cache');
    expect(mockCache.put).toHaveBeenCalledTimes(2);

    const [url0, response0] = mockCache.put.mock.calls[0];
    expect(url0).toContain('Xenova/gpt2');
    expect(url0).toContain('config.json');
    expect(response0.headers.get('content-length')).toBe('3');
  });

  it('should keep explicit model paths unchanged', async () => {
    const mockCache = { put: vi.fn() };
    const mockCaches = {
      keys: vi.fn(async () => ['transformers-cache']),
      open: vi.fn(async () => mockCache),
    };
    globalThis.caches = mockCaches;

    await globalThis.lmLite.loadModelToCache('acme/gpt2', {
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

    await globalThis.lmLite.loadModelToCache('gpt2', {
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
    const llm = new BrowserRuntime({});

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

function mockModelCaches(contents) {
  const stores = Object.fromEntries(Object.entries(contents).map(([name, entries]) => {
    const map = new Map(Object.entries(entries));
    return [name, {
      keys: vi.fn(async () => [...map.keys()].map(url => ({ url }))),
      match: vi.fn(async ({ url }) => {
        const [length, body = ''] = [map.get(url)].flat();
        return new Response(body, { headers: length == null ? {} : { 'Content-Length': String(length) } });
      }),
      delete: vi.fn(async ({ url }) => map.delete(url)),
    }];
  }));
  vi.stubGlobal('caches', {
    keys: vi.fn(async () => Object.keys(stores)),
    open: vi.fn(async (name) => stores[name]),
  });
  return stores;
}

const HUB = 'https://huggingface.co';

describe('getCachedModels()', () => {
  beforeEach(() => vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('offline'); })));
  afterEach(() => vi.unstubAllGlobals());

  it('returns null when the Cache API is unavailable', async () => {
    vi.stubGlobal('caches', undefined);
    expect(await getCachedModels()).toBeNull();
  });

  it('groups Hub entries by model and revision with summed sizes', async () => {
    mockModelCaches({
      'transformers-cache': {
        [`${HUB}/Xenova/gpt2/resolve/main/config.json`]: 10,
        [`${HUB}/Xenova/gpt2/resolve/main/onnx/model.onnx`]: 90,
        [`${HUB}/Xenova/gpt2/resolve/v1/config.json`]: 5,
      },
      'app-cache': { [`${HUB}/acme/ignored/resolve/main/config.json`]: 1 },
    });

    expect(await getCachedModels()).toEqual([
      { model_id: 'Xenova/gpt2', revision: 'main', model_type: null, task: null, files: ['config.json', 'onnx/model.onnx'], file_sizes: { 'config.json': 10, 'onnx/model.onnx': 90 }, size: 100 },
      { model_id: 'Xenova/gpt2', revision: 'v1', model_type: null, task: null, files: ['config.json'], file_sizes: { 'config.json': 5 }, size: 5 },
    ]);
  });

  it('reports null size when a content-length header is missing', async () => {
    mockModelCaches({
      'transformers-cache': {
        [`${HUB}/Xenova/gpt2/resolve/main/config.json`]: 10,
        [`${HUB}/Xenova/gpt2/resolve/main/vocab.json`]: null,
      },
    });

    const [group] = await getCachedModels();
    expect(group.size).toBeNull();
    expect(group.file_sizes).toEqual({ 'config.json': 10, 'vocab.json': null });
  });

  it('reads the architecture from cached config.json and the task from the Hub', async () => {
    mockModelCaches({
      'transformers-cache': {
        [`${HUB}/Xenova/gpt2/resolve/main/config.json`]: [20, '{"model_type": "gpt2"}'],
        [`${HUB}/Xenova/bert/resolve/main/config.json`]: [3, 'not json'],
      },
    });
    fetch.mockImplementation(async (url) => url.endsWith('/gpt2')
      ? new Response(JSON.stringify({ pipeline_tag: 'text-generation' }))
      : new Response('', { status: 404 }));

    const [gpt2, bert] = await getCachedModels();
    expect(gpt2).toMatchObject({ model_type: 'gpt2', task: 'text-generation' });
    expect(bert).toMatchObject({ model_type: null, task: null });
    expect(fetch).toHaveBeenCalledWith(`${HUB}/api/models/Xenova/gpt2`, expect.objectContaining({ signal: expect.anything() }));
  });

  it('groups non-Hub entries under a null model', async () => {
    const url = 'https://example.test/files/models/Xenova/gpt2/config.json';
    mockModelCaches({ 'transformers-cache': { [url]: 3 } });

    expect(await getCachedModels()).toEqual([
      { model_id: null, revision: null, model_type: null, task: null, files: [url], file_sizes: { [url]: 3 }, size: 3 },
    ]);
  });
});

describe('clearModelCache(modelName)', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('deletes only entries of the resolved model across revisions', async () => {
    const stores = mockModelCaches({
      'transformers-cache': {
        [`${HUB}/Xenova/gpt2/resolve/main/config.json`]: 1,
        [`${HUB}/Xenova/gpt2/resolve/v1/config.json`]: 1,
        [`${HUB}/acme/gpt2/resolve/main/config.json`]: 1,
        [`${HUB}/Xenova/gpt2-large/resolve/main/config.json`]: 1,
      },
      'app-cache': { [`${HUB}/Xenova/gpt2/resolve/main/config.json`]: 1 },
    });
    const deleteDatabase = vi.fn();
    vi.stubGlobal('indexedDB', { deleteDatabase });

    const result = await clearModelCache('gpt2@v1');

    expect(result).toEqual({
      model_id: 'Xenova/gpt2',
      deleted_files: [
        `${HUB}/Xenova/gpt2/resolve/main/config.json`,
        `${HUB}/Xenova/gpt2/resolve/v1/config.json`,
      ],
    });
    expect((await stores['transformers-cache'].keys()).map(r => r.url)).toEqual([
      `${HUB}/acme/gpt2/resolve/main/config.json`,
      `${HUB}/Xenova/gpt2-large/resolve/main/config.json`,
    ]);
    expect(stores['app-cache'].delete).not.toHaveBeenCalled();
    expect(deleteDatabase).not.toHaveBeenCalled();
  });

  it('matches explicit model IDs exactly', async () => {
    mockModelCaches({
      'transformers-cache': {
        [`${HUB}/Xenova/gpt2/resolve/main/config.json`]: 1,
        [`${HUB}/acme/gpt2/resolve/main/config.json`]: 1,
      },
    });

    const result = await clearModelCache('acme/gpt2');

    expect(result.deleted_files).toEqual([`${HUB}/acme/gpt2/resolve/main/config.json`]);
  });
});

describe('getModelInfo()', () => {
  const SIZES = { 'config.json': 10, 'onnx/model_quantized.onnx': 1000 };
  let registry;

  beforeEach(async () => {
    ({ ModelRegistry: registry } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2'));
    registry.get_available_dtypes.mockReset().mockResolvedValue(['fp32', 'q8', 'q4']);
    registry.get_file_metadata.mockReset().mockImplementation(async (_id, file) => ({ exists: true, size: SIZES[file] }));
    registry.is_pipeline_cached_files.mockReset();
    vi.stubGlobal('caches', {});
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ pipeline_tag: 'text-generation' }))));
  });
  afterEach(() => vi.unstubAllGlobals());

  const cachedFlags = (...flags) => registry.is_pipeline_cached_files.mockResolvedValue({
    files: Object.keys(SIZES).map((file, i) => ({ file, cached: flags[i] })),
  });

  it.each([
    [[false, false], 'not_cached'],
    [[true, false], 'partial'],
    [[true, true], 'cached'],
  ])('reports cache flags %j as %s with total download size', async (flags, status) => {
    cachedFlags(...flags);
    const info = await getModelInfo('gpt2');

    expect(info).toMatchObject({
      model_id: 'Xenova/gpt2', revision: 'main', task: 'text-generation', backend: 'wasm',
      dtype: 'q8', cache_status: status, download_bytes: 1010, error: null,
    });
    expect(info.files[1]).toEqual({ name: 'onnx/model_quantized.onnx', size: 1000, cached: flags[1] });
    expect(registry.is_pipeline_cached_files).toHaveBeenCalledWith(
      'text-generation', 'Xenova/gpt2', { revision: 'main', device: 'wasm', dtype: 'q8' });
  });

  it('keeps an explicit dtype and revision and skips dtype discovery', async () => {
    cachedFlags(false, false);
    const info = await getModelInfo('org/m@v1', 'translation', 'webgpu', 'q4');

    expect(registry.get_available_dtypes).not.toHaveBeenCalled();
    expect(fetch).not.toHaveBeenCalled();
    expect(info).toMatchObject({ model_id: 'org/m', revision: 'v1', dtype: 'q4', backend: 'webgpu' });
    expect(registry.is_pipeline_cached_files).toHaveBeenCalledWith(
      'translation', 'org/m', { revision: 'v1', device: 'webgpu', dtype: 'q4' });
  });

  it('detects the task from the Hub pipeline tag when none is given', async () => {
    cachedFlags(false, false);
    fetch.mockResolvedValue(new Response(JSON.stringify({ pipeline_tag: 'feature-extraction' })));
    const info = await getModelInfo('org/embed');

    expect(info.task).toBe('feature-extraction');
    expect(String(fetch.mock.calls[0][0])).toBe('https://huggingface.co/api/models/org/embed');
    expect(registry.is_pipeline_cached_files.mock.calls[0][0]).toBe('feature-extraction');
  });

  it.each([
    ['no tag', async () => new Response('{}')],
    ['offline', async () => { throw new TypeError('Failed to fetch'); }],
  ])('reports a null task and an error when detection fails (%s)', async (_case, response) => {
    fetch.mockImplementation(response);
    const info = await getModelInfo('org/custom');

    expect(info).toMatchObject({ task: null, cache_status: 'unknown' });
    expect(info.error).toContain('No task found on the Hub for org/custom');
    expect(registry.is_pipeline_cached_files).not.toHaveBeenCalled();
  });

  it('leaves feature-extraction on the Transformers.js default dtype', async () => {
    cachedFlags(true, true);
    const info = await getModelInfo('org/embed', 'feature-extraction');

    expect(info.dtype).toBeNull();
    expect(registry.is_pipeline_cached_files.mock.calls[0][2]).not.toHaveProperty('dtype');
  });

  it('reports unknown sizes as null instead of guessing (offline)', async () => {
    cachedFlags(true, false);
    registry.get_file_metadata.mockImplementation(async (_id, file) => {
      if (file === 'config.json') return { exists: true, size: 10, fromCache: true };
      throw new TypeError('Failed to fetch');
    });
    const info = await getModelInfo('gpt2');

    expect(info.cache_status).toBe('partial');
    expect(info.files.map(f => f.size)).toEqual([10, null]);
    expect(info.download_bytes).toBeNull();
    expect(info.error).toBeNull();
  });

  it('reports unknown cache status when the Cache API is unavailable', async () => {
    vi.stubGlobal('caches', undefined);
    cachedFlags(false, false);
    expect((await getModelInfo('gpt2')).cache_status).toBe('unknown');
  });

  it('degrades to unknown with an error message when metadata cannot be fetched', async () => {
    registry.get_available_dtypes.mockRejectedValue(new Error('Could not locate file config.json'));
    const info = await getModelInfo('gpt2');

    expect(info).toMatchObject({ cache_status: 'unknown', download_bytes: null, files: [], dtype: null });
    expect(info.error).toContain('Could not locate file');
  });

  it('fails clearly when no default dtype is published', async () => {
    registry.get_available_dtypes.mockResolvedValue(['q4']);
    expect((await getModelInfo('gpt2')).error).toContain('No q8, int8, fp32 ONNX weights');
  });
});

describe('getModelTask()', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('returns the Hub pipeline tag for the resolved model, without the revision', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ pipeline_tag: 'translation' }))));

    expect(await getModelTask('opus-mt-en-fr@v1')).toBe('translation');
    expect(fetch).toHaveBeenCalledWith(
      'https://huggingface.co/api/models/Xenova/opus-mt-en-fr', expect.objectContaining({ signal: expect.anything() }));
  });

  it.each([
    ['untagged', async () => new Response('{}')],
    ['missing', async () => new Response('', { status: 404 })],
    ['offline', async () => { throw new TypeError('Failed to fetch'); }],
  ])('returns null for %s models', async (_case, response) => {
    vi.stubGlobal('fetch', vi.fn(response));
    expect(await getModelTask('org/model')).toBeNull();
  });
});

describe('loadTokenizer()', () => {
  const tokenizerJson = { model: { type: 'Unigram', vocab: ['<pad>', 'a', 'b'] } };
  const json = (body) => new Response(JSON.stringify(body));
  let requests;

  // Serves the given URL suffixes; everything else is a 404.
  const serve = (files) => {
    requests = [];
    vi.stubGlobal('fetch', vi.fn(async (url) => {
      requests.push(String(url));
      const match = Object.entries(files).find(([suffix]) => String(url).endsWith(suffix));
      return match ? json(match[1]) : new Response('', { status: 404, statusText: 'Not Found' });
    }));
  };

  afterEach(() => vi.unstubAllGlobals());

  it('loads a hosted model tokenizer without weights or a Hub lookup', async () => {
    serve({
      '/models/manifest.json': { models: ['org/potion'] },
      '/models/org/potion/tokenizer.json': tokenizerJson,
      '/models/org/potion/tokenizer_config.json': { tokenizer_class: 'BertTokenizerFast' },
    });
    const tokenizer = await loadTokenizer('org/potion', null, null, 'https://example.test/models/');

    expect(requests.slice(1).sort()).toEqual([
      'https://example.test/models/org/potion/tokenizer.json',
      'https://example.test/models/org/potion/tokenizer_config.json',
    ]);
    expect(tokenizer.vocab_size).toBe(3);
    expect(JSON.parse(tokenizer.encode('ab'))).toEqual([97, 98]);
    expect(JSON.parse(tokenizer.encode('ab', true))).toEqual([97, 98, 0]);
    expect(tokenizer.decode('[97,98]')).toBe('ab');
  });

  it('fetches Hub files at the requested revision and tolerates a missing tokenizer_config.json', async () => {
    serve({ '/tokenizer.json': tokenizerJson });
    const tokenizer = await loadTokenizer('gpt2@v1');

    expect(requests.sort()).toEqual([
      'https://huggingface.co/Xenova/gpt2/resolve/v1/tokenizer.json',
      'https://huggingface.co/Xenova/gpt2/resolve/v1/tokenizer_config.json',
    ]);
    expect(tokenizer.vocab_size).toBe(3);
  });

  it('resolves tokenizer_json paths against the JupyterLite site root', async () => {
    vi.stubGlobal('location', new URL('https://example.test/demo/extensions/kernel/worker.js'));
    serve({ '/tok.json': tokenizerJson });
    await loadTokenizer(null, 'files/models/org/potion/tok.json');

    expect(requests).toEqual(['https://example.test/demo/files/models/org/potion/tok.json']);
  });

  it('picks the configured tokenizer class and falls back to the base class', async () => {
    const transformers = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2');
    const construct = vi.spyOn(transformers.BertTokenizer.prototype, 'get_vocab');
    serve({ '/tok.json': tokenizerJson, '/bert.json': { tokenizer_class: 'BertTokenizerFast' }, '/x.json': { tokenizer_class: 'pipeline' } });

    await loadTokenizer(null, 'https://example.test/tok.json', 'https://example.test/bert.json');
    expect(construct).toHaveBeenCalledTimes(1);
    await loadTokenizer(null, 'https://example.test/tok.json', 'https://example.test/x.json');
    expect(construct).toHaveBeenCalledTimes(1);
    construct.mockRestore();
  });

  it.each([
    [[null, 'https://example.test/missing.json'], '404 Not Found'],
    [[null, 'https://example.test/tok.json', 'https://example.test/missing.json'], '404 Not Found'],
    [[null, 'data:application/json,{}'], 'HTTP(S)'],
  ])('fails at load time with the source in the message (%j)', async (args, message) => {
    serve({ '/tok.json': tokenizerJson });
    await expect(loadTokenizer(...args)).rejects.toThrow(`Failed to load tokenizer from '${args[1]}'`);
    await expect(loadTokenizer(...args)).rejects.toThrow(message);
  });
});

describe('bare ONNX models', () => {
  const ortUrl = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.2.3/dist/ort.webgpu.min.mjs';

  it('loads a session with the requested provider and describes it', async () => {
    const ort = await import(ortUrl);
    const bytes = new Uint8Array([1, 2, 3]);

    const model = await loadOnnxModel(bytes, 'webgpu');

    expect(ort.InferenceSession.create).toHaveBeenCalledWith(bytes, { executionProviders: ['webgpu'] });
    expect(ort.env.wasm.wasmPaths).toEqual({ mjs: 'a.mjs', wasm: 'a.wasm' });  // Transformers.js's own wasm files
    const info = JSON.parse(model.describe());
    expect(info).toMatchObject({ backend: 'webgpu', execution_provider: 'webgpu', size: 3 });
    expect(info.sha256).toHaveLength(64);
    expect(info.inputs).toEqual([{ name: 'x', dtype: 'float32', shape: ['batch', 3] }]);
  });

  describe('falling back from WebGPU', () => {
    const f16 = 'no available backend found. ERR: [webgpu] Error: Program Transpose requires f16 but the device does not support it';

    it('retries on wasm after a WebGPU-specific failure and says so', async () => {
      const ort = await import(ortUrl);
      ort.InferenceSession.create.mockRejectedValueOnce(new Error(f16));
      const bytes = new Uint8Array([1, 2, 3]);

      const info = JSON.parse((await loadOnnxModel(bytes, 'webgpu')).describe());

      expect(ort.InferenceSession.create.mock.calls.slice(-2).map(c => c[1].executionProviders)).toEqual([['webgpu'], ['wasm']]);
      expect(info).toMatchObject({ backend: 'wasm', execution_provider: 'wasm' });
      expect(info.backend_fallback_reason).toContain('requires f16');
    });

    it('uses wasm without an adapter, and not at all with backend_fallback false', async () => {
      const ort = await import(ortUrl);
      vi.stubGlobal('navigator', { gpu: { requestAdapter: async () => null } });
      ort.InferenceSession.create.mockClear();

      const info = JSON.parse((await loadOnnxModel(new Uint8Array([1]), 'webgpu')).describe());
      expect(info).toMatchObject({ backend: 'wasm', backend_fallback_reason: 'no WebGPU adapter in this browser' });
      expect(ort.InferenceSession.create.mock.calls.map(c => c[1].executionProviders)).toEqual([['wasm']]);

      await loadOnnxModel(new Uint8Array([1]), 'webgpu', null, true, false);
      expect(ort.InferenceSession.create.mock.calls.at(-1)[1].executionProviders).toEqual(['webgpu']);
    });

    it('does not retry a failure that is not about WebGPU, or any wasm failure', async () => {
      const ort = await import(ortUrl);
      ort.InferenceSession.create.mockClear();
      ort.InferenceSession.create.mockRejectedValueOnce(new Error('Out of memory'));
      await expect(loadOnnxModel(new Uint8Array([1]), 'webgpu')).rejects.toThrow('Out of memory');
      ort.InferenceSession.create.mockRejectedValueOnce(new Error(f16));
      await expect(loadOnnxModel(new Uint8Array([1]), 'wasm')).rejects.toThrow('requires f16');
      expect(ort.InferenceSession.create).toHaveBeenCalledTimes(2);
    });
  });

  it('fetches a URL, runs on raw buffers and returns one buffer per output', async () => {
    vi.stubGlobal('caches', undefined);  // no Cache API: the model is fetched, not kept
    vi.stubGlobal('fetch', vi.fn(async () => new Response(new Uint8Array([9]))));
    const model = await loadOnnxModel('https://example.test/m.onnx');
    const input = new Uint8Array(new Float32Array([1, 2, 3]).buffer);

    const [meta, y] = await model.run(JSON.stringify([{ name: 'x', dtype: 'float32', dims: [1, 3] }]), [input]);

    expect(JSON.parse(meta)).toEqual([{ name: 'y', dtype: 'float32', dims: [1, 3] }]);
    expect([...new Float32Array(y.buffer)]).toEqual([2, 4, 6]);
    await expect(model.run(JSON.stringify([{ name: 'x', dtype: 'float128', dims: [1] }]), [input]))
      .rejects.toThrow("Unsupported dtype 'float128' for input 'x'");
    vi.unstubAllGlobals();
  });

  it('keeps URLs in the Cache API, serves repeats from it and checks sha256', async () => {
    const store = new Map();
    const cache = {
      match: vi.fn(async (url) => store.get(url)?.clone()),
      put: vi.fn(async (url, response) => { store.set(url, response); }),
      delete: vi.fn(async (url) => store.delete(url)),
    };
    vi.stubGlobal('caches', { open: vi.fn(async () => cache) });
    const fetchMock = vi.fn(async () => new Response(new Uint8Array([1, 2, 3])));
    vi.stubGlobal('fetch', fetchMock);
    const status = async (...args) => {
      const model = await loadOnnxModel('https://example.test/m.onnx', 'wasm', ...args);
      return JSON.parse(model.describe());
    };

    const first = await status();
    expect(first.cache_status).toBe('downloaded');
    expect((await status(first.sha256)).cache_status).toBe('cached');
    expect(fetchMock).toHaveBeenCalledTimes(1);

    expect((await status(null, false)).cache_status).toBeNull();  // cache off: fetched, not stored
    expect(fetchMock).toHaveBeenCalledTimes(2);

    await expect(status('0'.repeat(64))).rejects.toThrow('SHA-256 mismatch');  // stale entry dropped, refetched, rejected
    expect(cache.delete).toHaveBeenCalledWith('https://example.test/m.onnx');
    await expect(loadOnnxModel(new Uint8Array([1]), 'wasm', '0'.repeat(64))).rejects.toThrow('SHA-256 mismatch');
    vi.unstubAllGlobals();
  });

  it('reports a failed download and releases the session on dispose', async () => {
    vi.stubGlobal('caches', undefined);
    vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 404, statusText: 'Not Found' })));
    await expect(loadOnnxModel('https://example.test/missing.onnx')).rejects.toThrow('404 Not Found');
    vi.unstubAllGlobals();

    const ort = await import(ortUrl);
    const model = await loadOnnxModel(new Uint8Array([1]));
    await model.dispose();
    expect(ort.session.release).toHaveBeenCalled();
  });
});

describe('runTasks', () => {
  const upper = (items) => items.map((text) => ({ translation_text: text.toUpperCase() }));

  function translator(taskPipeline) {
    const llm = new BrowserRuntime({ task: 'translation' });
    llm.taskPipeline = vi.fn(taskPipeline);
    return llm;
  }

  it('runs batches, reports progress after each one and keeps the order', async () => {
    const llm = translator(async (items) => upper([items].flat()));
    const progress = vi.fn();

    const { results, errors, cancelled } = await llm.runTasks(['a', 'b', 'c', 'd', 'e'], 2, '{"max_new_tokens": 4}', progress);

    expect(results.map((r) => r.translation_text)).toEqual(['A', 'B', 'C', 'D', 'E']);
    expect([errors, cancelled]).toEqual([[], false]);
    expect(llm.taskPipeline.mock.calls.map((call) => call[0])).toEqual([['a', 'b'], ['c', 'd'], ['e']]);
    expect(llm.taskPipeline).toHaveBeenCalledWith(['a', 'b'], { max_new_tokens: 4 });
    expect(progress.mock.calls).toEqual([[2, 5], [4, 5], [5, 5]]);  // ceil(5 / 2) calls
  });

  it('gives each batch its own copy of the options', async () => {
    const llm = new BrowserRuntime({ task: 'zero-shot-classification' });
    llm.taskPipeline = vi.fn(async (items) => items.map((sequence) => ({ sequence })));

    await llm.runTasks(['a', 'b'], 1, { candidate_labels: ['x', 'y'] });

    expect(llm.taskPipeline.mock.calls.map((call) => call[1])).toEqual([['x', 'y'], ['x', 'y']]);  // not consumed by the first
  });

  it('throws the first error unless asked to continue', async () => {
    const llm = translator(async (items) => {
      if (items.includes('bad')) { throw new Error('undecodable'); }
      return upper(items);
    });

    await expect(llm.runTasks(['a', 'bad', 'c'], 2)).rejects.toThrow('translation failed: undecodable');
    expect(llm._batchRun).toBeNull();  // the instance stays usable
    expect((await llm.runTasks(['a'], 2)).results).toHaveLength(1);
  });

  it('retries a failed batch input by input and reports the bad ones', async () => {
    const llm = translator(async (items) => {
      if (items.includes('bad')) { throw new Error('undecodable'); }
      return upper(items);
    });
    const progress = vi.fn();

    const { results, errors } = await llm.runTasks(['a', 'bad', 'c', 'd'], 3, {}, progress, true);

    expect(results).toEqual([{ translation_text: 'A' }, null, { translation_text: 'C' }, { translation_text: 'D' }]);
    expect(errors).toEqual([[1, 'translation failed: undecodable']]);
    expect(progress.mock.calls).toEqual([[3, 4], [4, 4]]);
  });

  it('runs speech recognition one clip per call', async () => {
    const llm = new BrowserRuntime({ task: 'automatic-speech-recognition' });
    llm.taskPipeline = vi.fn(async (clip) => ({ text: String(clip.length) }));

    const { results } = await llm.runTasks([new Float32Array(3), new Float32Array(5)], 8);

    expect(results).toEqual([{ text: '3' }, { text: '5' }]);
    expect(llm.taskPipeline).toHaveBeenCalledTimes(2);
  });

  it('stops after the current batch on cancel() and stays usable', async () => {
    const llm = translator(async (items) => upper([items].flat()));
    const progress = vi.fn(() => { llm.cancel(); });

    const run = await llm.runTasks(['a', 'b', 'c', 'd'], 2, {}, progress);

    expect(run.cancelled).toBe(true);
    expect(run.results.map((r) => r.translation_text)).toEqual(['A', 'B']);
    expect(await llm.cancel()).toBe(false);  // nothing is running now
    expect((await llm.runTasks(['x'], 2)).cancelled).toBe(false);
  });

  it('refuses a second run while one is active and rejects results of the wrong length', async () => {
    let release;
    const llm = translator(() => new Promise((resolve) => { release = () => resolve(upper(['a'])); }));
    const first = llm.runTasks(['a'], 1);
    await Promise.resolve();

    await expect(llm.runTasks(['b'], 1)).rejects.toThrow('Another map() is running');
    release();
    await first;

    const short = translator(async () => []);
    await expect(short.runTasks(['a', 'b'], 2)).rejects.toThrow('returned 0 results for 2 inputs');
  });
});

describe('speech tasks', () => {
  it.each(['text-to-speech', 'text-to-audio'])('%s returns the sampling rate and the samples as one byte buffer', async (task) => {
    const llm = new BrowserRuntime({ task });
    const audio = new Float32Array([0.5, -0.25, 1]);
    llm.taskPipeline = vi.fn(async () => ({ audio, sampling_rate: 16000 }));

    const [rate, bytes] = await llm.runTask('Hello', '{"speaker": "x"}');

    expect(llm.taskPipeline).toHaveBeenCalledWith('Hello', { speaker: 'x' });
    expect(rate).toBe(16000);
    expect(bytes).toBeInstanceOf(Uint8Array);
    expect([...new Float32Array(bytes.buffer)]).toEqual([0.5, -0.25, 1]);
  });

  it('copies the samples of an audio view into its own buffer', async () => {
    const llm = new BrowserRuntime({ task: 'text-to-speech' });
    const shared = new Float32Array([9, 1, 2, 9]);
    llm.taskPipeline = vi.fn(async () => ({ audio: shared.subarray(1, 3), sampling_rate: 8000 }));

    const [, bytes] = await llm.runTask('Hi');

    expect(bytes.byteLength).toBe(8);
    expect([...new Float32Array(bytes.buffer)]).toEqual([1, 2]);
  });

  it('runs one text per call in runTasks', async () => {
    const llm = new BrowserRuntime({ task: 'text-to-speech' });
    llm.taskPipeline = vi.fn(async (text) => ({ audio: new Float32Array(text.length), sampling_rate: 16000 }));

    const { results } = await llm.runTasks(['One.', 'Three'], 8);

    expect(llm.taskPipeline.mock.calls.map((call) => call[0])).toEqual(['One.', 'Three']);
    expect(results.map(([rate, bytes]) => [rate, bytes.byteLength / 4])).toEqual([[16000, 4], [16000, 5]]);
  });
});

describe('microphone recording', () => {
  it('encodes samples as a 16-bit PCM mono WAV and clips them', () => {
    const wav = encodeWav(new Float32Array([0, 0.5, -0.5, 1, -1, 2]), 16000);
    const view = new DataView(wav.buffer);
    const text = (offset) => String.fromCharCode(...wav.subarray(offset, offset + 4));

    expect([text(0), text(8), text(12), text(36)]).toEqual(['RIFF', 'WAVE', 'fmt ', 'data']);
    expect(wav.byteLength).toBe(44 + 12);
    expect(view.getUint32(4, true)).toBe(36 + 12);
    expect([view.getUint16(20, true), view.getUint16(22, true), view.getUint32(24, true), view.getUint16(34, true)])
      .toEqual([1, 1, 16000, 16]);  // PCM, mono, rate, bits
    expect(view.getUint32(40, true)).toBe(12);
    expect([0, 1, 2, 3, 4, 5].map((i) => view.getInt16(44 + i * 2, true))).toEqual([0, 16384, -16383, 32767, -32767, 32767]);  // Math.round rounds -16383.5 up
  });

  describe('AudioRecorder', () => {
    const track = { stop: vi.fn() };
    let recorder;

    beforeEach(() => {
      track.stop.mockClear();
      const stream = { getTracks: () => [track] };
      vi.stubGlobal('navigator', { mediaDevices: { getUserMedia: vi.fn(async () => stream) } });
      vi.stubGlobal('MediaRecorder', class {
        constructor() { recorder = this; this.mimeType = 'audio/webm'; }
        start() {}
        stop() {
          this.ondataavailable({ data: new Blob([new Uint8Array([1, 2, 3])]) });
          this.onstop();
        }
      });
      const decoded = { duration: 0.5, numberOfChannels: 1, sampleRate: 48000 };
      vi.stubGlobal('AudioContext', class {
        decodeAudioData = vi.fn(async () => decoded);
        close = vi.fn(async () => {});
      });
      vi.stubGlobal('OfflineAudioContext', class {
        constructor(channels, length, rate) { Object.assign(this, { channels, length, rate }); this.destination = {}; }
        createBufferSource() { return { connect: vi.fn(), start: vi.fn(), set buffer(value) { this.value = value; } }; }
        async startRendering() {
          return { numberOfChannels: this.channels, getChannelData: () => new Float32Array(this.length).fill(0.25) };
        }
      });
    });

    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it('records, releases the microphone and returns a 16 kHz mono WAV', async () => {
      const audio = new AudioRecorder();
      expect(audio.recording).toBe(false);

      await audio.start();
      expect(audio.recording).toBe(true);
      await expect(audio.start()).rejects.toThrow('Already recording');
      const wav = await audio.stop();

      expect(audio.recording).toBe(false);
      expect(track.stop).toHaveBeenCalled();
      const view = new DataView(wav.buffer);
      expect(view.getUint32(24, true)).toBe(16000);
      expect(view.getUint32(40, true)).toBe(0.5 * 16000 * 2);  // half a second of 16-bit samples
      expect(recorder).toBeDefined();
      expect(audio.info).toEqual({
        seconds: 0.5, peak: 0.25, channels: 1, sampleRate: 48000, mimeType: 'audio/webm', raw: false, gain: 1,
      });
    });

    it('can record again, and refuses to stop when it is not recording', async () => {
      const audio = new AudioRecorder();
      await expect(audio.stop()).rejects.toThrow('Not recording');

      await audio.start();
      await audio.stop();
      await audio.start();

      expect(audio.recording).toBe(true);
    });

    it('asks for the browser\'s processing by default, and for none (with two channels) when raw', async () => {
      const audio = new AudioRecorder();

      await audio.start();
      await audio.stop();
      await audio.start({ raw: true });
      const info = (await audio.stop(), audio.info);

      expect(navigator.mediaDevices.getUserMedia.mock.calls.map((call) => call[0])).toEqual([
        { audio: true },
        { audio: { channelCount: { ideal: 2 }, echoCancellation: false, noiseSuppression: false, autoGainControl: false } },
      ]);
      // A raw recording has no gain control, so a quiet one (peak 0.25) is scaled up to a peak of 0.9.
      expect(info).toMatchObject({ raw: true, peak: 0.25, gain: 3.6 });
    });

    it('keeps the channels apart until they are mixed, and describes a stereo recording', async () => {
      vi.stubGlobal('AudioContext', class {
        decodeAudioData = vi.fn(async () => ({ duration: 0.5, numberOfChannels: 2, sampleRate: 44100 }));
        close = vi.fn(async () => {});
      });
      const audio = new AudioRecorder();

      await audio.start();
      await audio.stop();

      expect(audio.info).toMatchObject({ channels: 2, sampleRate: 44100, peak: 0.25 });  // two equal channels average to themselves
    });

    it('explains when the page cannot record', async () => {
      vi.stubGlobal('navigator', {});

      await expect(new AudioRecorder().start()).rejects.toThrow('HTTPS or localhost');
    });
  });
});

describe('mixChannels', () => {
  const tone = (sign = 1, length = 100) => Float32Array.from({ length }, (_, i) => sign * 0.5 * Math.sin(i / 5));
  const peak = (data) => data.reduce((max, value) => Math.max(max, Math.abs(value)), 0);

  it('returns a single channel as it is', () => {
    const channel = tone();

    expect(mixChannels([channel])).toBe(channel);
  });

  it('averages channels that agree', () => {
    const mixed = mixChannels([tone(), tone()]);

    expect([...mixed]).toEqual([...tone()]);
  });

  it('keeps the louder channel when the average would cancel out (opposite polarity)', () => {
    const mixed = mixChannels([tone(1), tone(-1)]);

    expect(peak(mixed)).toBeCloseTo(0.5, 5);  // not silence
  });

  it('keeps the loudest channel when the other is nearly silent, and stays silent for silence', () => {
    const quiet = new Float32Array(100);

    expect(peak(mixChannels([quiet, tone()]))).toBeCloseTo(0.5, 5);
    expect(peak(mixChannels([quiet, new Float32Array(100)]))).toBe(0);
  });
});

describe('image output tasks', () => {
  // Like Transformers.js's RawImage: pixels, a size, and toBlob() for a PNG.
  const image = (width, height, channels, fill = 7) => ({
    width, height, channels, data: new Uint8ClampedArray(width * height * channels).fill(fill),
    toBlob: vi.fn(async () => new Blob([new Uint8Array([0x89, 0x50, 0x4e, 0x47, width, height])])),
  });
  const tensor = (rows, columns) => ({
    dims: [rows, columns], ort_tensor: {}, data: Float32Array.from({ length: rows * columns }, (_, i) => i / 2),
  });
  const unpack = ([structure, ...buffers]) => ({ structure: JSON.parse(structure), buffers });

  function pipe(task, output) {
    const llm = new BrowserRuntime({ task });
    llm.taskPipeline = vi.fn(async () => output);
    return llm;
  }

  it('returns an image as one PNG buffer with its description', async () => {
    const llm = pipe('background-removal', image(3, 2, 4));

    const { structure, buffers } = unpack(await llm.runTask('https://example.test/a.jpg', '{}'));

    expect(structure).toEqual({ $buffer: 0, width: 3, height: 2, channels: 4 });
    expect([...buffers[0]]).toEqual([0x89, 0x50, 0x4e, 0x47, 3, 2]);
    expect(buffers).toHaveLength(1);
  });

  it('returns the raw pixels with as_array, and does not pass that option to Transformers.js', async () => {
    const llm = pipe('image-to-image', image(2, 2, 3, 9));

    const { buffers } = unpack(await llm.runTask('https://example.test/a.jpg', '{"as_array": true, "other": 1}'));

    expect([...buffers[0]]).toEqual(new Array(12).fill(9));
    expect(llm.taskPipeline.mock.calls[0][1]).toEqual({ other: 1 });
  });

  it('packs a depth map as an image plus a float32 tensor', async () => {
    const llm = pipe('depth-estimation', { predicted_depth: tensor(2, 3), depth: image(3, 2, 1) });

    const { structure, buffers } = unpack(await llm.runTask('https://example.test/a.jpg', '{}'));

    // Each description points at its buffer, whatever the order the buffers were made in.
    const { predicted_depth: depth, depth: map } = structure;
    expect([depth.dtype, depth.shape, map.width, map.height, map.channels]).toEqual(['float32', [2, 3], 3, 2, 1]);
    expect(new Set([depth.$buffer, map.$buffer])).toEqual(new Set([0, 1]));
    expect([...new Float32Array(buffers[depth.$buffer].buffer)]).toEqual([0, 0.5, 1, 1.5, 2, 2.5]);
    expect([...buffers[map.$buffer]]).toEqual([0x89, 0x50, 0x4e, 0x47, 3, 2]);
  });

  it('packs every segment of a segmentation with its label and score', async () => {
    const llm = pipe('image-segmentation', [
      { label: 'cat', score: 0.9, mask: image(2, 2, 1) },
      { label: 'sky', score: null, mask: image(2, 2, 1) },
    ]);

    const { structure, buffers } = unpack(await llm.runTask('https://example.test/a.jpg', '{}'));

    expect(structure).toEqual([
      { label: 'cat', score: 0.9, mask: { $buffer: 0, width: 2, height: 2, channels: 1 } },
      { label: 'sky', score: null, mask: { $buffer: 1, width: 2, height: 2, channels: 1 } },
    ]);
    expect(buffers).toHaveLength(2);
  });

  it('handles a batch of images as a list of results', async () => {
    const llm = pipe('background-removal', [image(1, 1, 4), image(2, 1, 4)]);

    const { structure } = unpack(await llm.runTask([new Uint8Array([1]), new Uint8Array([2])], '{}'));

    expect(structure.map((item) => item.width)).toEqual([1, 2]);
  });

  it('runs one image per call in runTasks', async () => {
    const llm = pipe('image-to-image', image(1, 1, 3));

    const { results } = await llm.runTasks(['https://example.test/a.jpg', 'https://example.test/b.jpg'], 8);

    expect(llm.taskPipeline).toHaveBeenCalledTimes(2);
    expect(results.map(([structure]) => JSON.parse(structure).width)).toEqual([1, 1]);
  });
});
