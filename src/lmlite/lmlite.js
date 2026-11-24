// lmLite JS runtime

const DEFAULT_MODEL_NAMESPACE = 'Xenova';

// Pinned: scripts/lite-build.sh self-hosts this exact version (and its onnxruntime-web files) in each site.
const TRANSFORMERS_VERSION = '4.3.2';
const TRANSFORMERS_URL = `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${TRANSFORMERS_VERSION}`;

// What the runtimes in this project report when they cannot allocate memory (WASM RangeError, ONNX Runtime
// allocation failures, the WebGPU buffer limit).
const MEMORY_ERROR = /out of memory|failed to allocate|bad_alloc/i;

// The error for a model that failed to load or to warm up: it keeps the runtime's own message, and names the stage,
// the backend, the dtypes tried and the runtime versions, with advice that fits the failure.
function loadFailure({ stage, what, model, backend, error, env }) {
  const message = String(error?.message ?? error);
  const dtypes = error?.dtypesTried?.length ? `, dtype ${error.dtypesTried.join(' then ')}` : '';
  const onnxVersion = env?.backends?.onnx?.versions?.web;
  const runtime = `Transformers.js ${TRANSFORMERS_VERSION}${onnxVersion ? `, onnxruntime-web ${onnxVersion}` : ''}`;
  const advice = MEMORY_ERROR.test(message)
    ? 'The runtime ran out of memory, which usually means the model is too large for this browser or device, ' +
      'not that it is unsupported. Try a smaller model or dtype (for example q4), close other tabs, or reload the page to free memory.'
    : stage === 'warm up'
      ? 'The model loaded but could not generate one token, so it may not be supported by this runtime or backend.'
      : `This model may not be supported by the current Transformers.js browser runtime or backend '${backend}'. ` +
        'Check the WebGPU guide and model/dtype availability in the Transformers.js docs before picking a model.';
  return new Error(`Failed to ${stage} ${what} model '${model}': ${message}. ${advice} (backend '${backend}'${dtypes}; ${runtime})`);
}

// Transformers.js tasks exposed through the generic pipeline interface.
// Generation, embeddings and text classification keep their dedicated methods.
// Tasks that return images (the others return labels, boxes and vectors); their inputs are images too.
const IMAGE_OUTPUT_TASKS = ['background-removal', 'depth-estimation', 'image-segmentation', 'image-to-image'];
const IMAGE_TASKS = [
  'image-classification', 'zero-shot-image-classification', 'image-feature-extraction', 'object-detection',
  ...IMAGE_OUTPUT_TASKS,
];
const SPEECH_TASKS = ['text-to-speech', 'text-to-audio'];  // Transformers.js treats them as one task
// Tasks whose input is audio: a WAV URL, or mono samples at 16 kHz.
const AUDIO_INPUT_TASKS = ['automatic-speech-recognition', 'audio-classification'];
const SUPPORTED_TASKS = ['zero-shot-classification', ...AUDIO_INPUT_TASKS, 'translation', ...IMAGE_TASKS, ...SPEECH_TASKS];

function unsupportedTaskError(task) {
  return new Error(
    `Unsupported task '${task}'. Supported pipeline tasks are: ${SUPPORTED_TASKS.join(', ')}. ` +
    `Use generate()/chat(), embed() or classify() for text generation, embeddings and text classification.`
  );
}

// Sampling rate expected by Whisper and most Transformers.js speech recognition models.
const AUDIO_SAMPLING_RATE = 16000;

// Dtype order tried when no dtype is configured (generation, classification and task pipelines).
const DEFAULT_DTYPES = ['q8', 'int8', 'fp32'];

// Decode PCM or float WAV into mono Float32Array samples at samplingRate (mirrors _read_wav in llm.py).
function decodeWav(buffer, samplingRate = AUDIO_SAMPLING_RATE) {
  const view = new DataView(buffer);
  const tag = (offset) => String.fromCharCode(...new Uint8Array(buffer, offset, 4));
  if (buffer.byteLength < 12 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE') {
    throw new Error("Audio must be a WAV file when AudioContext is unavailable (e.g. in a notebook kernel).");
  }
  let format, channels, rate, width, data;
  for (let offset = 12; offset + 8 <= buffer.byteLength;) {
    const size = view.getUint32(offset + 4, true);
    if (tag(offset) === 'fmt ') {
      format = view.getUint16(offset + 8, true);
      channels = view.getUint16(offset + 10, true);
      rate = view.getUint32(offset + 12, true);
      width = view.getUint16(offset + 22, true) / 8;
      if (format === 0xFFFE) {
        format = view.getUint16(offset + 32, true);  // WAVE_FORMAT_EXTENSIBLE sub-format
      }
    } else if (tag(offset) === 'data') {
      data = { start: offset + 8, size: Math.min(size, buffer.byteLength - offset - 8) };
    }
    offset += 8 + size + (size % 2);
  }
  if (!format || !data) {
    throw new Error("Invalid WAV file: missing fmt or data chunk.");
  }
  const read = {
    '1:1': (o) => (view.getUint8(o) - 128) / 128,
    '1:2': (o) => view.getInt16(o, true) / 32768,
    '1:3': (o) => ((view.getUint8(o + 2) << 24 | view.getUint8(o + 1) << 16 | view.getUint8(o) << 8) >> 8) / 8388608,
    '1:4': (o) => view.getInt32(o, true) / 2147483648,
    '3:4': (o) => view.getFloat32(o, true),
    '3:8': (o) => view.getFloat64(o, true),
  }[`${format}:${width}`];
  if (!read) {
    throw new Error(`Unsupported WAV encoding (format ${format}, ${width * 8}-bit).`);
  }
  const frames = Math.floor(data.size / (width * channels));
  let samples = new Float32Array(frames);
  for (let i = 0; i < frames; i++) {
    let sum = 0;
    for (let c = 0; c < channels; c++) {
      sum += read(data.start + (i * channels + c) * width);
    }
    samples[i] = sum / channels;
  }
  if (rate !== samplingRate && frames > 1) {
    // Linear interpolation is sufficient for speech; resample offline for best quality.
    const count = Math.max(1, Math.round(frames * samplingRate / rate));
    const step = (frames - 1) / Math.max(1, count - 1);
    const resampled = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const position = i * step;
      const left = Math.floor(position);
      const right = Math.min(left + 1, frames - 1);
      resampled[i] = samples[left] + (samples[right] - samples[left]) * (position - left);
    }
    samples = resampled;
  }
  return samples;
}

async function fetchWavSamples(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch audio from ${url}: ${response.status} ${response.statusText}`);
  }
  return decodeWav(await response.arrayBuffer());
}

function resolveModelId(modelName) {
  if (!modelName || typeof modelName !== 'string') {
    return modelName;
  }
  return modelName.includes('/') ? modelName : `${DEFAULT_MODEL_NAMESPACE}/${modelName}`;
}

function modelIdCandidates(modelName) {
  return [...new Set([resolveModelId(modelName), modelName].filter(Boolean))];
}

// The JupyterLite site root URL (ending in '/'), or null when the page or worker URL does not identify it.
function siteRoot() {
  const location = globalThis.location;
  if (!location || !['http:', 'https:'].includes(location.protocol)) {
    return null;
  }
  const marker = location.pathname.match(/\/(?:extensions|lab|retro|tree)\//);
  return marker ? `${location.origin}${location.pathname.slice(0, marker.index)}/` : null;
}

let transformersModule = null;

// Hosted model id -> base URL of its hosted files.
const hostedModelBases = new Map();

// Transformers.js 4.3 checks whether tokenizer_config.json exists on the Hub even for local_files_only models
// (get_tokenizer_files ignores the option, and skips the local check for an HTTP localModelPath). Serve Hub
// requests for hosted models from the hosted files, so a site works where huggingface.co is blocked.
function routeHostedModels(env) {
  if (env.fetch.lmliteHosted) {
    return;
  }
  const fetchFn = env.fetch;
  const routed = (input, init) => {
    const match = typeof input === 'string' && input.match(/^https:\/\/huggingface\.co\/([^/]+\/[^/]+)\/resolve\/main\/(.+)$/);
    const base = match && hostedModelBases.get(match[1]);
    return fetchFn(base ? `${base}${match[1]}/${match[2]}` : input, init);
  };
  routed.lmliteHosted = true;
  env.fetch = routed;
}

// Encoded bytes (ArrayBuffer/Uint8Array), a URL string, or ["raw", pixels, width, height, channels] -> RawImage.
async function toRawImage(value) {
  const { RawImage } = await loadTransformers();
  if (typeof value === 'string') {
    return await RawImage.read(value);
  }
  if (Array.isArray(value) && value[0] === 'raw') {
    const [, pixels, width, height, channels] = value;
    return new RawImage(new Uint8ClampedArray(pixels), width, height, channels);
  }
  try {
    return await RawImage.fromBlob(new Blob([value]));
  } catch (error) {
    throw new Error(`The image could not be decoded: ${String(error?.message ?? error)}`, { cause: error });
  }
}

// The result of an image-output task as [JSON structure, ...one Uint8Array per image or tensor]: every RawImage in
// `value` becomes {$buffer: i, width, height, channels} and every Tensor {$buffer: i, dtype, shape}, so each image
// crosses the bridge as one buffer. Images are PNG files, or raw pixels with `raw`. A worker has no <canvas>: the
// PNG comes from RawImage.toBlob(), which uses OffscreenCanvas there.
async function packImages(value, raw) {
  const buffers = [];
  const isImage = (item) => item && typeof item === 'object' && item.width && item.height && item.channels && item.data;
  const isTensor = (item) => item && typeof item === 'object' && item.dims && item.ort_tensor;
  const walk = async (item) => {
    if (isImage(item)) {
      const bytes = raw
        ? new Uint8Array(item.data.buffer.slice(item.data.byteOffset, item.data.byteOffset + item.data.byteLength))
        : new Uint8Array(await (await item.toBlob('image/png')).arrayBuffer());
      buffers.push(bytes);
      return { $buffer: buffers.length - 1, width: item.width, height: item.height, channels: item.channels };
    }
    if (isTensor(item)) {
      const data = Float32Array.from(item.data);
      buffers.push(new Uint8Array(data.buffer));
      return { $buffer: buffers.length - 1, dtype: 'float32', shape: [...item.dims] };
    }
    if (Array.isArray(item)) {
      return await Promise.all(item.map(walk));
    }
    if (item && typeof item === 'object') {
      return Object.fromEntries(await Promise.all(Object.entries(item).map(async ([key, entry]) => [key, await walk(entry)])));
    }
    return item;
  };
  const structure = await walk(value);
  return [JSON.stringify(structure), ...buffers];
}

// Transformers.js from the site's vendor/transformers/ when the build put it there, else from jsDelivr.
function loadTransformers() {
  transformersModule ??= (async () => {
    const root = siteRoot();
    if (root) {
      const base = `${root}vendor/transformers/`;
      let transformers = null;
      try {
        transformers = await import(`${base}transformers.min.js`);
      } catch {
        // Not self-hosted: fall back to the CDN below.
      }
      if (transformers) {
        // Transformers.js points onnxruntime-web at jsDelivr, picking the file variant for the browser; keep the
        // variant, serve it from the site.
        const wasm = transformers.env.backends.onnx.wasm;
        if (wasm.wasmPaths && typeof wasm.wasmPaths === 'object') {
          wasm.wasmPaths = Object.fromEntries(
            Object.entries(wasm.wasmPaths).map(([kind, url]) => [kind, base + url.split('/').pop()])
          );
        }
        return transformers;
      }
    }
    return import(TRANSFORMERS_URL);
  })().catch((error) => {
    transformersModule = null;  // let the next call retry, e.g. after a network error
    throw error;
  });
  return transformersModule;
}

const MANIFEST_ATTEMPTS = 3;
const MANIFEST_RETRY_MS = 250;

async function getHostedModelBase(modelName, configuredBase = null) {
  const [modelId, revision] = resolveModelId(modelName).split('@', 2);
  // Hosted trees represent one snapshot, not arbitrary Hub revisions.
  if (revision && revision !== 'main') {
    return null;
  }
  let base = configuredBase;
  if (!base) {
    const root = siteRoot();
    if (!root) {
      return null;
    }
    base = `${root}files/models/`;
  }
  const url = new URL(base.endsWith('/') ? base : `${base}/`, globalThis.location?.href);
  if (!['http:', 'https:'].includes(url.protocol)) {
    throw new Error('hosted_models_url must be an HTTP(S) URL');
  }
  const manifestUrl = new URL('manifest.json', url);
  let response;
  let failure = null;  // what the last attempt hit, if it did not give a usable answer
  // GitHub Pages and other hosts answer 5xx now and then, and a worker's fetch can fail: try a few times.
  for (let attempt = 1; attempt <= MANIFEST_ATTEMPTS; attempt++) {
    try {
      response = await fetch(manifestUrl, { signal: AbortSignal.timeout(15000) });
      failure = response.status >= 500 ? new Error(`Hosted model manifest request failed: ${response.status} ${response.url}`) : null;
    } catch (error) {
      response = null;
      failure = error;
    }
    if (!failure) {
      break;
    }
    if (attempt < MANIFEST_ATTEMPTS) {
      await new Promise((resolve) => setTimeout(resolve, MANIFEST_RETRY_MS * attempt));
    }
  }
  if (!configuredBase) {
    // The site's own manifest is optional: its absence, or a host that will not serve it, means "nothing hosted",
    // and the model loads from the Hub. Only an explicit hosted_models_url is held to a working manifest.
    if (failure) {
      console.warn(`Could not read the site's hosted model manifest (${failure.message ?? failure}); loading from the Hub.`);
      return null;
    }
    if (!response.ok) {
      return null;  // 404 when the site hosts no models; anything else is not a manifest either
    }
  }
  if (failure) {
    throw failure;
  }
  if (!response.ok) {
    throw new Error(`Hosted model manifest request failed: ${response.status} ${response.url}`);
  }
  const manifest = await response.json();
  if (!Array.isArray(manifest.models) || manifest.models.some(id =>
    typeof id !== 'string' || !/^[\w.-]+\/[\w.-]+$/.test(id) ||
    id.split('/').some(part => part === '.' || part === '..')
  )) {
    throw new Error('Invalid hosted model manifest: expected a models array of namespace/model IDs');
  }
  return manifest.models.includes(modelId) ? url.href : null;
}

function modelFilePathFromUrl(requestUrl, modelId) {
  const pathname = new URL(requestUrl).pathname;
  const modelPrefix = `/${modelId}/resolve/`;
  const modelPathStart = pathname.indexOf(modelPrefix);
  if (modelPathStart >= 0) {
    const revisionAndPath = pathname.slice(modelPathStart + modelPrefix.length);
    const pathStart = revisionAndPath.indexOf('/');
    if (pathStart >= 0) {
      return revisionAndPath.slice(pathStart + 1);
    }
  }
  const hostedPrefix = `/${modelId}/`;
  const hostedPathStart = pathname.indexOf(hostedPrefix);
  if (hostedPathStart >= 0) {
    return pathname.slice(hostedPathStart + hostedPrefix.length);
  }
  return pathname.split('/').pop();
}

// Softmax over one step's final logits: the probability of any token and the n most probable [id, prob].
function tokenDistribution(logits, n) {
  let max = -Infinity;
  for (const x of logits) {
    if (x > max) {
      max = x;
    }
  }
  let sum = 0;
  const top = [];  // [id, logit], descending
  for (let id = 0; id < logits.length; id++) {
    const x = logits[id];
    sum += Math.exp(x - max);
    if (top.length < n || x > top.at(-1)[1]) {
      let i = top.length;
      while (i > 0 && top[i - 1][1] < x) {
        i--;
      }
      top.splice(i, 0, [id, x]);
      if (top.length > n) {
        top.pop();
      }
    }
  }
  const prob = (x) => Math.exp(x - max) / sum;
  return { prob: (id) => prob(logits[id]), top: top.map(([id, x]) => [id, prob(x)]) };
}

// generate() options that report each chosen token with its probability and the most probable
// other tokens. Transformers.js calls a custom logits processor last, right before it picks a
// token, then passes the pick to the streamer; probabilities are what the picker saw.
function tokenRecorder(tokenizer, alternatives, onRecord) {
  let step = null;
  let prompt = true;
  const ids = [];
  let decoded = '';
  return {
    logits_processor: [(inputIds, logits) => {
      step = tokenDistribution(logits.data, alternatives + 1);  // batch of one
      return logits;
    }],
    streamer: {
      put(value) {
        if (prompt) {
          prompt = false;
          return;
        }
        const id = Number(value[0][0]);
        ids.push(id);
        // Decode the whole answer so multi-byte characters split across tokens come out whole.
        const full = tokenizer.decode(ids, { skip_special_tokens: true });
        const complete = !full.endsWith('\uFFFD');
        const text = complete ? full.slice(decoded.length) : '';
        if (complete) {
          decoded = full;
        }
        if (!text && !tokenizer.decode([id], { skip_special_tokens: true })) {
          return;  // special tokens such as end-of-turn
        }
        onRecord({
          text,
          prob: step.prob(id),
          alternatives: step.top.filter(([other]) => other !== id).slice(0, alternatives)
            .map(([other, prob]) => ({ text: tokenizer.decode([other]), prob })),
        });
      },
      end() {},
    },
  };
}

// On WASM a generation step only awaits microtasks, which starves macrotasks: kernels resume
// Python (streamed chunks, cancel() from a widget) through setTimeout. Yield one per step.
function yieldBeforeEachStep(model) {
  if (typeof model?.forward !== 'function') {
    return;
  }
  const forward = model.forward.bind(model);
  model.forward = async (...args) => {
    await new Promise((resolve) => setTimeout(resolve, 0));
    return forward(...args);
  };
}

class BrowserRuntime {
  constructor(config) {
    this.generatorModel = config.generator_model;
    this.embeddingModel = config.embedding_model;
    this.classificationModel = config.classification_model ?? null;
    this.task = config.task ?? null;
    this.taskModel = config.task_model ?? null;
    this.backend = (config.backend ?? 'wasm').toLowerCase();
    this.embeddingBackend = config.embedding_backend == null ? null : config.embedding_backend.toLowerCase();
    this.dtype = config.dtype ?? null;
    this.dtypeFallbacks = Array.isArray(config.dtype_fallbacks) ? config.dtype_fallbacks : null;
    // Optional explicit dtype for the embedding (feature-extraction) pipeline.
    // Pin it to make embeddings reproducible: a consumer that precomputes anchor
    // vectors offline must load the SAME quantization the browser runs, or the
    // cosines (and any margin/threshold tuned on them) drift. When null, the
    // feature-extraction pipeline keeps Transformers.js's default dtype.
    this.embeddingDtype = config.embedding_dtype ?? null;
    this.max_new_tokens = config.max_new_tokens ?? 50;
    this.return_full_text = config.return_full_text ?? true;
    this.temperature = config.temperature ?? 0.7;
    this.top_k = config.top_k ?? 50;
    this.do_sample = config.do_sample ?? true;
    this.hostedModelBase = config.hosted_model_base ?? null;
    this.hostedModels = config.hosted_models ?? [];

    this.generator = null;
    this.embedder = null;
    this.classifier = null;
    this.taskPipeline = null;
  }

  _resolveDtypes(defaultDtypes) {
    if (this.dtypeFallbacks && this.dtypeFallbacks.length > 0) {
      return this.dtypeFallbacks;
    }
    if (this.dtype) {
      return [this.dtype];
    }
    return defaultDtypes;
  }

  _isRecoverableDtypeError(message) {
    return (
      message.includes('Could not locate file') ||
      message.includes('404') ||
      message.includes('not found') ||
      message.includes("Can't create a session") ||
      message.includes('ERROR_CODE')
    );
  }

  async _loadPipelineWithDtypeFallback(pipelineFn, task, modelId, extraOptions, dtypes = DEFAULT_DTYPES) {
    let lastError;
    const tried = [];
    for (const dtype of this._resolveDtypes(dtypes)) {
      tried.push(dtype);
      try {
        return await pipelineFn(task, modelId, { ...extraOptions, dtype });
      } catch (error) {
        const msg = String(error?.message ?? error);
        if (this._isRecoverableDtypeError(msg)) {
          console.warn(`dtype '${dtype}' not available for ${modelId}, trying next…`);
          lastError = error;
        } else {
          throw Object.assign(error, { dtypesTried: tried });
        }
      }
    }
    throw Object.assign(lastError, { dtypesTried: tried });
  }

  async init(progressCallback = null) {
    const report = (model, status, file = '', loaded = 0, total = 0) => {
      if (progressCallback) {
        progressCallback(model, status, file, loaded, total);
      }
    };
    const onProgress = (model) => (progress) => {
      if (progress.status === 'progress' || progress.status === 'downloading') {
        report(model, 'progress', progress.file ?? '', progress.loaded ?? 0, progress.total ?? 0);
      } else if (progress.status === 'ready') {
        report(model, 'initializing');
      }
    };
    report('', 'initializing');
    const { pipeline, env } = await loadTransformers();

    if (!['wasm', 'webgpu'].includes(this.backend)) {
      throw new Error(
        `Invalid backend '${this.backend}'. Supported backends are 'wasm' and 'webgpu'.`
      );
    }

    if (this.embeddingBackend !== null && !['wasm', 'webgpu'].includes(this.embeddingBackend)) {
      throw new Error(
        `Invalid embedding_backend '${this.embeddingBackend}'. Supported backends are 'wasm' and 'webgpu'.`
      );
    }

    if (this.task !== null && !SUPPORTED_TASKS.includes(this.task)) {
      throw unsupportedTaskError(this.task);
    }
    if (this.task !== null && !this.taskModel) {
      throw new Error(`Task '${this.task}' requires a model.`);
    }

    env.allowLocalModels = Boolean(this.hostedModelBase);
    if (this.hostedModelBase) {
      env.localModelPath = this.hostedModelBase;
      for (const model of this.hostedModels) {
        hostedModelBases.set(model, this.hostedModelBase);
      }
      routeHostedModels(env);
    }
    env.useBrowserCache = true;
    console.log('Using browser cache (Hugging Face Cache API - Transformers.js v4)');
    const sourceOptions = (model) => this.hostedModels.includes(model)
      ? { local_files_only: true }
      : {};

    if (this.generatorModel) {
      let stage = 'load';
      try {
        const generatorModelId = resolveModelId(this.generatorModel).replace(/@main$/, '');
        report(generatorModelId, 'loading');
        console.log(`Loading text-generation model: ${generatorModelId}`);
        this.generator = await this._loadPipelineWithDtypeFallback(
          pipeline,
          "text-generation",
          generatorModelId,
          {
            device: this.backend,
            ...sourceOptions(generatorModelId),
            progress_callback: onProgress(generatorModelId)
          }
        );
        yieldBeforeEachStep(this.generator.model);
        console.log(`Model loaded successfully: ${this.generatorModel}`);
        console.log(`Warming up model...`);
        report(generatorModelId, 'warming up');
        stage = 'warm up';
        await this.generator("test", { max_new_tokens: 1 });
        console.log(`Model warm-up complete`);
      } catch (error) {
        throw loadFailure({ stage, what: 'text-generation', model: this.generatorModel, backend: this.backend, error, env });
      }
    }

    if (this.embeddingModel) {
      try {
        const embeddingModelId = resolveModelId(this.embeddingModel).replace(/@main$/, '');
        report(embeddingModelId, 'loading');
        console.log(`Loading embedding model: ${embeddingModelId}`);
        // Embedding device selection is independent of the generation/classification backend.
        const embeddingOptions = {
          ...(this.embeddingBackend === null ? {} : { device: this.embeddingBackend }),
          ...sourceOptions(embeddingModelId),
          progress_callback: onProgress(embeddingModelId),
        };
        if (this.embeddingDtype) {
          // Pin EXACTLY this dtype. This is deliberately independent of the
          // global `dtype` / `dtype_fallbacks` (which target the generator): we
          // do NOT route through _loadPipelineWithDtypeFallback / _resolveDtypes,
          // because those would let a generator-oriented dtype override the
          // embedding pin. A pinned embedding dtype must be reproducible (an
          // offline-precomputed vector set depends on it), so there is no
          // fallback to another dtype here.
          console.log(`Pinning embedding dtype: ${this.embeddingDtype}`);
          this.embedder = await pipeline("feature-extraction", embeddingModelId, {
            ...embeddingOptions,
            dtype: this.embeddingDtype,
          });
        } else {
          // Default: let Transformers.js choose the feature-extraction dtype.
          this.embedder = await pipeline("feature-extraction", embeddingModelId, embeddingOptions);
        }
        console.log(`Embedding model loaded successfully: ${this.embeddingModel}`);
      } catch (error) {
        throw new Error(
          `Failed to load embedding model '${this.embeddingModel}': ${error.message}`
        );
      }
    }

    if (this.classificationModel) {
      try {
        const classificationModelId = resolveModelId(this.classificationModel).replace(/@main$/, '');
        report(classificationModelId, 'loading');
        console.log(`Loading classification model: ${classificationModelId}`);
        this.classifier = await this._loadPipelineWithDtypeFallback(
          pipeline,
          "text-classification",
          classificationModelId,
          {
            device: this.backend,
            ...sourceOptions(classificationModelId),
            progress_callback: onProgress(classificationModelId)
          }
        );
        console.log(`Classification model loaded successfully: ${this.classificationModel}`);
      } catch (error) {
        throw new Error(
          `Failed to load classification model '${this.classificationModel}': ${error.message}`
        );
      }
    }

    if (this.task) {
      try {
        const taskModelId = resolveModelId(this.taskModel).replace(/@main$/, '');
        report(taskModelId, 'loading');
        console.log(`Loading ${this.task} model: ${taskModelId}`);
        this.taskPipeline = await this._loadPipelineWithDtypeFallback(
          pipeline,
          this.task,
          taskModelId,
          {
            device: this.backend,
            ...sourceOptions(taskModelId),
            progress_callback: onProgress(taskModelId)
          }
        );
        console.log(`${this.task} model loaded successfully: ${this.taskModel}`);
      } catch (error) {
        throw loadFailure({ stage: 'load', what: this.task, model: this.taskModel, backend: this.backend, error, env });
      }
    }
    report('', 'ready');
  }

  async exportModelFiles(modelName) {
    try {
      const modelId = resolveModelId(modelName);
      const modelCandidates = modelIdCandidates(modelName);
      console.log(`Starting export for model: ${modelId}`);

      // Check both Cache API and IndexedDB
      const cacheNames = await caches.keys();
      console.log('Available Cache API caches:', cacheNames);

      // List all IndexedDB databases to find the right one
      const databases = await indexedDB.databases();
      console.log('Available IndexedDB databases:', databases.map(db => db.name));

      // First try Cache API (Transformers.js v4 uses this)
      const modelCacheName = cacheNames.find(name =>
        name.includes('transformers') ||
        name.includes('huggingface') ||
        name.includes('onnx')
      );

      if (modelCacheName) {
        console.log(`Found model cache: ${modelCacheName}`);
        return await this._exportFromCacheAPI(modelName, modelId, modelCacheName);
      }

      // Try different possible database names for IndexedDB (older versions)
      const possibleDbNames = [
        'transformers-cache',
        'transformers_cache',
        'huggingface-assets',
        'transformers.js',
        ...databases.map(db => db.name)
      ];

      let db = null;
      let dbName = null;

      for (const name of possibleDbNames) {
        try {
          db = await new Promise((resolve, reject) => {
            const request = indexedDB.open(name);
            request.onsuccess = () => {
              if (request.result.objectStoreNames.length > 0) {
                resolve(request.result);
              } else {
                request.result.close();
                resolve(null);
              }
            };
            request.onerror = () => resolve(null);
          });
          if (db) {
            dbName = name;
            console.log(`Found database: ${dbName}`);
            break;
          }
        } catch (e) {
          console.log(`Database ${name} not found or empty`);
        }
      }

      if (!db) {
        console.error('No IndexedDB cache found. Available databases:', databases);
        return `Error: No model cache database found. Models may still be downloading or using a different cache mechanism.`;
      }

      console.log(`Object stores in ${dbName}:`, Array.from(db.objectStoreNames));

      const storeName = db.objectStoreNames[0];
      const transaction = db.transaction([storeName], 'readonly');
      const store = transaction.objectStore(storeName);

      const allKeys = await new Promise((resolve, reject) => {
        const request = store.getAllKeys();
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      });

      console.log(`Total cached entries: ${allKeys.length}`);
      console.log(`Sample keys:`, allKeys.slice(0, 5));

      // Filter keys for this model
      const modelKeys = allKeys.filter(key =>
        modelCandidates.some(candidate => String(key).includes(candidate))
      );

      console.log(`Found ${modelKeys.length} cached files for ${modelId}`);

      if (modelKeys.length === 0) {
        return `Error: No cached files found for model '${modelId}'. Model may not be cached yet.`;
      }

      // Create directory for model files
      const pyodide = globalThis.pyodide;
      if (!pyodide) {
        return 'Error: Pyodide not available, cannot export to filesystem';
      }

      const modelDir = `/drive/models/${modelId}`;
      pyodide.FS.mkdirTree(modelDir);
      console.log(`Created directory: ${modelDir}`);

      let exportedCount = 0;

      // Export each file
      for (const key of modelKeys) {
        const data = await new Promise((resolve, reject) => {
          const request = store.get(key);
          request.onsuccess = () => resolve(request.result);
          request.onerror = () => reject(request.error);
        });

        console.log(`Processing key: ${key}, data type:`, typeof data, data);

        if (data) {
          let arrayBuffer;

          // Handle different data formats
          if (data instanceof Blob) {
            arrayBuffer = await data.arrayBuffer();
          } else if (data.blob instanceof Blob) {
            arrayBuffer = await data.blob.arrayBuffer();
          } else if (data instanceof ArrayBuffer) {
            arrayBuffer = data;
          } else if (data.data instanceof ArrayBuffer) {
            arrayBuffer = data.data;
          } else {
            console.warn(`Unknown data format for key ${key}:`, data);
            continue;
          }

          const uint8Array = new Uint8Array(arrayBuffer);

          const filename = modelFilePathFromUrl(String(key), modelId);
          const filepath = `${modelDir}/${filename}`;
          pyodide.FS.mkdirTree(filepath.slice(0, filepath.lastIndexOf('/')));

          pyodide.FS.writeFile(filepath, uint8Array);
          console.log(`Exported: ${filepath} (${uint8Array.length} bytes)`);
          exportedCount++;
        }
      }

      db.close();

      const message = `Successfully exported ${exportedCount} files to ${modelDir}`;
      console.log(message);
      return message;

    } catch (error) {
      const errorMsg = `Export failed: ${error.message}`;
      console.error(errorMsg, error);
      return errorMsg;
    }
  }

  async _exportFromCacheAPI(modelName, modelId, cacheName) {
    try {
      const cache = await caches.open(cacheName);
      const requests = await cache.keys();
      const modelCandidates = modelIdCandidates(modelName);

      console.log(`Total cached requests: ${requests.length}`);

      // Filter for this model
      const modelRequests = requests.filter(req =>
        modelCandidates.some(candidate => req.url.includes(candidate))
      );

      console.log(`Found ${modelRequests.length} cached files for ${modelId}`);

      if (modelRequests.length === 0) {
        return `Error: No cached files found for model '${modelId}' in Cache API.`;
      }

      // Access Pyodide filesystem (in web worker context)
      // Try different ways to access pyodide
      let pyodide = null;
      try {
        pyodide = self.pyodide;
      } catch (e) {}

      if (!pyodide) {
        try {
          pyodide = globalThis.pyodide;
        } catch (e) {}
      }

      if (!pyodide) {
        console.error('Pyodide not found in self or globalThis');
        // Return data to Python side instead
        return await this._exportViaPython(modelId, cacheName, modelRequests);
      }

      console.log('Pyodide found, exporting via FS:', pyodide);

      const modelDir = `/drive/models/${modelId}`;

      // Create directory using Pyodide FS
      try {
        pyodide.FS.mkdirTree(modelDir);
        console.log(`Created directory: ${modelDir}`);
      } catch (e) {
        console.error('Failed to create directory:', e);
        return `Error: Failed to create directory ${modelDir}: ${e.message}`;
      }

      let exportedCount = 0;

      for (const request of modelRequests) {
        const response = await cache.match(request);
        if (response) {
          const arrayBuffer = await response.arrayBuffer();
          const uint8Array = new Uint8Array(arrayBuffer);

          const filename = modelFilePathFromUrl(request.url, modelId);

          const filepath = `${modelDir}/${filename}`;
          pyodide.FS.mkdirTree(filepath.slice(0, filepath.lastIndexOf('/')));
          pyodide.FS.writeFile(filepath, uint8Array);
          console.log(`Exported: ${filepath} (${uint8Array.length} bytes)`);
          exportedCount++;
        }
      }

      const message = `Successfully exported ${exportedCount} files to ${modelDir}`;
      console.log(message);
      return message;

    } catch (error) {
      const errorMsg = `Cache API export failed: ${error.message}`;
      console.error(errorMsg, error);
      return errorMsg;
    }
  }

  async _exportViaPython(modelId, cacheName, modelRequests) {
    // Export by returning data to Python side
    console.log('Exporting via Python side');

    const cache = await caches.open(cacheName);
    const files = [];

    for (const request of modelRequests) {
      const response = await cache.match(request);
      if (response) {
        const arrayBuffer = await response.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);

        const filename = modelFilePathFromUrl(request.url, modelId);

        files.push({
          filename: filename,
          data: uint8Array
        });
      }
    }

    return {
      success: true,
      model: modelId,
      files: files
    };
  }

  async dispose() {
    const errors = [];
    for (const name of ['generator', 'embedder', 'classifier', 'taskPipeline']) {
      const pipeline = this[name];
      if (pipeline === null) {
        continue;
      }
      try {
        await pipeline.dispose();
        this[name] = null;
      } catch (error) {
        errors.push(new Error(`Failed to dispose ${name}: ${String(error?.message ?? error)}`, { cause: error }));
      }
    }
    if (errors.length) {
      throw new AggregateError(errors, errors.map(error => error.message).join('; '));
    }
  }

  async generate(prompt, returnFullText = this.return_full_text, extra = {}) {
    if (!this.generator) {
      throw new Error("Generator model not initialized. Call init() first.");
    }
    if (this._generation?.cancelled) {
      // An interrupted generation stops after its current token; let it finish.
      await this._generation.done;
    }
    if (this._generation) {
      throw new Error("Another generation is running on this instance; await it or call cancel() first.");
    }
    // Claim the instance before awaiting anything, so cancel() during the import below is not lost.
    const generation = this._generation = { stopping_criteria: null, cancelled: false };
    const run = (async () => {
      try {
        const { InterruptableStoppingCriteria } = await loadTransformers();
        generation.stopping_criteria = new InterruptableStoppingCriteria();
        if (generation.cancelled) {
          generation.stopping_criteria.interrupt();
        }
        return await this.generator(prompt, {
          max_new_tokens: this.max_new_tokens,
          return_full_text: returnFullText,
          temperature: this.temperature,
          top_k: this.top_k,
          do_sample: this.do_sample,
          ...extra,
          stopping_criteria: generation.stopping_criteria,
        });
      } finally {
        this._generation = null;
      }
    })();
    generation.done = run.catch(() => {});

    try {
      const output = await run;
      return output[0].generated_text;
    } catch (error) {
      throw new Error(
        `Text generation failed: ${error.message}. ` +
        `This could be due to: ` +
        `(1) Model still downloading - wait and retry, ` +
        `(2) Memory constraints in browser, ` +
        `(3) ONNX Runtime error - try a different model like 'gpt2', ` +
        `(4) Model incompatibility with browser WASM environment.`
      );
    }
  }

  // Stops the running generate()/chat()/stream after the current token (they return the partial text), or the
  // running runTasks() after the current batch (it returns the results so far).
  async cancel() {
    const generation = this._generation;
    const batches = this._batchRun;
    if (!generation && !batches) {
      return false;
    }
    if (batches) {
      batches.cancelled = true;
    }
    if (generation) {
      generation.cancelled = true;
      generation.stopping_criteria?.interrupt();  // null until the runtime has loaded
    }
    await Promise.all([generation?.done, batches?.done]);
    return true;
  }

  async chat(messages) {
    const input = this._chatInput(messages);
    const output = await this.generate(input, false);
    if (typeof input === 'string') {
      return output.trim();
    }
    const answer = Array.isArray(output) ? output.at(-1) : null;
    if (answer?.role !== 'assistant' || typeof answer.content !== 'string') {
      throw new Error("Chat pipeline returned an invalid assistant message.");
    }
    return answer.content.trim();
  }

  async chatStream(messages) {
    return this._stream(this._chatInput(messages));
  }

  async generateStream(prompt) {
    return this._stream(prompt);
  }

  async chatTokens(messages, alternatives = 1) {
    return this._stream(this._chatInput(messages), alternatives);
  }

  async generateTokens(prompt, alternatives = 1) {
    return this._stream(prompt, alternatives);
  }

  // Pull-based so both Python bridges can await chunks without JS→Python callbacks.
  // With `alternatives`, each chunk is a JSON token record instead of text.
  async _stream(input, alternatives = null) {
    if (!this.generator) {
      throw new Error("Generator model not initialized. Call init() first.");
    }
    const { TextStreamer } = await loadTransformers();
    const items = [];
    let wake = () => {};
    const push = (item) => {
      items.push(item);
      wake();
    };
    const extra = alternatives === null
      ? {
        streamer: new TextStreamer(this.generator.tokenizer, {
          skip_prompt: true,
          skip_special_tokens: true,
          callback_function: push,
        }),
      }
      : tokenRecorder(this.generator.tokenizer, alternatives, (record) => push(JSON.stringify(record)));
    this.generate(input, false, extra).then(() => push(''), push);
    return {
      // Resolves to the next nonempty chunk, '' once generation ends, or rejects with its error.
      read: async () => {
        while (!items.length) {
          await new Promise((resolve) => { wake = resolve; });
        }
        const item = items[0];
        if (typeof item !== 'string') {
          throw item;
        }
        if (item) {
          items.shift();
        }
        return item;
      },
    };
  }

  _chatInput(messages) {
    if (!this.generator) {
      throw new Error("Generator model not initialized. Call init() first.");
    }
    if (typeof messages === 'string') {
      messages = JSON.parse(messages);
    }
    if (!Array.isArray(messages) || messages.length === 0) {
      throw new TypeError("Chat requires a nonempty array of messages.");
    }
    const normalized = messages.map((message) => {
      if (message == null || typeof message !== 'object' ||
          typeof message.role !== 'string' || !message.role.trim() ||
          typeof message.content !== 'string') {
        throw new TypeError("Each chat message requires a nonempty string role and string content.");
      }
      return { role: message.role.toLowerCase(), content: message.content };
    });

    if (this.generator.tokenizer?.chat_template) {
      // The pipeline applies the tokenizer's template and avoids duplicate special tokens.
      return normalized;
    }

    return normalized.map(({ role, content }) => {
      const label = role === 'user' ? 'User' : role === 'assistant' ? 'Assistant' : role;
      return `${label}: ${content}\n`;
    }).join('') + 'Assistant:';
  }

  async embed(text) {
    if (!this.embedder) {
      throw new Error("No embedding model configured");
    }

    // Return a single sentence embedding, not the raw per-token tensor.
    // Without `pooling`, feature-extraction returns the unpooled
    // [1, tokens, hidden] last_hidden_state; flattening that gives a vector
    // whose length scales with token count and which is not a sentence
    // representation, so nearest-neighbour ranking over it is unreliable. Mean
    // pooling (attention-mask weighted) collapses it to one fixed-dim sentence
    // vector, as sentence-embedding models such as multilingual-e5 expect.
    // `normalize: true` makes the returned `.data` unit-length so a consumer
    // doing a bare dot product gets cosine similarity; it only rescales and
    // does not change the ranking of a consumer that normalizes itself.
    const result = await this.embedder(text, { pooling: "mean", normalize: true });
    return Array.from(result.data);
  }

  async classify(text, top_k = null) {
    if (!this.classifier) {
      throw new Error("No classification model configured");
    }

    if (top_k === null || top_k === undefined) {
      return await this.classifier(text);
    }
    return await this.classifier(text, { top_k });
  }

  async runTask(inputs, options = {}) {
    if (!this.taskPipeline) {
      throw new Error(this.task ? `${this.task} pipeline not initialized` : "No pipeline task configured");
    }
    // Options arrive as JSON so plain objects cross both Python browser bridges.
    if (typeof options === 'string') {
      options = JSON.parse(options);
    }
    options = options ?? {};
    if (AUDIO_INPUT_TASKS.includes(this.task) && typeof inputs === 'string' && typeof AudioContext === 'undefined') {
      // Kernels run in Web Workers, where Transformers.js cannot decode URLs without AudioContext.
      inputs = await fetchWavSamples(inputs);
    } else if (AUDIO_INPUT_TASKS.includes(this.task) && typeof inputs !== 'string' && !(inputs instanceof Float32Array)) {
      // Raw samples (e.g. a Python list or Float64Array) become the Float32Array audio input.
      inputs = Float32Array.from(inputs);
    }
    if (IMAGE_TASKS.includes(this.task)) {
      const isBatch = Array.isArray(inputs) && inputs[0] !== 'raw';
      inputs = isBatch ? await Promise.all(inputs.map(toRawImage)) : await toRawImage(inputs);
    }
    if (IMAGE_OUTPUT_TASKS.includes(this.task)) {
      const raw = options.as_array === true;  // pixels instead of a PNG; not a Transformers.js option
      delete options.as_array;
      return await packImages(await this._callTask(inputs, options), raw);
    }
    if (SPEECH_TASKS.includes(this.task)) {
      // [sampling rate, the float32 samples as raw bytes]: one buffer crosses the bridge, not one value per sample.
      const { audio, sampling_rate } = await this._callTask(inputs, options);
      return [sampling_rate, new Uint8Array(audio.buffer.slice(audio.byteOffset, audio.byteOffset + audio.byteLength))];
    }
    if (this.task === 'zero-shot-classification' || this.task === 'zero-shot-image-classification') {
      const labels = options.candidate_labels;
      delete options.candidate_labels;
      if (labels === undefined || labels === null) {
        throw new TypeError(`${this.task} requires candidate_labels.`);
      }
      return await this._callTask(inputs, labels, options);
    }
    if (this.task === 'image-feature-extraction') {
      // One plain list of floats per image. Many exports have no pooler layer, so by default take the
      // first (CLS) token of the last hidden state; `pool: true` asks the model for its pooled output.
      let features = (await this._callTask(inputs, options)).tolist();
      if (Array.isArray(features[0]?.[0])) {
        features = features.map(tokens => tokens[0]);
      }
      return Array.isArray(inputs) ? features : features[0];
    }
    const output = await this._callTask(inputs, options);
    // A single translation input returns one result, like the other tasks.
    if (this.task === 'translation' && !Array.isArray(inputs) && Array.isArray(output) && output.length === 1) {
      return output[0];
    }
    return output;
  }

  // Runs `inputs` in batches of `batchSize` inputs per task call, calling onProgress(done, total) after each batch.
  // With continueOnError, a failing batch is retried input by input and the failures are returned as
  // [index, message] pairs, with null in place of their results; otherwise the first error is thrown.
  // cancel() stops it after the current batch. Returns {results, errors, cancelled}, where results covers the
  // inputs run so far.
  async runTasks(inputs, batchSize = 8, options = {}, onProgress = null, continueOnError = false) {
    if (this._batchRun) {
      throw new Error("Another map() is running on this instance; await it or call cancel() first.");
    }
    if (typeof options === 'string') {
      options = JSON.parse(options);
    }
    const run = this._batchRun = { cancelled: false };
    run.done = new Promise((resolve) => { run.finish = resolve; });
    const total = inputs.length;
    // A list of numbers is one audio clip, speech synthesis takes one text, and an image result is one structure:
    // those tasks run one input per call.
    const oneAtATime = AUDIO_INPUT_TASKS.includes(this.task) || SPEECH_TASKS.includes(this.task)
      || IMAGE_OUTPUT_TASKS.includes(this.task);
    const size = oneAtATime ? 1 : Math.max(1, batchSize);
    const callBatch = async (items) => {
      const taskOptions = { ...options };  // runTask() consumes candidate_labels
      const output = oneAtATime
        ? [await this.runTask(items[0], taskOptions)]
        : await this.runTask(items, taskOptions);
      if (!Array.isArray(output) || output.length !== items.length) {
        throw new Error(`${this.task} returned ${Array.isArray(output) ? output.length : 'no list of'} results for ${items.length} inputs`);
      }
      return output;
    };
    const results = [];
    const errors = [];
    try {
      for (let start = 0; start < total && !run.cancelled; start += size) {
        const items = inputs.slice(start, start + size);
        try {
          results.push(...await callBatch(items));
        } catch (error) {
          if (!continueOnError) {
            throw error;
          }
          for (const [offset, item] of items.entries()) {
            try {
              results.push(...await callBatch([item]));
            } catch (itemError) {
              results.push(null);
              errors.push([start + offset, String(itemError?.message ?? itemError)]);
            }
          }
        }
        onProgress?.(results.length, total);
      }
      return { results, errors, cancelled: run.cancelled };
    } finally {
      this._batchRun = null;
      run.finish();
    }
  }

  async _callTask(...args) {
    try {
      return await this.taskPipeline(...args);
    } catch (error) {
      throw new Error(`${this.task} failed: ${String(error?.message ?? error)}`, { cause: error });
    }
  }
}

// --- Bare ONNX models ---

// Transformers.js does not export onnxruntime-web, so load the build it uses: the frontend from the site's
// vendor/transformers/ when a build put it there, else from jsDelivr. The wasm files are Transformers.js's own.
let ortModule = null;

function loadOrt() {
  ortModule ??= (async () => {
    const { env } = await loadTransformers();
    const onnx = env.backends.onnx;
    const file = 'ort.webgpu.min.mjs';  // includes the WASM execution provider
    const root = siteRoot();
    let ort = null;
    if (root) {
      try {
        ort = await import(`${root}vendor/transformers/${file}`);
      } catch {
        // Not self-hosted: fall back to the CDN below.
      }
    }
    ort ??= await import(`https://cdn.jsdelivr.net/npm/onnxruntime-web@${onnx.versions.web}/dist/${file}`);
    ort.env.wasm.wasmPaths = onnx.wasm.wasmPaths;
    return ort;
  })().catch((error) => {
    ortModule = null;
    throw error;
  });
  return ortModule;
}

const TENSOR_ARRAYS = {
  float32: Float32Array, float64: Float64Array, int8: Int8Array, uint8: Uint8Array, int16: Int16Array,
  uint16: Uint16Array, int32: Int32Array, uint32: Uint32Array, int64: BigInt64Array, uint64: BigUint64Array,
  bool: Uint8Array,
  // Float16Array where the browser has it; onnxruntime-web also takes the raw half-float bits.
  float16: globalThis.Float16Array ?? Uint16Array,
};

class OnnxModel {
  constructor(session, info) {
    this.session = session;
    this.info = info;
  }

  // JSON, so the description crosses both Python bridges.
  describe() {
    return JSON.stringify(this.info);
  }

  // `meta` is JSON [{name, dtype, dims}]; `buffers` holds one Uint8Array of raw little-endian values per input.
  // Returns [JSON [{name, dtype, dims}], ...one Uint8Array per output].
  async run(meta, buffers) {
    const ort = await loadOrt();
    const feeds = {};
    JSON.parse(meta).forEach(({ name, dtype, dims }, i) => {
      const Array_ = TENSOR_ARRAYS[dtype];
      if (!Array_) {
        throw new TypeError(`Unsupported dtype '${dtype}' for input '${name}'.`);
      }
      const copy = new Uint8Array(buffers[i]);  // a fresh, aligned buffer
      feeds[name] = new ort.Tensor(dtype, new Array_(copy.buffer), dims);
    });
    const results = await this.session.run(feeds);
    const described = [];
    const outputs = [];
    for (const name of this.session.outputNames) {
      const tensor = results[name];
      const data = await tensor.getData();  // also reads back GPU outputs
      described.push({ name, dtype: tensor.type, dims: [...tensor.dims] });
      outputs.push(new Uint8Array(data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)));
      tensor.dispose?.();
    }
    return [JSON.stringify(described), ...outputs];
  }

  async dispose() {
    await this.session?.release();
    this.session = null;
  }
}

function describeValues(metadata, names) {
  const byName = new Map((metadata ?? []).map((value) => [value.name, value]));
  return names.map((name) => {
    const { type = null, shape = null } = byName.get(name) ?? {};
    return { name, dtype: type, shape };  // symbolic dimensions stay strings
  });
}

const ONNX_CACHE = 'lmlite-onnx-models';  // contains 'onnx', so clearModelCache() deletes it

async function sha256Hex(bytes) {
  if (!globalThis.crypto?.subtle) {
    return null;  // not a secure context
  }
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))]
    .map((byte) => byte.toString(16).padStart(2, '0')).join('');
}

// Bytes of a model URL, from the Cache API when they are there and match `sha256`. Returns {bytes, sha256, cache},
// where `cache` is 'cached' (served from the cache), 'downloaded' (stored) or null (not stored).
async function fetchOnnxBytes(url, expected, useCache) {
  const cache = useCache && typeof caches !== 'undefined' ? await caches.open(ONNX_CACHE) : null;
  const hit = await cache?.match(url);
  if (hit) {
    const bytes = new Uint8Array(await hit.arrayBuffer());
    const hash = hit.headers.get('x-lmlite-sha256') || await sha256Hex(bytes);
    if (!expected || hash === expected) {
      return { bytes, sha256: hash, cache: 'cached' };
    }
    await cache.delete(url);  // the file at this URL changed
  }
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} for ${url}`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  const hash = await sha256Hex(bytes);
  if (expected && hash !== expected) {
    throw new Error(`SHA-256 mismatch for ${url}: expected ${expected}, got ${hash}`);
  }
  let stored = null;
  try {
    await cache?.put(url, new Response(bytes, { headers: { 'x-lmlite-sha256': hash ?? '' } }));
    stored = cache ? 'downloaded' : null;
  } catch {
    // Quota exceeded or storage blocked: the model still loads, it is just not kept.
  }
  return { bytes, sha256: hash, cache: stored };
}

// `source` is a URL or the model's bytes. With `sha256`, the bytes must match it. URLs are kept in the Cache API
// unless `useCache` is false. Throws if the backend cannot create the session.
async function loadOnnxModel(source, backend = 'wasm', expectedSha256 = null, useCache = true) {
  const ort = await loadOrt();
  let loaded;
  if (typeof source === 'string') {
    loaded = await fetchOnnxBytes(source, expectedSha256, useCache);
  } else {
    const bytes = new Uint8Array(source);
    loaded = { bytes, sha256: await sha256Hex(bytes), cache: null };
    if (expectedSha256 && loaded.sha256 !== expectedSha256) {
      throw new Error(`SHA-256 mismatch: expected ${expectedSha256}, got ${loaded.sha256}`);
    }
  }
  const session = await ort.InferenceSession.create(loaded.bytes, { executionProviders: [backend] });
  return new OnnxModel(session, {
    backend,
    execution_provider: backend,  // one provider is requested, so creation succeeding means it is the one used
    sha256: loaded.sha256,
    size: loaded.bytes.byteLength,
    cache_status: loaded.cache,
    inputs: describeValues(session.inputMetadata, session.inputNames),
    outputs: describeValues(session.outputMetadata, session.outputNames),
  });
}

// --- Microphone recording ---

// Main thread only: kernels run in Web Workers, which have no navigator.mediaDevices, so the recorder widget loads
// this file on the page too. What Whisper and most speech recognition models expect:
const RECORDING_RATE = 16000;

// A 16-bit PCM mono WAV file from float samples in [-1, 1].
function encodeWav(samples, sampleRate) {
  const bytes = new Uint8Array(44 + samples.length * 2);
  const view = new DataView(bytes.buffer);
  const text = (offset, value) => [...value].forEach((char, i) => view.setUint8(offset + i, char.charCodeAt(0)));
  text(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  text(8, 'WAVE');
  text(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);  // PCM
  view.setUint16(22, 1, true);  // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  samples.forEach((sample, i) => view.setInt16(44 + i * 2, Math.round(Math.max(-1, Math.min(1, sample)) * 32767), true));
  return bytes;
}

// One channel from the channels of a recording. Averaging is best, but channels recorded in opposite polarity (some
// stereo inputs do) cancel to silence, and a dead channel halves the level, so when the average is clearly quieter
// than the loudest channel, use that one.
function mixChannels(channels) {
  if (channels.length === 1) {
    return channels[0];
  }
  const rms = (data) => Math.sqrt(data.reduce((sum, value) => sum + value * value, 0) / Math.max(1, data.length));
  const mix = new Float32Array(channels[0].length);
  for (const channel of channels) {
    channel.forEach((value, i) => { mix[i] += value / channels.length; });
  }
  const loudest = channels.reduce((best, channel) => (rms(channel) > rms(best) ? channel : best));
  return rms(mix) < 0.7 * rms(loudest) ? loudest : mix;
}

// Records the microphone and returns a 16 kHz mono WAV, which speech recognition pipelines take as bytes.
// After stop(), `info` describes the recording: seconds, peak (0 to 1, near 0 means silence), and the browser's
// channels, sampleRate and mimeType, with the `raw` option and the `gain` applied to it.
//
// By default the browser processes the microphone (echo cancellation, noise suppression, gain control). Any of
// them mixes the input to mono first, and a stereo input with opposite polarity in its channels mixes to exact
// silence. `start({raw: true})` turns all of them off, keeps the channels (mixChannels() deals with them) and, in
// place of the gain control, scales a quiet recording up to a peak of 0.9 (at most 20 times).
class AudioRecorder {
  constructor() {
    this.recorder = null;
    this.stream = null;
    this.chunks = [];
    this.info = null;
    this.raw = false;
  }

  get recording() {
    return this.recorder !== null;
  }

  async start({ raw = false } = {}) {
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      throw new Error('Recording needs a page on HTTPS or localhost, in a browser with MediaRecorder.');
    }
    if (this.recording) {
      throw new Error('Already recording.');
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: raw
        ? { channelCount: { ideal: 2 }, echoCancellation: false, noiseSuppression: false, autoGainControl: false }
        : true,
    });
    this.raw = raw;
    this.chunks = [];
    this.recorder = new MediaRecorder(this.stream);
    this.recorder.ondataavailable = (event) => event.data.size && this.chunks.push(event.data);
    this.recorder.start();
  }

  // Stops and returns the recording as WAV bytes.
  async stop() {
    const { recorder, stream } = this;
    if (!recorder) {
      throw new Error('Not recording.');
    }
    const stopped = new Promise((resolve) => { recorder.onstop = resolve; });
    recorder.stop();
    await stopped;
    stream.getTracks().forEach((track) => track.stop());
    this.recorder = this.stream = null;
    // Browsers record WebM/Opus (Chrome, Firefox) or MP4/AAC (Safari); AudioContext decodes both.
    const context = new AudioContext();
    try {
      const decoded = await context.decodeAudioData(await new Blob(this.chunks, { type: recorder.mimeType }).arrayBuffer());
      // An offline context of the target rate resamples; the channels stay apart until mixChannels() has looked at them.
      const offline = new OfflineAudioContext(
        decoded.numberOfChannels, Math.max(1, Math.ceil(decoded.duration * RECORDING_RATE)), RECORDING_RATE);
      const source = offline.createBufferSource();
      source.buffer = decoded;
      source.connect(offline.destination);
      source.start();
      const rendered = await offline.startRendering();
      const mixed = mixChannels(Array.from({ length: rendered.numberOfChannels }, (_, i) => rendered.getChannelData(i)));
      const peak = mixed.reduce((max, value) => Math.max(max, Math.abs(value)), 0);
      const gain = this.raw && peak >= 0.005 ? Math.min(20, 0.9 / peak) : 1;
      this.info = {
        seconds: mixed.length / RECORDING_RATE,
        peak,
        channels: decoded.numberOfChannels,
        sampleRate: decoded.sampleRate,
        mimeType: recorder.mimeType,
        raw: this.raw,
        gain,
      };
      return encodeWav(gain === 1 ? mixed : mixed.map((value) => value * gain), RECORDING_RATE);
    } finally {
      this.chunks = [];
      await context.close();
    }
  }
}

// --- Cache injection ---

async function isModelCached(modelName) {
  const cacheNames = await caches.keys();
  const candidates = modelIdCandidates(modelName);

  for (const cacheName of cacheNames) {
    const cache = await caches.open(cacheName);
    const requests = await cache.keys();

    const hasModel = requests.some((request) =>
      candidates.some(candidate => request.url.includes(candidate))
    );
    if (hasModel) {
      return true;
    }
  }

  return false;
}

async function loadModelToCache(modelName, files) {
  const cacheNames = await caches.keys();
  const modelId = resolveModelId(modelName);
  const cacheName = cacheNames.find(n =>
    n.includes('transformers') || n.includes('huggingface')
  ) || 'transformers-cache';

  const cache = await caches.open(cacheName);
  let count = 0;

  for (const [filename, data] of Object.entries(files)) {
    const url = `https://huggingface.co/${modelId}/resolve/main/${filename}`;
    await cache.put(url, new Response(data, {
      status: 200,
      // Content-Length lets getModelInfo() report sizes of injected files.
      headers: { 'Content-Type': 'application/octet-stream', 'Content-Length': String(data.byteLength ?? data.length) }
    }));
    console.log(`Cached: ${url} (${data.length} bytes)`);
    count++;
  }

  console.log(`Loaded ${count} files for ${modelId} into browser cache`);
}

async function modelCacheNames() {
  return (await caches.keys()).filter((name) => {
    const lowerName = name.toLowerCase();
    return lowerName.includes('transformers') || lowerName.includes('huggingface') || lowerName.includes('onnx');
  });
}

async function getCachedModels() {
  if (typeof caches === 'undefined') {
    return null;
  }
  const groups = new Map();
  for (const cacheName of await modelCacheNames()) {
    const cache = await caches.open(cacheName);
    for (const request of await cache.keys()) {
      const match = new URL(request.url).pathname.match(/^\/([^/]+\/[^/]+)\/resolve\/([^/]+)\/(.+)$/);
      const [modelId, revision, file] = match ? match.slice(1) : [null, null, request.url];
      const key = `${modelId}@${revision}`;
      if (!groups.has(key)) {
        groups.set(key, { model_id: modelId, revision, model_type: null, task: null, files: [], file_sizes: {}, size: 0 });
      }
      const group = groups.get(key);
      group.files.push(file);
      const response = await cache.match(request);
      const length = response?.headers.get('content-length');
      group.file_sizes[file] = length == null ? null : Number(length);
      group.size = group.size === null || length == null ? null : group.size + Number(length);
      if (modelId && file === 'config.json') {
        group.model_type = (await response.json().catch(() => null))?.model_type ?? null;
      }
    }
  }
  // The task comes from the Hub, so it stays null offline.
  await Promise.all([...groups.values()].filter(g => g.model_id).map(async (group) => {
    group.task = await getModelTask(group.model_id);
  }));
  return [...groups.values()];
}

async function clearModelCacheEntries(modelName) {
  // Exact prefix on the resolved ID only, so 'gpt2' never matches 'acme/gpt2'.
  const modelId = resolveModelId(modelName.split('@')[0]);
  const prefix = `/${modelId}/resolve/`;
  const deletedFiles = [];
  for (const cacheName of await modelCacheNames()) {
    const cache = await caches.open(cacheName);
    for (const request of await cache.keys()) {
      if (new URL(request.url).pathname.startsWith(prefix) && await cache.delete(request)) {
        deletedFiles.push(request.url);
      }
    }
  }
  return { model_id: modelId, deleted_files: deletedFiles };
}

async function clearModelCache(modelName = null) {
  if (modelName) {
    return clearModelCacheEntries(modelName);
  }
  const deletedCaches = [];
  const deletedDatabases = [];
  const knownDatabaseNames = ['transformers-cache', 'transformers_cache', 'huggingface-assets', 'transformers.js'];

  for (const cacheName of await modelCacheNames()) {
    if (await caches.delete(cacheName)) {
      deletedCaches.push(cacheName);
    }
  }

  const candidateDatabaseNames = new Set(knownDatabaseNames);
  if (typeof indexedDB.databases === 'function') {
    const runtimeDatabases = await indexedDB.databases();
    for (const db of runtimeDatabases) {
      if (db?.name) {
        candidateDatabaseNames.add(db.name);
      }
    }
  }

  for (const dbName of candidateDatabaseNames) {
    const lowerName = dbName.toLowerCase();
    if (
      lowerName.includes('transformers') ||
      lowerName.includes('huggingface') ||
      lowerName.includes('onnx')
    ) {
      await new Promise((resolve) => {
        const request = indexedDB.deleteDatabase(dbName);
        request.onsuccess = () => resolve();
        request.onerror = () => resolve();
        request.onblocked = () => resolve();
      });
      deletedDatabases.push(dbName);
    }
  }

  return {
    deleted_caches: deletedCaches,
    deleted_databases: deletedDatabases,
  };
}

// --- Model metadata and cache status ---

// Best-effort, pre-load view of the files a pipeline needs and whether the
// Transformers.js Cache API already holds them. Unknown values stay null/'unknown'.
// Without a task, the Hub's pipeline tag is used.
async function getModelInfo(modelName, task = null, backend = 'wasm', dtype = null) {
  const [modelId, revision = 'main'] = resolveModelId(modelName).split('@', 2);
  const info = {
    model_id: modelId, revision, task: task ?? null, backend, dtype: dtype ?? null,
    cache_status: 'unknown', download_bytes: null, files: [], error: null,
  };
  try {
    const { ModelRegistry, env } = await loadTransformers();
    env.useBrowserCache = true;
    if (!info.task) {
      info.task = await getModelTask(modelId);
      if (!info.task) {
        throw new Error(`No task found on the Hub for ${modelId}; pass task explicitly`);
      }
    }
    if (!info.dtype && info.task !== 'feature-extraction') {
      // Mirror _loadPipelineWithDtypeFallback; feature-extraction keeps the Transformers.js default.
      const available = await ModelRegistry.get_available_dtypes(modelId, { revision });
      info.dtype = DEFAULT_DTYPES.find(d => available.includes(d)) ?? null;
      if (!info.dtype) {
        throw new Error(`No ${DEFAULT_DTYPES.join(', ')} ONNX weights found for ${modelId}`);
      }
    }
    const options = { revision, device: backend, ...(info.dtype ? { dtype: info.dtype } : {}) };
    const { files } = await ModelRegistry.is_pipeline_cached_files(info.task, modelId, options);
    info.files = files.map(({ file, cached }) => ({ name: file, size: null, cached }));
    if (typeof caches !== 'undefined') {
      const cachedCount = files.filter(f => f.cached).length;
      info.cache_status = cachedCount === files.length ? 'cached' : cachedCount ? 'partial' : 'not_cached';
    }
    // Sizes come from cached headers or a 1-byte range request; offline misses stay null.
    const sizes = await Promise.allSettled(
      files.map(({ file }) => ModelRegistry.get_file_metadata(modelId, file, { revision }))
    );
    sizes.forEach((result, i) => {
      info.files[i].size = result.value?.size ?? null;
    });
    if (info.files.every(f => f.size !== null)) {
      info.download_bytes = info.files.reduce((total, f) => total + f.size, 0);
    }
  } catch (error) {
    info.error = String(error?.message ?? error);
  }
  return info;
}

// --- Tokenize-only handles ---

async function fetchTokenizerFile(url, optional = false) {
  const response = await fetch(url);
  if (optional && response.status === 404) {
    return {};
  }
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText} for ${url}`);
  }
  return response.json();
}

// Fetches only tokenizer.json and tokenizer_config.json: no weights, no Hub task lookup, and nothing
// written to the Cache API, where a tokenizer-only entry would make isModelCached() report the model cached.
async function loadTokenizer(modelName = null, tokenizerJson = null, tokenizerConfig = null, hostedModelsUrl = null) {
  try {
    let jsonUrl, configUrl;
    if (modelName) {
      const [modelId, revision = 'main'] = resolveModelId(modelName).split('@', 2);
      const base = await getHostedModelBase(modelName, hostedModelsUrl);
      const root = base ? new URL(`${modelId}/`, base).href : `https://huggingface.co/${modelId}/resolve/${revision}/`;
      [jsonUrl, configUrl] = [`${root}tokenizer.json`, `${root}tokenizer_config.json`];
    } else {
      // Relative paths resolve against the JupyterLite site root, like hosted files/models/.
      const resolve = (path) => {
        const url = new URL(path, siteRoot() ?? globalThis.location?.href);
        if (!['http:', 'https:'].includes(url.protocol)) {
          throw new Error(`tokenizer files must be HTTP(S) URLs, got ${url.href}`);
        }
        return url.href;
      };
      jsonUrl = resolve(tokenizerJson);
      configUrl = tokenizerConfig ? resolve(tokenizerConfig) : null;
    }
    const [transformers, json, config] = await Promise.all([
      loadTransformers(),
      fetchTokenizerFile(jsonUrl),
      // A model without tokenizer_config.json still tokenizes from tokenizer.json alone.
      configUrl ? fetchTokenizerFile(configUrl, Boolean(modelName)) : {},
    ]);
    // Pick the tokenizer class as AutoTokenizer does.
    const { PreTrainedTokenizer } = transformers;
    const name = config.tokenizer_class?.replace(/Fast$/, '');
    const named = name ? transformers[name] : null;
    const Tokenizer = named?.prototype instanceof PreTrainedTokenizer ? named : PreTrainedTokenizer;
    const tokenizer = new Tokenizer(json, config);
    return {
      vocab_size: tokenizer.get_vocab().size,
      // JSON keeps ids plain Python ints across both kernel bridges.
      encode: (text, addSpecialTokens = false) =>
        JSON.stringify(tokenizer.encode(text, { add_special_tokens: addSpecialTokens })),
      decode: (ids) => tokenizer.decode(JSON.parse(ids)),
    };
  } catch (error) {
    throw new Error(
      `Failed to load tokenizer from '${modelName ?? tokenizerJson}': ${String(error?.message ?? error)}`,
      { cause: error },
    );
  }
}

// --- Factory ---

function createRuntime(config) {
  return new BrowserRuntime(config);
}

// --- Similarity helpers ---

function cosineSimilarity(a, b) {
  let dot = 0.0, normA = 0.0, normB = 0.0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function cosineSimilarityBatch(queryVec, docVecs) {
  return docVecs.map(doc => cosineSimilarity(queryVec, doc));
}

// --- Model info helpers ---

async function fetchHubModel(modelId, timeoutMs = null) {
  try {
    const url = `https://huggingface.co/api/models/${modelId}`;
    const response = await fetch(url, timeoutMs ? { signal: AbortSignal.timeout(timeoutMs) } : undefined);
    if (!response.ok) {
      console.warn(`Failed to fetch model info: ${response.status} ${response.statusText}`);
      return null;
    }
    return await response.json();
  } catch (error) {
    console.warn(`Error fetching model info for ${modelId}:`, error);
    return null;
  }
}

async function getModelLatestRevision(modelId) {
  return (await fetchHubModel(modelId))?.sha || null;
}

// The Hub pipeline tag (e.g. 'text-generation'), or null offline or for untagged models.
async function getModelTask(modelName) {
  return (await fetchHubModel(resolveModelId(modelName).split('@')[0], 5000))?.pipeline_tag ?? null;
}

// --- expose to Python as globalThis.lmLite (critical) ---

// One namespace keeps the kernel's global scope clean.
globalThis.lmLite = {
  BrowserRuntime,
  createRuntime,
  // The names before the runtime covered more than language models; pages that call them keep working.
  BrowserLLM: BrowserRuntime,
  createLLM: createRuntime,
  isModelCached,
  loadModelToCache,
  clearModelCache,
  getCachedModels,
  cosineSimilarity,
  cosineSimilarityBatch,
  getModelLatestRevision,
  getHostedModelBase,
  getModelInfo,
  getModelTask,
  loadTokenizer,
  loadOnnxModel,
  AudioRecorder,
  encodeWav,
  mixChannels,
};

undefined;
