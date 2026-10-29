// lmLite JS runtime

const DEFAULT_MODEL_NAMESPACE = 'Xenova';

const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers';

// Transformers.js tasks exposed through the generic pipeline interface.
// Generation, embeddings and text classification keep their dedicated methods.
const SUPPORTED_TASKS = ['zero-shot-classification', 'automatic-speech-recognition', 'translation'];

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

async function getHostedModelBase(modelName, configuredBase = null) {
  const [modelId, revision] = resolveModelId(modelName).split('@', 2);
  // Hosted trees represent one snapshot, not arbitrary Hub revisions.
  if (revision && revision !== 'main') {
    return null;
  }
  let base = configuredBase;
  if (!base) {
    const location = globalThis.location;
    if (!location || !['http:', 'https:'].includes(location.protocol)) {
      return null;
    }
    const marker = location.pathname.match(/\/(?:extensions|lab|retro|tree)\//);
    if (!marker) {
      return null;
    }
    base = `${location.origin}${location.pathname.slice(0, marker.index)}/files/models/`;
  }
  const url = new URL(base.endsWith('/') ? base : `${base}/`, globalThis.location?.href);
  if (!['http:', 'https:'].includes(url.protocol)) {
    throw new Error('hosted_models_url must be an HTTP(S) URL');
  }
  const response = await fetch(new URL('manifest.json', url), { signal: AbortSignal.timeout(15000) });
  if (response.status === 404 && !configuredBase) {
    return null;
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

class BrowserLLM {
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
    for (const dtype of this._resolveDtypes(dtypes)) {
      try {
        return await pipelineFn(task, modelId, { ...extraOptions, dtype });
      } catch (error) {
        const msg = String(error?.message ?? error);
        if (this._isRecoverableDtypeError(msg)) {
          console.warn(`dtype '${dtype}' not available for ${modelId}, trying next…`);
          lastError = error;
        } else {
          throw error;
        }
      }
    }
    throw lastError;
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
    const { pipeline, env } = await import(TRANSFORMERS_URL);

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
    }
    env.useBrowserCache = true;
    console.log('Using browser cache (Hugging Face Cache API - Transformers.js v4)');
    const sourceOptions = (model) => this.hostedModels.includes(model)
      ? { local_files_only: true }
      : {};

    if (this.generatorModel) {
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
        await this.generator("test", { max_new_tokens: 1 });
        console.log(`Model warm-up complete`);
      } catch (error) {
        const errorMessage = String(error?.message ?? error);
        throw new Error(
          `Failed to load text-generation model '${this.generatorModel}': ${errorMessage}. ` +
          `This model may not be supported by the current Transformers.js browser runtime or backend '${this.backend}'. ` +
          `Check the WebGPU guide and model/dtype availability in the Transformers.js docs before picking a model.`
        );
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
        throw new Error(
          `Failed to load ${this.task} model '${this.taskModel}': ${String(error?.message ?? error)}. ` +
          `This model may not be supported by the current Transformers.js browser runtime or backend '${this.backend}'.`
        );
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

  async generate(prompt, returnFullText = this.return_full_text, streamer = undefined) {
    if (!this.generator) {
      throw new Error("Generator model not initialized. Call init() first.");
    }
    const { InterruptableStoppingCriteria } = await import(TRANSFORMERS_URL);
    if (this._generation?.cancelled) {
      // An interrupted generation stops after its current token; let it finish.
      await this._generation.done;
    }
    if (this._generation) {
      throw new Error("Another generation is running on this instance; await it or call cancel() first.");
    }
    const generation = this._generation = { stopping_criteria: new InterruptableStoppingCriteria(), cancelled: false };
    const run = (async () => {
      try {
        return await this.generator(prompt, {
          max_new_tokens: this.max_new_tokens,
          return_full_text: returnFullText,
          temperature: this.temperature,
          top_k: this.top_k,
          do_sample: this.do_sample,
          streamer,
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

  // Stops the running generate()/chat()/stream after the current token; they return the partial text.
  async cancel() {
    const generation = this._generation;
    if (!generation) {
      return false;
    }
    generation.cancelled = true;
    generation.stopping_criteria.interrupt();
    await generation.done;
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

  // Pull-based so both Python bridges can await chunks without JS→Python callbacks.
  async _stream(input) {
    if (!this.generator) {
      throw new Error("Generator model not initialized. Call init() first.");
    }
    const { TextStreamer } = await import(TRANSFORMERS_URL);
    const items = [];
    let wake = () => {};
    const push = (item) => {
      items.push(item);
      wake();
    };
    const streamer = new TextStreamer(this.generator.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: push,
    });
    this.generate(input, false, streamer).then(() => push(''), push);
    return {
      // Resolves to the next nonempty text chunk, '' once generation ends, or rejects with its error.
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
    if (this.task === 'automatic-speech-recognition' && typeof inputs === 'string' && typeof AudioContext === 'undefined') {
      // Kernels run in Web Workers, where Transformers.js cannot decode URLs without AudioContext.
      inputs = await fetchWavSamples(inputs);
    } else if (this.task === 'automatic-speech-recognition' && typeof inputs !== 'string' && !(inputs instanceof Float32Array)) {
      // Raw samples (e.g. a Python list or Float64Array) become the Float32Array audio input.
      inputs = Float32Array.from(inputs);
    }
    if (this.task === 'zero-shot-classification') {
      const labels = options.candidate_labels;
      delete options.candidate_labels;
      if (labels === undefined || labels === null) {
        throw new TypeError("zero-shot-classification requires candidate_labels.");
      }
      return await this._callTask(inputs, labels, options);
    }
    const output = await this._callTask(inputs, options);
    // A single translation input returns one result, like the other tasks.
    if (this.task === 'translation' && !Array.isArray(inputs) && Array.isArray(output) && output.length === 1) {
      return output[0];
    }
    return output;
  }

  async _callTask(...args) {
    try {
      return await this.taskPipeline(...args);
    } catch (error) {
      throw new Error(`${this.task} failed: ${String(error?.message ?? error)}`, { cause: error });
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
        groups.set(key, { model_id: modelId, revision, model_type: null, task: null, files: [], size: 0 });
      }
      const group = groups.get(key);
      group.files.push(file);
      const response = await cache.match(request);
      const length = response?.headers.get('content-length');
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
    const { ModelRegistry, env } = await import(TRANSFORMERS_URL);
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

// --- Factory ---

function createLLM(config) {
  return new BrowserLLM(config);
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

// --- expose globally (critical for Python) ---

globalThis.BrowserLLM = BrowserLLM;
globalThis.createLLM = createLLM;
globalThis.isModelCached = isModelCached;
globalThis.loadModelToCache = loadModelToCache;
globalThis.clearModelCache = clearModelCache;
globalThis.getCachedModels = getCachedModels;
globalThis.cosineSimilarity = cosineSimilarity;
globalThis.cosineSimilarityBatch = cosineSimilarityBatch;
globalThis.getModelLatestRevision = getModelLatestRevision;
globalThis.getHostedModelBase = getHostedModelBase;
globalThis.getModelInfo = getModelInfo;
globalThis.getModelTask = getModelTask;

undefined;
