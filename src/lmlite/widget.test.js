// The request() handler of the camera and microphone widgets. Their JavaScript is a string inside the Python module,
// so it is read from there and run against a fake page and a fake widget model.
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { readFileSync } from 'node:fs';

const load = async (file) => {
  const source = readFileSync(new URL(`./${file}`, import.meta.url), 'utf8').match(/_ESM = """([\s\S]*?)"""/)[1];
  return (await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`)).default;
};

let elements;
const fakeElement = () => {
  const element = {
    style: {}, children: [], textContent: '', disabled: false, value: '',
    append(...nodes) { this.children.push(...nodes); },
    play: () => Promise.resolve(),
    getContext: () => ({ drawImage() {} }),
    toBlob: (done) => done({ arrayBuffer: async () => new Uint8Array([0xff, 0xd8, 0xff]).buffer }),
  };
  elements.push(element);
  return element;
};

// The widget model: values Python synced, what the page sent back, and the handlers Python's changes call.
const fakeModel = (values = {}) => {
  const handlers = {};
  const model = {
    values, sent: {},
    get: (name) => model.values[name],
    set(name, value) { model.sent[name] = value; model.values[name] = value; },
    save_changes() {},
    on: (event, handler) => { handlers[event] = handler; },
    python(name, value) { model.values[name] = value; handlers[`change:${name}`]?.(); },
  };
  return model;
};

const button = (text) => elements.filter((element) => element.textContent === text);
const barOf = (cancel) => elements.find((element) => element.children.includes(cancel));
const shown = (bar) => bar.style.display !== 'none';
const settle = () => new Promise((resolve) => setTimeout(resolve, 0));

beforeEach(() => {
  elements = [];
  globalThis.document = { createElement: fakeElement };
  globalThis.URL.createObjectURL = vi.fn(() => 'blob:fake');
});
afterEach(() => {
  delete globalThis.document;
  delete globalThis.navigator?.mediaDevices;
  delete globalThis.lmLite;
  vi.useRealTimers();
});

describe('CameraCapture request()', () => {
  let tracks, model, cancel, shoot, bar, video, mount;
  beforeEach(() => {
    tracks = [{ stop: vi.fn() }];
    vi.stubGlobal('navigator', { mediaDevices: { getUserMedia: vi.fn(async () => ({ getTracks: () => tracks })) } });
    mount = async (values) => {
      model = fakeModel({ pending_request: {}, ...values });
      const widget = await load('camera.py');
      const el = fakeElement();
      widget.render({ model, el });
      cancel = button('Cancel')[0];
      [, shoot] = button('Take photo');
      bar = barOf(cancel);
      video = elements[1];  // the first element created is the container
      video.videoWidth = video.videoHeight = 100;
      await settle();
      video.onplaying?.();
    };
  });
  afterEach(() => vi.unstubAllGlobals());

  it('shows the prompt and starts the camera, but captures nothing until the tap', async () => {
    await mount();
    model.python('pending_request', { id: 1, prompt: 'Tap to photograph the desk' });
    await settle();

    expect(shown(bar)).toBe(true);
    expect(bar.children[0].textContent).toBe('Tap to photograph the desk');
    expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(1);
    expect(model.sent.photo).toBeUndefined();
  });

  it('answers with the photo first and the request id after it, then hides the prompt and stops its camera', async () => {
    await mount();
    model.python('pending_request', { id: 1, prompt: 'Tap' });
    await settle();
    video.onplaying();
    const order = [];
    const set = model.set;
    model.set = (name, value) => { order.push(name); set(name, value); };
    await shoot.onclick();

    expect(order.indexOf('photo')).toBeLessThan(order.indexOf('answered'));
    expect(model.sent.answered).toBe(1);
    expect(shown(bar)).toBe(false);
    expect(tracks[0].stop).toHaveBeenCalled();
  });

  it('reports Cancel with the request id and stops its camera', async () => {
    await mount();
    model.python('pending_request', { id: 2, prompt: 'Tap' });
    await settle();
    cancel.onclick();

    expect(model.sent.cancelled).toBe(2);
    expect(shown(bar)).toBe(false);
    expect(tracks[0].stop).toHaveBeenCalled();
  });

  it('stops a camera that was granted after the request ended', async () => {
    let grant;
    navigator.mediaDevices.getUserMedia = vi.fn(() => new Promise((resolve) => { grant = resolve; }));
    await mount();
    model.python('pending_request', { id: 7, prompt: 'Tap' });
    cancel.onclick();  // while the permission prompt is open
    grant({ getTracks: () => tracks });
    await settle();

    expect(tracks[0].stop).toHaveBeenCalled();
  });

  it('reports a refused camera with the request id and name', async () => {
    navigator.mediaDevices.getUserMedia = vi.fn(async () => { throw Object.assign(new Error('no'), { name: 'NotAllowedError' }); });
    await mount();
    model.python('pending_request', { id: 3, prompt: 'Tap' });
    await settle();

    expect(model.sent.error).toEqual({ id: 3, name: 'NotAllowedError', message: 'no' });
    expect(shown(bar)).toBe(false);
  });

  it('hides the prompt when Python clears the request (a timeout), and does not show it again', async () => {
    await mount();
    model.python('pending_request', { id: 4, prompt: 'Tap' });
    await settle();
    model.python('pending_request', {});
    expect(shown(bar)).toBe(false);
    expect(tracks[0].stop).toHaveBeenCalled();

    model.python('pending_request', { id: 4, prompt: 'Tap' });
    expect(shown(bar)).toBe(false);
  });

  it('shows a request made before the widget was rendered', async () => {
    await mount({ pending_request: { id: 5, prompt: 'Early' } });

    expect(shown(bar)).toBe(true);
    expect(bar.children[0].textContent).toBe('Early');
  });
});

describe('AudioRecorder request()', () => {
  let recorder, model, cancel, bar, mount;
  beforeEach(() => {
    recorder = {
      recording: false,
      info: { seconds: 1, peak: 0.5, channels: 1, sampleRate: 16000, mimeType: 'audio/wav' },
      start: vi.fn(async () => { recorder.recording = true; }),
      stop: vi.fn(async () => { recorder.recording = false; return new Uint8Array([82, 73, 70, 70]); }),
    };
    globalThis.lmLite = { AudioRecorder: function () { return recorder; } };
    mount = async (values) => {
      model = fakeModel({ pending_request: {}, ...values });
      const widget = await load('audio.py');
      await widget.render({ model, el: fakeElement() });
      cancel = button('Cancel')[0];
      bar = barOf(cancel);
    };
  });

  const record = async () => {
    const go = button('Record')[1];  // the request's button; the first is the widget's own
    go.onclick();
    await settle();
    return go;
  };

  it('shows the prompt and records nothing until the tap', async () => {
    await mount();
    model.python('pending_request', { id: 1, prompt: 'Say it' });

    expect(shown(bar)).toBe(true);
    expect(bar.children[0].textContent).toBe('Say it');
    expect(recorder.start).not.toHaveBeenCalled();
  });

  it('answers with the recording first and the request id after it, then hides the prompt', async () => {
    await mount();
    model.python('pending_request', { id: 1, prompt: 'Say it' });
    const go = await record();
    expect(go.textContent).toBe('Stop');
    const order = [];
    const set = model.set;
    model.set = (name, value) => { order.push(name); set(name, value); };
    go.onclick();
    await settle();

    expect(order.indexOf('recording')).toBeLessThan(order.indexOf('answered'));
    expect(model.sent.answered).toBe(1);
    expect(shown(bar)).toBe(false);
  });

  it('stops by itself after max_seconds and answers', async () => {
    vi.useFakeTimers();
    await mount();
    model.python('pending_request', { id: 1, prompt: 'Say it', max_seconds: 2 });
    button('Record')[1].onclick();
    await vi.advanceTimersByTimeAsync(0);
    expect(recorder.recording).toBe(true);
    await vi.advanceTimersByTimeAsync(2000);

    expect(recorder.stop).toHaveBeenCalled();
    expect(model.sent.answered).toBe(1);
  });

  it('on Cancel during a recording, releases the microphone, drops the recording and reports the id', async () => {
    await mount();
    model.python('pending_request', { id: 2, prompt: 'Say it' });
    await record();
    await cancel.onclick();

    expect(recorder.stop).toHaveBeenCalled();
    expect(model.sent.cancelled).toBe(2);
    expect(model.sent.recording).toBeUndefined();
    expect(shown(bar)).toBe(false);
  });

  it('stops a microphone that was granted after the request ended', async () => {
    let grant;
    recorder.start = vi.fn(() => new Promise((resolve) => { grant = () => { recorder.recording = true; resolve(); }; }));
    await mount();
    model.python('pending_request', { id: 7, prompt: 'Say it' });
    button('Record')[1].onclick();
    await cancel.onclick();  // while the permission prompt is open
    grant();
    await settle();

    expect(recorder.stop).toHaveBeenCalled();
    expect(model.sent.active).not.toBe(true);
  });

  it('reports a refused microphone with the request id and name', async () => {
    recorder.start = vi.fn(async () => { throw Object.assign(new Error('denied'), { name: 'NotAllowedError' }); });
    await mount();
    model.python('pending_request', { id: 3, prompt: 'Say it' });
    await record();

    expect(model.sent.error).toEqual({ id: 3, name: 'NotAllowedError', message: 'denied' });
    expect(shown(bar)).toBe(false);
  });

  it('drops a recording that started before the request, so it cannot answer it', async () => {
    await mount();
    await recorder.start();
    model.python('pending_request', { id: 4, prompt: 'Say it' });
    await settle();

    expect(recorder.stop).toHaveBeenCalled();
    expect(model.sent.answered).toBeUndefined();
  });

  it('hides the prompt when Python clears the request, and does not show it again', async () => {
    await mount();
    model.python('pending_request', { id: 5, prompt: 'Say it' });
    model.python('pending_request', {});
    expect(shown(bar)).toBe(false);

    model.python('pending_request', { id: 5, prompt: 'Say it' });
    expect(shown(bar)).toBe(false);
  });

  it('shows a request made before the widget was rendered', async () => {
    await mount({ pending_request: { id: 6, prompt: 'Early' } });

    expect(shown(bar)).toBe(true);
  });
});
