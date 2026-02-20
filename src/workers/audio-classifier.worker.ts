import {
  AudioClassifier,
  AudioClassifierResult
} from '@mediapipe/tasks-audio';
import {
  FilesetResolver
} from '@mediapipe/tasks-vision';

// @ts-ignore
if (typeof self.import === 'undefined') {
  // @ts-ignore
  self.import = (url) => import(/* @vite-ignore */ url);
}

// MediaPipe Emscripten fallback
// @ts-ignore
self.createMediapipeTasksAudioModule = self.createMediapipeTasksAudioModule || undefined;

let audioClassifier: AudioClassifier | undefined = undefined;
let isInitializing = false;
let currentOptions: any = {};
let basePath = '/';

let isProcessing = false;

self.onmessage = async (event) => {
  const { type } = event.data;

  // Simple queue/lock
  while (isProcessing) {
    await new Promise(resolve => setTimeout(() => resolve(true), 10));
  }
  isProcessing = true;

  try {
    if (type === 'INIT') {
      const { modelAssetPath, delegate, maxResults, scoreThreshold, runningMode, baseUrl } = event.data;
      basePath = baseUrl || '/';
      currentOptions = { modelAssetPath, delegate, maxResults, scoreThreshold, runningMode };
      await initClassifier();
      self.postMessage({ type: 'INIT_DONE' });
    } else if (type === 'SET_OPTIONS') {
      if ('maxResults' in event.data) currentOptions.maxResults = event.data.maxResults;
      if ('scoreThreshold' in event.data) currentOptions.scoreThreshold = event.data.scoreThreshold;

      if (audioClassifier) {
        await audioClassifier.setOptions({
          maxResults: currentOptions.maxResults,
          scoreThreshold: currentOptions.scoreThreshold
        });
        self.postMessage({ type: 'OPTIONS_UPDATED' });
      }
    } else if (type === 'CLASSIFY') {
      const { audioData, sampleRate } = event.data;
      if (!audioClassifier) {
        console.warn('AudioClassifier not initialized yet.');
        self.postMessage({ type: 'CLASSIFY_ERROR', error: 'Not initialized' });
        return;
      }

      const startTimeMs = performance.now();
      let results: AudioClassifierResult[] = [];

      try {
        results = audioClassifier.classify(audioData, sampleRate);
      } catch (e: any) {
        console.error("Worker classification error:", e);
        self.postMessage({ type: 'CLASSIFY_ERROR', error: e.message || 'Classification failed' });
        return;
      }

      const inferenceTime = performance.now() - startTimeMs;
      self.postMessage({
        type: 'CLASSIFY_RESULT',
        results: results,
        inferenceTime: inferenceTime
      });
    } else if (type === 'CLEANUP') {
      if (audioClassifier) {
        audioClassifier.close();
        audioClassifier = undefined;
      }
      self.postMessage({ type: 'CLEANUP_DONE' });
    }
  } catch (error: any) {
    console.error("Audio Classifier Worker Error:", error);
    self.postMessage({ type: 'ERROR', error: error?.message || String(error) });
  } finally {
    isProcessing = false;
  }
};

async function loadModel(path: string) {
  const response = await fetch(path);
  const reader = response.body?.getReader();
  const contentLength = +response.headers.get('Content-Length')!;

  if (!reader) {
    return await response.arrayBuffer();
  }

  let receivedLength = 0;
  const chunks = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    receivedLength += value.length;
    self.postMessage({
      type: 'LOAD_PROGRESS',
      loaded: receivedLength,
      total: contentLength
    });
  }

  const chunksAll = new Uint8Array(receivedLength);
  let position = 0;
  for (let chunk of chunks) {
    chunksAll.set(chunk, position);
    position += chunk.length;
  }

  return chunksAll.buffer;
}

async function initClassifier() {
  if (isInitializing) return;
  isInitializing = true;

  try {
    if (audioClassifier) {
      audioClassifier.close();
      audioClassifier = undefined;
    }

    const wasmPath = new URL(`${basePath}wasm`, self.location.origin).href;
    console.log(`Loading audio tasks from ${wasmPath}`);

    // Fetch model manually for progress
    const modelBuffer = await loadModel(currentOptions.modelAssetPath);

    // WORKAROUND: Vite + MediaPipe module workers fail to inject ModuleFactory via importScripts.
    // We must manually fetch the WASM loader and eval it in the global scope.
    const wasmLoaderUrl = `${wasmPath}/audio_wasm_internal.js`;
    try {
      const response = await fetch(wasmLoaderUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch WASM loader: ${response.status} ${response.statusText}`);
      }
      const loaderCode = await response.text();
      (0, eval)(loaderCode);
      console.log('Manually loaded audio_wasm_internal.js');
    } catch (e) {
      console.error('Failed to manually load WASM loader:', e);
    }

    let audioFileset;
    // Check if FilesetResolver has forAudioTasks (it might not if imported from vision)
    // @ts-ignore
    if (typeof FilesetResolver.forAudioTasks === 'function') {
      // @ts-ignore
      audioFileset = await FilesetResolver.forAudioTasks(wasmPath);
    } else {
      console.warn('FilesetResolver.forAudioTasks missing, constructing manually');
      audioFileset = {
        wasmLoaderPath: `${wasmPath}/audio_wasm_internal.js`,
        wasmBinaryPath: `${wasmPath}/audio_wasm_internal.wasm`
      };
    }

    audioClassifier = await AudioClassifier.createFromOptions(audioFileset, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
      maxResults: currentOptions.maxResults,
      scoreThreshold: currentOptions.scoreThreshold
    });
    console.log('AudioClassifier created');

  } catch (e) {
    console.error('Worker initialization failed:', e);
    throw e;
  } finally {
    isInitializing = false;
  }
}
