import {
  AudioClassifier,
  AudioClassifierResult
} from '@mediapipe/tasks-audio';
import { loadModel } from '../utils/model-loader';
import { loadWasmModule } from '../utils/wasm-loader';

// @ts-ignore: Pre-declare the factory MediaPipe will try to attach
self.createMediapipeTasksAudioModule = self.createMediapipeTasksAudioModule || undefined;

let audioClassifier: AudioClassifier | undefined = undefined;
let isInitializing = false;
let currentOptions: any = {};
let basePath = '/';
let isProcessing = false;

self.onmessage = async (event) => {
  const { type } = event.data;

  while (isProcessing) {
    await new Promise(resolve => setTimeout(resolve, 10));
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

      self.postMessage({
        type: 'CLASSIFY_RESULT',
        results: results,
        inferenceTime: performance.now() - startTimeMs
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

async function initClassifier() {
  if (isInitializing) return;
  isInitializing = true;

  try {
    if (audioClassifier) {
      audioClassifier.close();
      audioClassifier = undefined;
    }

    const formattedBasePath = basePath.endsWith('/') ? basePath : `${basePath}/`;
    const wasmPath = new URL(`${formattedBasePath}wasm`, self.location.origin).href.replace(/\/$/, '');

    // Fetch model manually for progress
    const modelBuffer = await loadModel(currentOptions.modelAssetPath, (loaded, total) => {
      self.postMessage({ type: 'LOAD_PROGRESS', loaded, total });
    });

    // 1. Inject the loader using our shared utility
    await loadWasmModule(`${wasmPath}/audio_wasm_internal.js`);

    // 2. Hardcoded SIMD paths (Bypasses buggy FilesetResolver)
    const audioFileset = {
      wasmLoaderPath: `${wasmPath}/audio_wasm_internal.js`,
      wasmBinaryPath: `${wasmPath}/audio_wasm_internal.wasm`
    };

    // 3. Initialize
    audioClassifier = await AudioClassifier.createFromOptions(audioFileset, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
      maxResults: currentOptions.maxResults,
      scoreThreshold: currentOptions.scoreThreshold
    });

    console.log('AudioClassifier created successfully');

  } catch (e) {
    console.error('Worker initialization failed:', e);
    throw e;
  } finally {
    isInitializing = false;
  }
}
