import { loadModel } from '../utils/model-loader';
import {
  HandLandmarker,
  FilesetResolver,
  HandLandmarkerResult
} from '@mediapipe/tasks-vision';

import { loadWasmModule } from '../utils/wasm-loader';

// MediaPipe Emscripten fallback
// @ts-ignore
self.createMediapipeTasksVisionModule = self.createMediapipeTasksVisionModule || undefined;

let handLandmarker: HandLandmarker | undefined = undefined;
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
      const { modelAssetPath, delegate, runningMode, numHands, minHandDetectionConfidence, minHandPresenceConfidence, minTrackingConfidence, baseUrl } = event.data;
      basePath = baseUrl || '/';
      currentOptions = { modelAssetPath, delegate, runningMode, numHands, minHandDetectionConfidence, minHandPresenceConfidence, minTrackingConfidence };
      await initLandmarker();
    } else if (type === 'SET_OPTIONS') {
      let needsUpdate = false;
      if ('numHands' in event.data) { currentOptions.numHands = event.data.numHands; needsUpdate = true; }
      if ('minHandDetectionConfidence' in event.data) { currentOptions.minHandDetectionConfidence = event.data.minHandDetectionConfidence; needsUpdate = true; }
      if ('minHandPresenceConfidence' in event.data) { currentOptions.minHandPresenceConfidence = event.data.minHandPresenceConfidence; needsUpdate = true; }
      if ('minTrackingConfidence' in event.data) { currentOptions.minTrackingConfidence = event.data.minTrackingConfidence; needsUpdate = true; }
      if ('runningMode' in event.data) { currentOptions.runningMode = event.data.runningMode; needsUpdate = true; }

      if (handLandmarker && needsUpdate) {
        await handLandmarker.setOptions({
          numHands: currentOptions.numHands,
          minHandDetectionConfidence: currentOptions.minHandDetectionConfidence,
          minHandPresenceConfidence: currentOptions.minHandPresenceConfidence,
          minTrackingConfidence: currentOptions.minTrackingConfidence,
          runningMode: currentOptions.runningMode
        });
        self.postMessage({ type: 'OPTIONS_UPDATED' });
      }
    } else if (type === 'DETECT_IMAGE' || type === 'DETECT_VIDEO') {
      const { bitmap, timestampMs } = event.data;
      if (!handLandmarker) {
        console.warn('HandLandmarker not initialized yet.');
        bitmap.close();
        self.postMessage({ type: 'DETECT_ERROR', error: 'Not initialized' });
        return;
      }

      const requiredMode = type === 'DETECT_IMAGE' ? 'IMAGE' : 'VIDEO';
      if (currentOptions.runningMode !== requiredMode) {
        currentOptions.runningMode = requiredMode;
        await handLandmarker.setOptions({ runningMode: requiredMode });
      }

      const startTimeMs = performance.now();
      let result: HandLandmarkerResult;

      try {
        if (requiredMode === 'VIDEO') {
          result = handLandmarker.detectForVideo(bitmap, timestampMs);
        } else {
          result = handLandmarker.detect(bitmap);
        }
      } catch (e: any) {
        console.error("Worker detection error:", e);
        bitmap.close();
        self.postMessage({ type: 'DETECT_ERROR', error: e.message || 'Detection failed' });
        return;
      }

      const inferenceTime = performance.now() - startTimeMs;
      bitmap.close();

      self.postMessage({
        type: 'DETECT_RESULT',
        mode: requiredMode,
        result: result,
        inferenceTime: inferenceTime
      });
    } else if (type === 'CLEANUP') {
      if (handLandmarker) {
        handLandmarker.close();
        handLandmarker = undefined;
      }
      self.postMessage({ type: 'CLEANUP_DONE' });
    }
  } catch (error: any) {
    console.error("Hand Landmarker Worker Error:", error);
    self.postMessage({ type: 'ERROR', error: error?.message || String(error) });
  } finally {
    isProcessing = false;
  }
};

async function initLandmarker() {
  if (isInitializing) return;
  isInitializing = true;

  try {
    if (handLandmarker) {
      handLandmarker.close();
      handLandmarker = undefined;
    }

    const wasmPath = new URL(`${basePath}wasm`, self.location.origin).href;

    // WORKAROUND: Vite + MediaPipe module workers fail to inject ModuleFactory via importScripts.
    const wasmLoaderUrl = `${wasmPath}/vision_wasm_internal.js`;
    // Inject the loader using our shared utility
    await loadWasmModule(wasmLoaderUrl);

    const vision = await FilesetResolver.forVisionTasks(wasmPath);
    const modelBuffer = await loadModel(currentOptions.modelAssetPath, (loaded, total) => {
      self.postMessage({
        type: 'LOAD_PROGRESS',
        loaded,
        total
      });
    });

    if (currentOptions.delegate === 'GPU') {
      console.warn('[Worker] GPU Delegate requested.');
    }

    const options = {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: currentOptions.delegate,
      },
      numHands: currentOptions.numHands || 2,
      minHandDetectionConfidence: currentOptions.minHandDetectionConfidence || 0.5,
      minHandPresenceConfidence: currentOptions.minHandPresenceConfidence || 0.5,
      minTrackingConfidence: currentOptions.minTrackingConfidence || 0.5,
      runningMode: currentOptions.runningMode,
    };

    try {
      handLandmarker = await HandLandmarker.createFromOptions(vision, options);
    } catch (finalError) {
      console.error('HandLandmarker initialization failed:', finalError);
      if (currentOptions.delegate === 'GPU') {
        console.warn('GPU init failed, falling back to CPU', finalError);
        self.postMessage({ type: 'DELEGATE_FALLBACK', newDelegate: 'CPU' });
        options.baseOptions.delegate = 'CPU';
        handLandmarker = await HandLandmarker.createFromOptions(vision, options);
      } else {
        throw finalError;
      }
    }
    self.postMessage({ type: 'INIT_DONE' });

  } catch (error: any) {
    console.error("Hand Landmarker Worker Init Error:", error);
    throw error;
  } finally {
    isInitializing = false;
  }
}
