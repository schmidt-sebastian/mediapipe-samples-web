import {
  GestureRecognizer,
  FilesetResolver,
  GestureRecognizerResult
} from '@mediapipe/tasks-vision';
import { loadModel } from '../utils/model-loader';
import { loadWasmModule } from '../utils/wasm-loader';

// MediaPipe Emscripten fallback
// @ts-ignore
self.createMediapipeTasksVisionModule = self.createMediapipeTasksVisionModule || undefined;

let gestureRecognizer: GestureRecognizer | undefined = undefined;
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
      const { modelAssetPath, delegate, minHandDetectionConfidence, minHandPresenceConfidence, minTrackingConfidence, numHands, runningMode, baseUrl } = event.data;
      basePath = baseUrl || '/';
      currentOptions = { modelAssetPath, delegate, minHandDetectionConfidence, minHandPresenceConfidence, minTrackingConfidence, numHands, runningMode };
      await initDetector();
    } else if (type === 'SET_OPTIONS') {
      if ('minHandDetectionConfidence' in event.data) currentOptions.minHandDetectionConfidence = event.data.minHandDetectionConfidence;
      if ('minHandPresenceConfidence' in event.data) currentOptions.minHandPresenceConfidence = event.data.minHandPresenceConfidence;
      if ('minTrackingConfidence' in event.data) currentOptions.minTrackingConfidence = event.data.minTrackingConfidence;
      if ('numHands' in event.data) currentOptions.numHands = event.data.numHands;
      if ('runningMode' in event.data) currentOptions.runningMode = event.data.runningMode;

      if (gestureRecognizer) {
        await gestureRecognizer.setOptions({
          minHandDetectionConfidence: currentOptions.minHandDetectionConfidence,
          minHandPresenceConfidence: currentOptions.minHandPresenceConfidence,
          minTrackingConfidence: currentOptions.minTrackingConfidence,
          numHands: currentOptions.numHands,
          runningMode: currentOptions.runningMode
        });
        self.postMessage({ type: 'OPTIONS_UPDATED' });
      }
    } else if (type === 'DETECT_IMAGE' || type === 'DETECT_VIDEO') {
      const { bitmap, timestampMs } = event.data;
      if (!gestureRecognizer) {
        // console.warn('GestureRecognizer not initialized yet.');
        bitmap.close();
        self.postMessage({ type: 'DETECT_ERROR', error: 'Not initialized' });
        return;
      }

      const requiredMode = type === 'DETECT_IMAGE' ? 'IMAGE' : 'VIDEO';
      if (currentOptions.runningMode !== requiredMode) {
        currentOptions.runningMode = requiredMode;
        await gestureRecognizer.setOptions({ runningMode: requiredMode });
      }

      const startTimeMs = performance.now();
      let result: GestureRecognizerResult;

      try {
        if (requiredMode === 'VIDEO') {
          result = gestureRecognizer.recognizeForVideo(bitmap, timestampMs);
        } else {
          result = gestureRecognizer.recognize(bitmap);
        }
      } catch (e: any) {
        console.error("Worker recognition error:", e);
        bitmap.close();
        self.postMessage({ type: 'DETECT_ERROR', error: e.message || 'Recognition failed' });
        return;
      }

      const inferenceTime = performance.now() - startTimeMs;
      bitmap.close();

      // @ts-ignore
      (self as any).postMessage({
        type: 'DETECT_RESULT',
        mode: requiredMode,
        result: result,
        inferenceTime: inferenceTime
      });
    } else if (type === 'CLEANUP') {
      if (gestureRecognizer) {
        gestureRecognizer.close();
        gestureRecognizer = undefined;
      }
      self.postMessage({ type: 'CLEANUP_DONE' });
    }
  } catch (error: any) {
    console.error("Gesture Recognizer Worker Error:", error);
    self.postMessage({ type: 'ERROR', error: error?.message || String(error) });
  } finally {
    isProcessing = false;
  }
};

async function initDetector() {
  console.log('[Worker] initDetector started');
  if (isInitializing) return;
  isInitializing = true;

  try {
    if (gestureRecognizer) {
      gestureRecognizer.close();
      gestureRecognizer = undefined;
    }

    const wasmPath = new URL(`${basePath}wasm`, self.location.origin).href;
    console.log('[Worker] WASM path:', wasmPath);
    const wasmLoaderUrl = `${wasmPath}/vision_wasm_internal.js`;

    await loadWasmModule(wasmLoaderUrl, ';ModuleFactory;');
    console.log('[Worker] WASM loaded');

    const vision = await FilesetResolver.forVisionTasks(wasmPath);
    console.log('[Worker] FilesetResolver initialized');

    const modelBuffer = await loadModel(currentOptions.modelAssetPath, (loaded, total) => {
      self.postMessage({
        type: 'LOAD_PROGRESS',
        loaded,
        total
      });
    });
    console.log('[Worker] Model loaded, size:', modelBuffer.byteLength);

    if (currentOptions.delegate === 'GPU') {
      console.warn('[Worker] GPU Delegate requested.');
    }

    try {
      gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
          modelAssetBuffer: new Uint8Array(modelBuffer),
          delegate: currentOptions.delegate,
        },
        minHandDetectionConfidence: currentOptions.minHandDetectionConfidence,
        minHandPresenceConfidence: currentOptions.minHandPresenceConfidence,
        minTrackingConfidence: currentOptions.minTrackingConfidence,
        numHands: currentOptions.numHands,
        runningMode: currentOptions.runningMode,
      });
      console.log('[Worker] GestureRecognizer created successfully');
    } catch (finalError) {
      console.error('GestureRecognizer initialization failed:', finalError);
      if (currentOptions.delegate === 'GPU') {
        console.warn('GPU init failed, falling back to CPU', finalError);
        self.postMessage({ type: 'DELEGATE_FALLBACK', newDelegate: 'CPU' });
        gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
          baseOptions: {
            modelAssetBuffer: new Uint8Array(modelBuffer),
            delegate: 'CPU',
          },
          minHandDetectionConfidence: currentOptions.minHandDetectionConfidence,
          minHandPresenceConfidence: currentOptions.minHandPresenceConfidence,
          minTrackingConfidence: currentOptions.minTrackingConfidence,
          numHands: currentOptions.numHands,
          runningMode: currentOptions.runningMode,
        });
      } else {
        throw finalError;
      }
    }
    self.postMessage({ type: 'INIT_DONE' });

  } catch (error: any) {
    console.error("Gesture Recognizer Worker Init Error:", error);
    throw error;
  } finally {
    isInitializing = false;
  }
}
