import { loadModel } from '../utils/model-loader';
import {
  FaceDetector,
  FilesetResolver,
  Detection,
} from '@mediapipe/tasks-vision';

import { loadWasmModule } from '../utils/wasm-loader';

// MediaPipe Emscripten fallback
// @ts-ignore
self.createMediapipeTasksVisionModule = self.createMediapipeTasksVisionModule || undefined;

let faceDetector: FaceDetector | undefined = undefined;
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
      const { modelAssetPath, delegate, minDetectionConfidence, minSuppressionThreshold, runningMode, baseUrl } = event.data;
      basePath = baseUrl || '/';
      currentOptions = { modelAssetPath, delegate, minDetectionConfidence, minSuppressionThreshold, runningMode };
      await initDetector();
    } else if (type === 'SET_OPTIONS') {
      if ('minDetectionConfidence' in event.data) currentOptions.minDetectionConfidence = event.data.minDetectionConfidence;
      if ('minSuppressionThreshold' in event.data) currentOptions.minSuppressionThreshold = event.data.minSuppressionThreshold;
      if ('runningMode' in event.data) currentOptions.runningMode = event.data.runningMode;
      
      if (faceDetector) {
        await faceDetector.setOptions({
          minDetectionConfidence: currentOptions.minDetectionConfidence,
          minSuppressionThreshold: currentOptions.minSuppressionThreshold,
          runningMode: currentOptions.runningMode
        });
        self.postMessage({ type: 'OPTIONS_UPDATED' });
      }
    } else if (type === 'DETECT_IMAGE' || type === 'DETECT_VIDEO') {
      const { bitmap, timestampMs } = event.data;
      if (!faceDetector) {
        console.warn('FaceDetector not initialized yet.');
        bitmap.close();
        self.postMessage({ type: 'DETECT_ERROR', error: 'Not initialized' });
        return;
      }

      const requiredMode = type === 'DETECT_IMAGE' ? 'IMAGE' : 'VIDEO';
      if (currentOptions.runningMode !== requiredMode) {
        currentOptions.runningMode = requiredMode;
        await faceDetector.setOptions({ runningMode: requiredMode });
      }

      const startTimeMs = performance.now();
      let detections: { detections: Detection[] };

      try {
        if (requiredMode === 'VIDEO') {
          detections = faceDetector.detectForVideo(bitmap, timestampMs);
        } else {
          detections = faceDetector.detect(bitmap);
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
        detections: detections,
        inferenceTime: inferenceTime
      });
    } else if (type === 'CLEANUP') {
      if (faceDetector) {
        faceDetector.close();
        faceDetector = undefined;
      }
      self.postMessage({ type: 'CLEANUP_DONE' });
    }
  } catch (error: any) {
    console.error("Face Detection Worker Error:", error);
    self.postMessage({ type: 'ERROR', error: error?.message || String(error) });
  } finally {
    isProcessing = false;
  }
};


// ... (rest of imports)

// (Deleted local loadModel function)

async function initDetector() {
  if (isInitializing) return;
  isInitializing = true;

  try {
    if (faceDetector) {
      faceDetector.close();
      faceDetector = undefined;
    }

    const wasmPath = new URL(`${basePath}wasm`, self.location.origin).href;
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

    const options = {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: (currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU') as 'CPU' | 'GPU',
      },
      minDetectionConfidence: currentOptions.minDetectionConfidence,
      minSuppressionThreshold: currentOptions.minSuppressionThreshold,
      runningMode: currentOptions.runningMode
    };

    try {
      faceDetector = await FaceDetector.createFromOptions(vision, options);
    } catch (error) {
      console.warn('FaceDetector init failed, retrying with CPU fallback if needed', error);
      if (currentOptions.delegate === 'GPU') {
        self.postMessage({ type: 'DELEGATE_FALLBACK', newDelegate: 'CPU' });
        options.baseOptions.delegate = 'CPU';
        faceDetector = await FaceDetector.createFromOptions(vision, options);
      } else {
        throw error;
      }
    }

    self.postMessage({ type: 'INIT_DONE' });

  } catch (error: any) {
    console.error("Face Detection Worker Init Error:", error);
    self.postMessage({ type: 'ERROR', error: error.message });
  } finally {
    isInitializing = false;
  }
}


