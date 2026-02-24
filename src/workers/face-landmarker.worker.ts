import { loadModel } from '../utils/model-loader';
import {
  FaceLandmarker,
  FilesetResolver,
  FaceLandmarkerResult
} from '@mediapipe/tasks-vision';

import { loadWasmModule } from '../utils/wasm-loader';

// MediaPipe Emscripten fallback
// @ts-ignore
self.createMediapipeTasksVisionModule = self.createMediapipeTasksVisionModule || undefined;

let faceLandmarker: FaceLandmarker | undefined = undefined;
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
      const { modelAssetPath, delegate, runningMode, numFaces, minFaceDetectionConfidence, minFacePresenceConfidence, minTrackingConfidence, outputFaceBlendshapes, outputFacialTransformationMatrixes, baseUrl } = event.data;
      basePath = baseUrl || '/';
      currentOptions = { modelAssetPath, delegate, runningMode, numFaces, minFaceDetectionConfidence, minFacePresenceConfidence, minTrackingConfidence, outputFaceBlendshapes, outputFacialTransformationMatrixes };
      await initLandmarker();
    } else if (type === 'SET_OPTIONS') {
      // Update options if changed
      let needsUpdate = false;
      if ('numFaces' in event.data) { currentOptions.numFaces = event.data.numFaces; needsUpdate = true; }
      if ('minFaceDetectionConfidence' in event.data) { currentOptions.minFaceDetectionConfidence = event.data.minFaceDetectionConfidence; needsUpdate = true; }
      if ('minFacePresenceConfidence' in event.data) { currentOptions.minFacePresenceConfidence = event.data.minFacePresenceConfidence; needsUpdate = true; }
      if ('minTrackingConfidence' in event.data) { currentOptions.minTrackingConfidence = event.data.minTrackingConfidence; needsUpdate = true; }
      if ('runningMode' in event.data) { currentOptions.runningMode = event.data.runningMode; needsUpdate = true; }

      if (faceLandmarker && needsUpdate) {
        await faceLandmarker.setOptions({
          numFaces: currentOptions.numFaces,
          minFaceDetectionConfidence: currentOptions.minFaceDetectionConfidence,
          minFacePresenceConfidence: currentOptions.minFacePresenceConfidence,
          minTrackingConfidence: currentOptions.minTrackingConfidence,
          runningMode: currentOptions.runningMode
        });
        self.postMessage({ type: 'OPTIONS_UPDATED' });
      }
    } else if (type === 'DETECT_IMAGE' || type === 'DETECT_VIDEO') {
      const { bitmap, timestampMs } = event.data;
      if (!faceLandmarker) {
        console.warn('FaceLandmarker not initialized yet.');
        bitmap.close();
        self.postMessage({ type: 'DETECT_ERROR', error: 'Not initialized' });
        return;
      }

      const requiredMode = type === 'DETECT_IMAGE' ? 'IMAGE' : 'VIDEO';
      if (currentOptions.runningMode !== requiredMode) {
        currentOptions.runningMode = requiredMode;
        await faceLandmarker.setOptions({ runningMode: requiredMode });
      }

      const startTimeMs = performance.now();
      let result: FaceLandmarkerResult;

      try {
        if (requiredMode === 'VIDEO') {
          result = faceLandmarker.detectForVideo(bitmap, timestampMs);
        } else {
          result = faceLandmarker.detect(bitmap);
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
      if (faceLandmarker) {
        faceLandmarker.close();
        faceLandmarker = undefined;
      }
      self.postMessage({ type: 'CLEANUP_DONE' });
    }
  } catch (error: any) {
    console.error("Face Landmarker Worker Error:", error);
    self.postMessage({ type: 'ERROR', error: error?.message || String(error) });
  } finally {
    isProcessing = false;
  }
};

async function initLandmarker() {
  if (isInitializing) return;
  isInitializing = true;

  try {
    if (faceLandmarker) {
      faceLandmarker.close();
      faceLandmarker = undefined;
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
      numFaces: currentOptions.numFaces || 1,
      minFaceDetectionConfidence: currentOptions.minFaceDetectionConfidence || 0.5,
      minFacePresenceConfidence: currentOptions.minFacePresenceConfidence || 0.5,
      minTrackingConfidence: currentOptions.minTrackingConfidence || 0.5,
      outputFaceBlendshapes: currentOptions.outputFaceBlendshapes || false,
      outputFacialTransformationMatrixes: currentOptions.outputFacialTransformationMatrixes || false,
      runningMode: currentOptions.runningMode,
    };

    try {
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, options);
    } catch (finalError) {
      console.error('FaceLandmarker initialization failed:', finalError);
      if (currentOptions.delegate === 'GPU') {
        console.warn('GPU init failed, falling back to CPU', finalError);
        self.postMessage({ type: 'DELEGATE_FALLBACK', newDelegate: 'CPU' });
        options.baseOptions.delegate = 'CPU';
        faceLandmarker = await FaceLandmarker.createFromOptions(vision, options);
      } else {
        throw finalError;
      }
    }
    self.postMessage({ type: 'INIT_DONE' });

  } catch (error: any) {
    console.error("Face Landmarker Worker Init Error:", error);
    throw error;
  } finally {
    isInitializing = false;
  }
}
