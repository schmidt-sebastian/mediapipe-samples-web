import {
  PoseLandmarker,
  FilesetResolver,
  PoseLandmarkerResult
} from '@mediapipe/tasks-vision';
import { loadModel } from '../utils/model-loader';
import { loadWasmModule } from '../utils/wasm-loader';

// MediaPipe Emscripten fallback
// @ts-ignore
self.createMediapipeTasksVisionModule = self.createMediapipeTasksVisionModule || undefined;

let poseLandmarker: PoseLandmarker | undefined = undefined;
let isInitializing = false;
let currentOptions: any = {};
let basePath = '/';

let isProcessing = false;

self.onmessage = async (event) => {
  const { type } = event.data;

  // Simple queue/lock to prevent calling WASM inference while setOptions is yielding the thread
  while (isProcessing) {
    await new Promise(resolve => setTimeout(() => resolve(true), 10));
  }
  isProcessing = true;

  try {
    if (type === 'INIT') {
      const { modelAssetPath, delegate, minPoseDetectionConfidence, minPosePresenceConfidence, minTrackingConfidence, numPoses, outputSegmentationMasks, runningMode, baseUrl } = event.data;
      basePath = baseUrl || '/';
      currentOptions = { modelAssetPath, delegate, minPoseDetectionConfidence, minPosePresenceConfidence, minTrackingConfidence, numPoses, outputSegmentationMasks, runningMode };
      await initDetector();
    } else if (type === 'SET_OPTIONS') {
      if ('minPoseDetectionConfidence' in event.data) currentOptions.minPoseDetectionConfidence = event.data.minPoseDetectionConfidence;
      if ('minPosePresenceConfidence' in event.data) currentOptions.minPosePresenceConfidence = event.data.minPosePresenceConfidence;
      if ('minTrackingConfidence' in event.data) currentOptions.minTrackingConfidence = event.data.minTrackingConfidence;
      if ('numPoses' in event.data) currentOptions.numPoses = event.data.numPoses;
      if ('outputSegmentationMasks' in event.data) currentOptions.outputSegmentationMasks = event.data.outputSegmentationMasks;
      if ('runningMode' in event.data) currentOptions.runningMode = event.data.runningMode;

      if (poseLandmarker) {
        await poseLandmarker.setOptions({
          minPoseDetectionConfidence: currentOptions.minPoseDetectionConfidence,
          minPosePresenceConfidence: currentOptions.minPosePresenceConfidence,
          minTrackingConfidence: currentOptions.minTrackingConfidence,
          numPoses: currentOptions.numPoses,
          outputSegmentationMasks: currentOptions.outputSegmentationMasks,
          runningMode: currentOptions.runningMode
        });
        self.postMessage({ type: 'OPTIONS_UPDATED' });
      }
    } else if (type === 'DETECT_IMAGE' || type === 'DETECT_VIDEO') {
      const { bitmap, timestampMs } = event.data;
      if (!poseLandmarker) {
        console.warn('PoseLandmarker not initialized yet.');
        bitmap.close();
        self.postMessage({ type: 'DETECT_ERROR', error: 'Not initialized' });
        return;
      }

      const requiredMode = type === 'DETECT_IMAGE' ? 'IMAGE' : 'VIDEO';
      if (currentOptions.runningMode !== requiredMode) {
        currentOptions.runningMode = requiredMode;
        await poseLandmarker.setOptions({ runningMode: requiredMode });
      }

      const startTimeMs = performance.now();
      let result: PoseLandmarkerResult;

      try {
        if (requiredMode === 'VIDEO') {
          result = poseLandmarker.detectForVideo(bitmap, timestampMs);
        } else {
          result = poseLandmarker.detect(bitmap);
        }
      } catch (e: any) {
        console.error("Worker detection error:", e);
        bitmap.close();
        self.postMessage({ type: 'DETECT_ERROR', error: e.message || 'Detection failed' });
        return;
      }

      const inferenceTime = performance.now() - startTimeMs;
      bitmap.close();

      const transferables: Transferable[] = [];
      if (result.segmentationMasks) {
        for (const mask of result.segmentationMasks) {
          // @ts-ignore
          if (mask.getAsImageBitmap) {
            // @ts-ignore
            transferables.push(mask.getAsImageBitmap());
          }
        }
      }

      // @ts-ignore
      (self as any).postMessage({
        type: 'DETECT_RESULT',
        mode: requiredMode,
        result: result,
        inferenceTime: inferenceTime
      }, transferables);
    } else if (type === 'CLEANUP') {
      if (poseLandmarker) {
        poseLandmarker.close();
        poseLandmarker = undefined;
      }
      self.postMessage({ type: 'CLEANUP_DONE' });
    }
  } catch (error: any) {
    console.error("Pose Landmarker Worker Error:", error);
    self.postMessage({ type: 'ERROR', error: error?.message || String(error) });
  } finally {
    isProcessing = false;
  }
};

async function initDetector() {
  if (isInitializing) return;
  isInitializing = true;

  try {
    if (poseLandmarker) {
      poseLandmarker.close();
      poseLandmarker = undefined;
    }

    const wasmPath = new URL(`${basePath}wasm`, self.location.origin).href;
    const wasmLoaderUrl = `${wasmPath}/vision_wasm_internal.js`;

    // Inject the loader
    await loadWasmModule(wasmLoaderUrl, ';ModuleFactory;');

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

    try {
      poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetBuffer: new Uint8Array(modelBuffer),
          delegate: currentOptions.delegate,
        },
        minPoseDetectionConfidence: currentOptions.minPoseDetectionConfidence,
        minPosePresenceConfidence: currentOptions.minPosePresenceConfidence,
        minTrackingConfidence: currentOptions.minTrackingConfidence,
        numPoses: currentOptions.numPoses,
        outputSegmentationMasks: currentOptions.outputSegmentationMasks,
        runningMode: currentOptions.runningMode,
      });
    } catch (finalError) {
      console.error('PoseLandmarker initialization failed:', finalError);
      if (currentOptions.delegate === 'GPU') {
        console.warn('GPU init failed, falling back to CPU', finalError);
        self.postMessage({ type: 'DELEGATE_FALLBACK', newDelegate: 'CPU' });
        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetBuffer: new Uint8Array(modelBuffer),
            delegate: 'CPU',
          },
          minPoseDetectionConfidence: currentOptions.minPoseDetectionConfidence,
          minPosePresenceConfidence: currentOptions.minPosePresenceConfidence,
          minTrackingConfidence: currentOptions.minTrackingConfidence,
          numPoses: currentOptions.numPoses,
          outputSegmentationMasks: currentOptions.outputSegmentationMasks,
          runningMode: currentOptions.runningMode,
        });
      } else {
        throw finalError;
      }
    }
    self.postMessage({ type: 'INIT_DONE' });

  } catch (error: any) {
    console.error("Pose Landmarker Worker Init Error:", error);
    throw error;
  } finally {
    isInitializing = false;
  }
}
