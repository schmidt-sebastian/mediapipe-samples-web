import {
  ObjectDetector,
  FilesetResolver,
  Detection,
} from '@mediapipe/tasks-vision';

// @ts-ignore
if (typeof self.import === 'undefined') {
  // @ts-ignore
  self.import = (url) => import(/* @vite-ignore */ url);
}

// MediaPipe Emscripten fallback
// @ts-ignore
self.createMediapipeTasksVisionModule = self.createMediapipeTasksVisionModule || undefined;

let objectDetector: ObjectDetector | undefined = undefined;
let isInitializing = false;
let currentOptions: any = {};
let basePath = '/';

let isProcessing = false;

self.onmessage = async (event) => {
  const { type } = event.data;

  // Simple queue/lock to prevent calling WASM inference while setOptions is yielding the thread
  while (isProcessing) {
    await new Promise(resolve => setTimeout(result => resolve(true), 10));
  }
  isProcessing = true;

  try {
    if (type === 'INIT') {
      const { modelAssetPath, delegate, scoreThreshold, maxResults, runningMode, baseUrl } = event.data;
      basePath = baseUrl || '/';
      currentOptions = { modelAssetPath, delegate, scoreThreshold, maxResults, runningMode };
      await initDetector();
      self.postMessage({ type: 'INIT_DONE' });
    } else if (type === 'SET_OPTIONS') {
      if ('scoreThreshold' in event.data) currentOptions.scoreThreshold = event.data.scoreThreshold;
      if ('maxResults' in event.data) currentOptions.maxResults = event.data.maxResults;
      if ('runningMode' in event.data) currentOptions.runningMode = event.data.runningMode;
      if (objectDetector) {
        await objectDetector.setOptions({
          scoreThreshold: currentOptions.scoreThreshold,
          maxResults: currentOptions.maxResults,
          runningMode: currentOptions.runningMode
        });
        self.postMessage({ type: 'OPTIONS_UPDATED' });
      }
    } else if (type === 'DETECT_IMAGE' || type === 'DETECT_VIDEO') {
      const { bitmap, timestampMs } = event.data;
      if (!objectDetector) {
        console.warn('ObjectDetector not initialized yet.');
        bitmap.close();
        self.postMessage({ type: 'DETECT_ERROR', error: 'Not initialized' });
        return;
      }

      const requiredMode = type === 'DETECT_IMAGE' ? 'IMAGE' : 'VIDEO';
      if (currentOptions.runningMode !== requiredMode) {
        currentOptions.runningMode = requiredMode;
        await objectDetector.setOptions({ runningMode: requiredMode });
      }

      const startTimeMs = performance.now();
      let detections: { detections: Detection[] };

      try {
        if (requiredMode === 'VIDEO') {
          detections = objectDetector.detectForVideo(bitmap, timestampMs);
        } else {
          detections = objectDetector.detect(bitmap);
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
      if (objectDetector) {
        objectDetector.close();
        objectDetector = undefined;
      }
      self.postMessage({ type: 'CLEANUP_DONE' });
    }
  } catch (error: any) {
    console.error("Object Detection Worker Error:", error);
    self.postMessage({ type: 'ERROR', error: error?.message || String(error) });
  } finally {
    isProcessing = false;
  }
};

async function initDetector() {
  if (isInitializing) return;
  isInitializing = true;

  try {
    if (objectDetector) {
      objectDetector.close();
      objectDetector = undefined;
    }

    const wasmPath = new URL(`${basePath}wasm`, self.location.origin).href;

    // WORKAROUND: Vite + MediaPipe module workers fail to inject ModuleFactory via importScripts.
    // We must manually fetch the WASM loader and eval it in the global scope.
    const wasmLoaderUrl = `${wasmPath}/vision_wasm_internal.js`;
    const response = await fetch(wasmLoaderUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch WASM loader: ${response.status} ${response.statusText}`);
    }
    const loaderCode = await response.text();
    // Use indirect eval to ensure global scope execution
    (0, eval)(loaderCode);

    const vision = await FilesetResolver.forVisionTasks(wasmPath);

    if (currentOptions.delegate === 'GPU') {
      console.warn('[Worker] GPU Delegate requested, but GPU delegate may be unstable in Web Worker depending on browser. Falling back to CPU if it crashes.');
    }

    try {
      objectDetector = await ObjectDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: currentOptions.modelAssetPath,
          delegate: currentOptions.delegate,
        },
        scoreThreshold: currentOptions.scoreThreshold,
        maxResults: currentOptions.maxResults,
        runningMode: currentOptions.runningMode,
      });
    } catch (finalError) {
      console.error('ObjectDetector initialization failed:', finalError);
      throw finalError;
    }
  } finally {
    isInitializing = false;
  }
}
