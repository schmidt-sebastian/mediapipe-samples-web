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
  console.log("[Worker] Object Detection Worker v2.0 (Flatbuffer Fix)");
  const { type } = event.data;

  // Simple queue/lock to prevent calling WASM inference while setOptions is yielding the thread
  while (isProcessing) {
    await new Promise(resolve => setTimeout(() => resolve(true), 10));
  }
  isProcessing = true;

  try {
    if (type === 'INIT') {
      const { modelAssetPath, delegate, scoreThreshold, maxResults, runningMode, baseUrl } = event.data;
      basePath = baseUrl || '/';
      currentOptions = { modelAssetPath, delegate, scoreThreshold, maxResults, runningMode };
      await initDetector();
      // INIT_DONE is sent by initDetector now to ensure it's after the createFromOptions call
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
    const wasmLoaderUrl = `${wasmPath}/vision_wasm_internal.js`;
    // We can just rely on FilesetResolver if we trust it, but keeping the manual fetch for consistency if needed.
    // However, FilesetResolver.forVisionTasks(wasmPath) usually works if the loader is available.
    // The previous code did manual fetch and eval. Let's keep it to be safe.
    try {
      const response = await fetch(wasmLoaderUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch WASM loader: ${response.status} ${response.statusText}`);
      }
      const loaderCode = await response.text();
      (0, eval)(loaderCode);
    } catch (e) {
      console.error('Failed to manually load WASM loader:', e);
    }

    const vision = await FilesetResolver.forVisionTasks(wasmPath);
    // const vision = {
    //   wasmLoaderPath: `${wasmPath}/vision_wasm_internal.js`,
    //   wasmBinaryPath: `${wasmPath}/vision_wasm_internal.wasm`
    // };

    // Manual fetch to get buffer and report progress
    const modelBuffer = await loadModel(currentOptions.modelAssetPath);

    if (currentOptions.delegate === 'GPU') {
      console.warn('[Worker] GPU Delegate requested, but GPU delegate may be unstable in Web Worker depending on browser. Falling back to CPU if it crashes.');
    }

    try {
      objectDetector = await ObjectDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetBuffer: new Uint8Array(modelBuffer),
          delegate: currentOptions.delegate,
        },
        scoreThreshold: currentOptions.scoreThreshold,
        maxResults: currentOptions.maxResults,
        runningMode: currentOptions.runningMode,
      });
    } catch (finalError) {
      console.error('ObjectDetector initialization failed:', finalError);
      // Fallback to CPU if GPU fails
      if (currentOptions.delegate === 'GPU') {
        console.warn('GPU init failed, falling back to CPU', finalError);
        self.postMessage({ type: 'DELEGATE_FALLBACK', newDelegate: 'CPU' });
        objectDetector = await ObjectDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetBuffer: new Uint8Array(modelBuffer),
            delegate: 'CPU',
          },
          scoreThreshold: currentOptions.scoreThreshold,
          maxResults: currentOptions.maxResults,
          runningMode: currentOptions.runningMode,
        });
      } else {
        throw finalError;
      }
    }
    self.postMessage({ type: 'INIT_DONE' });

  } catch (error: any) {
    console.error("Object Detection Worker Init Error:", error);
    throw error;
  } finally {
    isInitializing = false;
  }
}
