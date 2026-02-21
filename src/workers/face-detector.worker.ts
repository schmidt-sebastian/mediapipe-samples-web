import {
  FaceDetector,
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
    if (faceDetector) {
      faceDetector.close();
      faceDetector = undefined;
    }

    const wasmPath = new URL(`${basePath}wasm`, self.location.origin).href;
    const vision = await FilesetResolver.forVisionTasks(wasmPath);
    const modelBuffer = await loadModel(currentOptions.modelAssetPath);

    if (currentOptions.delegate === 'GPU') {
      console.warn('[Worker] GPU Delegate requested.');
    }

    try {
      faceDetector = await FaceDetector.createFromOptions(vision, {
        baseOptions: {
          modelAssetBuffer: new Uint8Array(modelBuffer),
          delegate: currentOptions.delegate,
        },
        minDetectionConfidence: currentOptions.minDetectionConfidence,
        minSuppressionThreshold: currentOptions.minSuppressionThreshold,
        runningMode: currentOptions.runningMode,
      });
    } catch (finalError) {
      console.error('FaceDetector initialization failed:', finalError);
      if (currentOptions.delegate === 'GPU') {
        console.warn('GPU init failed, falling back to CPU', finalError);
        self.postMessage({ type: 'DELEGATE_FALLBACK', newDelegate: 'CPU' });
        faceDetector = await FaceDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetBuffer: new Uint8Array(modelBuffer),
            delegate: 'CPU',
          },
          minDetectionConfidence: currentOptions.minDetectionConfidence,
          minSuppressionThreshold: currentOptions.minSuppressionThreshold,
          runningMode: currentOptions.runningMode,
        });
      } else {
        throw finalError;
      }
    }
    self.postMessage({ type: 'INIT_DONE' });

  } catch (error: any) {
    console.error("Face Detection Worker Init Error:", error);
    throw error;
  } finally {
    isInitializing = false;
  }
}
