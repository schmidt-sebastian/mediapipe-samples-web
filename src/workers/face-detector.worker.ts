import {
  FaceDetector,
  FilesetResolver,
  Detection,
} from '@mediapipe/tasks-vision';

// MediaPipe's worker runtime expects self.import to exist in non-module workers.
if (typeof (self as any).import !== 'function') {
  (self as any).import = (url: string) => import(/* @vite-ignore */ url);
}

let faceDetector: FaceDetector | undefined = undefined;
let isInitializing = false;
let currentOptions: any = {};
let basePath = '/';

let isProcessing = false;
let lastVideoTimestampMs = -1;

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
          const safeTimestampMs = timestampMs > lastVideoTimestampMs ? timestampMs : lastVideoTimestampMs + 1;
          lastVideoTimestampMs = safeTimestampMs;
          detections = faceDetector.detectForVideo(bitmap, safeTimestampMs);
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
      lastVideoTimestampMs = -1;
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
  if (!response.ok) {
    throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
  }
  const reader = response.body?.getReader();
  const contentLength = +(response.headers.get('Content-Length') || '0');

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
    await ensureVisionWasmFactory(wasmPath);
    const vision = await FilesetResolver.forVisionTasks(wasmPath);
    const modelBuffer = await loadModel(currentOptions.modelAssetPath);

    let delegate: 'CPU' | 'GPU' = currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU';
    if (delegate === 'GPU') {
      console.warn('[Worker] GPU delegate requested, forcing CPU in worker context.');
      delegate = 'CPU';
      self.postMessage({ type: 'DELEGATE_FALLBACK', newDelegate: 'CPU' });
    }

    faceDetector = await FaceDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate,
      },
      minDetectionConfidence: currentOptions.minDetectionConfidence,
      minSuppressionThreshold: currentOptions.minSuppressionThreshold,
      runningMode: currentOptions.runningMode,
    });
    self.postMessage({ type: 'INIT_DONE' });

  } catch (error: any) {
    console.error("Face Detection Worker Init Error:", error);
    throw error;
  } finally {
    isInitializing = false;
  }
}

async function ensureVisionWasmFactory(wasmPath: string) {
  if (typeof (self as any).ModuleFactory === 'function') {
    return;
  }

  const wasmLoaderUrl = `${wasmPath}/vision_wasm_internal.js`;
  const response = await fetch(wasmLoaderUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch WASM loader: ${response.status} ${response.statusText}`);
  }

  const loaderCode = await response.text();
  const factory = (0, eval)(`${loaderCode}; ModuleFactory;`);
  if (typeof factory !== 'function') {
    throw new Error('Failed to initialize vision WASM ModuleFactory.');
  }
  (self as any).ModuleFactory = factory;
}
