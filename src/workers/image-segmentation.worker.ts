import {
  ImageSegmenter,
  FilesetResolver,
  ImageSegmenterResult
} from '@mediapipe/tasks-vision';

// @ts-ignore
if (typeof self.import === 'undefined') {
  // @ts-ignore
  self.import = (url) => import(/* @vite-ignore */ url);
}

// MediaPipe Emscripten fallback
// @ts-ignore
self.createMediapipeTasksVisionModule = self.createMediapipeTasksVisionModule || undefined;

let imageSegmenter: ImageSegmenter | undefined = undefined;
let renderCanvas: OffscreenCanvas | undefined = undefined;
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
      const { modelAssetPath, delegate, runningMode, baseUrl } = event.data;
      basePath = baseUrl || '/';
      currentOptions = { modelAssetPath, delegate, runningMode };
      await initSegmenter();
      const labels = imageSegmenter ? imageSegmenter.getLabels() : [];
      self.postMessage({ type: 'INIT_DONE', labels });
    } else if (type === 'SET_OPTIONS') {
      if ('runningMode' in event.data) currentOptions.runningMode = event.data.runningMode;
      if (imageSegmenter) {
        await imageSegmenter.setOptions({
          runningMode: currentOptions.runningMode
        });
        self.postMessage({ type: 'OPTIONS_UPDATED' });
      }
    } else if (type === 'SEGMENT_IMAGE' || type === 'SEGMENT_VIDEO') {
      const { bitmap, timestampMs, colors } = event.data;
      if (!imageSegmenter) {
        console.warn('ImageSegmenter not initialized yet.');
        bitmap.close();
        self.postMessage({ type: 'SEGMENT_ERROR', error: 'Not initialized' });
        return;
      }

      const requiredMode = type === 'SEGMENT_IMAGE' ? 'IMAGE' : 'VIDEO';
      if (currentOptions.runningMode !== requiredMode) {
        currentOptions.runningMode = requiredMode;
        await imageSegmenter.setOptions({ runningMode: requiredMode });
      }

      const startTimeMs = performance.now();

      const callback = async (result: ImageSegmenterResult) => {
        try {
          const inferenceTime = performance.now() - startTimeMs;
          bitmap.close();

          let maskBitmap: ImageBitmap | null = null;
          let width = 0;
          let height = 0;

          if (result.categoryMask && colors) {
            width = result.categoryMask.width;
            height = result.categoryMask.height;

            // Recreate offscreen canvas ensures clean state and avoids artifacts from transferToImageBitmap
            if (!renderCanvas || renderCanvas.width !== width || renderCanvas.height !== height) {
              renderCanvas = new OffscreenCanvas(width, height);
            }

            const ctx = renderCanvas.getContext('2d') as OffscreenCanvasRenderingContext2D;
            if (ctx) {
              const maskData = result.categoryMask.getAsUint8Array();
              const imageData = ctx.createImageData(width, height);
              const data = imageData.data;

              for (let i = 0; i < maskData.length; i++) {
                const categoryIndex = maskData[i];
                const color = colors[categoryIndex] || [0, 0, 0, 0];
                const offset = i * 4;
                data[offset] = color[0];
                data[offset + 1] = color[1];
                data[offset + 2] = color[2];
                data[offset + 3] = color[3];
              }
              ctx.putImageData(imageData, 0, 0);

              // Use transferToImageBitmap for 2D as it's efficient and safe
              maskBitmap = renderCanvas.transferToImageBitmap();
            } else {
              console.error('Failed to get 2D context');
            }

            // Free WASM memory
            result.categoryMask.close();
          }

          (self as any).postMessage({
            type: 'SEGMENT_RESULT',
            mode: requiredMode,
            maskBitmap: maskBitmap,
            width: width,
            height: height,
            inferenceTime: inferenceTime
          }, maskBitmap ? [maskBitmap] : []);
        } catch (e: any) {
          console.error("Worker callback error:", e);
          (self as any).postMessage({ type: 'SEGMENT_ERROR', error: e.message || 'Worker callback error', mode: requiredMode });
        }
      };

      try {
        if (requiredMode === 'VIDEO') {
          imageSegmenter.segmentForVideo(bitmap, timestampMs, callback);
        } else {
          const result = imageSegmenter.segment(bitmap);
          callback(result);
        }
      } catch (e: any) {
        console.error("Worker segment error:", e);
        bitmap.close();
        self.postMessage({ type: 'SEGMENT_ERROR', error: e.message || 'Segmentation failed' });
        return;
      }
    } else if (type === 'CLEANUP') {
      if (imageSegmenter) {
        imageSegmenter.close();
        imageSegmenter = undefined;
      }
      self.postMessage({ type: 'CLEANUP_DONE' });
    }
  } catch (error: any) {
    console.error("Image Segmentation Worker Error:", error);
    self.postMessage({ type: 'ERROR', error: error?.message || String(error) });
  } finally {
    isProcessing = false;
  }
};

async function loadModel(path: string) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load model from ${path}: ${response.status} ${response.statusText}`);
  }
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

async function initSegmenter() {
  if (isInitializing) return;
  isInitializing = true;

  try {
    if (imageSegmenter) {
      imageSegmenter.close();
      imageSegmenter = undefined;
    }

    const wasmPath = new URL(`${basePath}wasm`, self.location.origin).href;

    // WORKAROUND: Vite + MediaPipe module workers fail to inject ModuleFactory via importScripts.
    const wasmLoaderUrl = `${wasmPath}/vision_wasm_internal.js`;
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

    // Manually fetch model to report progress
    const modelBuffer = await loadModel(currentOptions.modelAssetPath);

    // Override options to use buffer
    const options = {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: currentOptions.delegate,
      },
      outputCategoryMask: true,
      outputConfidenceMasks: false,
      runningMode: currentOptions.runningMode
    };

    try {
      imageSegmenter = await ImageSegmenter.createFromOptions(vision, options);
    } catch (error) {
      // Fallback to CPU if GPU fails
      console.warn('GPU init failed, falling back to CPU', error);
      if (currentOptions.delegate === 'GPU') {
        self.postMessage({ type: 'DELEGATE_FALLBACK', newDelegate: 'CPU' });
        options.baseOptions.delegate = 'CPU';
        imageSegmenter = await ImageSegmenter.createFromOptions(vision, options);
      } else {
        throw error;
      }
    }

  } catch (error: any) {
    console.error("Image Segmentation Worker Init Error:", error);
    throw error;
  } finally {
    isInitializing = false;
  }
}
