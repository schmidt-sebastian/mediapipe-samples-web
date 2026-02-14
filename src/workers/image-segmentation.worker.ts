import {
  ImageSegmenter,
  FilesetResolver,
  ImageSegmenterResult,
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
let isInitializing = false;
let currentOptions: any = {};
let basePath = '/';

self.onmessage = async (event) => {
  const { type } = event.data;

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
      const { bitmap, timestampMs } = event.data;
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

      const callback = (result: ImageSegmenterResult) => {
        const inferenceTime = performance.now() - startTimeMs;
        bitmap.close();

        // Convert the mask (either WebGLTexture or Float32Array/Uint8Array depending on outputType)
        // ImageSegmenter defaults to outputCategoryMask: true, outputConfidenceMasks: false
        // Currently, it returns WebGLTexture if built with GPU delegate natively,
        // but in worker we must ensure it transfers a raw buffer or ImageBitmap if possible!

        // Let's rely on standard 'categoryMask' buffer approach for now
        // since transferring WebGL textures across worker threads is not trivial unless using OffscreenCanvas

        let maskData = null;
        let width = 0;
        let height = 0;

        // ImageSegmenter supports `MPMask` which has `getAsUint8Array` or `getAsFloat32Array`. 
        // We need to transfer these buffers to avoid structured clone bottleneck!
        if (result.categoryMask) {
          // We extract the array view
          const floatArray = result.categoryMask.getAsFloat32Array();
          width = result.categoryMask.width;
          height = result.categoryMask.height;

          // Copy the data so we can transfer the raw buffer
          // getAsFloat32Array() returns a Float32Array backed by WASM heap.
          // We must copy it into a purely local buffer to transfer it.
          maskData = new Float32Array(floatArray).buffer;

          // Close the original mask to free WASM memory!
          result.categoryMask.close();
        }

        // @ts-ignore
        self.postMessage({
          type: 'SEGMENT_RESULT',
          mode: requiredMode,
          maskData: maskData,
          width: width,
          height: height,
          inferenceTime: inferenceTime
        }, maskData ? [maskData] : []);
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
  }
};

async function initSegmenter() {
  if (isInitializing) return;
  isInitializing = true;

  try {
    if (imageSegmenter) {
      imageSegmenter.close();
      imageSegmenter = undefined;
    }

    const wasmPath = new URL(`${basePath}wasm`, self.location.origin).href;

    const wasmLoaderUrl = `${wasmPath}/vision_wasm_internal.js`;
    const response = await fetch(wasmLoaderUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch WASM loader: ${response.status} ${response.statusText}`);
    }
    const loaderCode = await response.text();
    (0, eval)(loaderCode);

    const vision = await FilesetResolver.forVisionTasks(wasmPath);

    // We MUST use `outputCategoryMask` for normal segmentation (Uint8/Float32 masks)
    // GPU delegate can output WebGL textures, but we cannot pass them over postMessage directly
    // unless we render them to an OffscreenCanvas and call `transferToImageBitmap()`.
    // BUT! getAsFloat32Array() handles downloading the texture automatically in MediaPipe!
    try {
      imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: currentOptions.modelAssetPath,
          delegate: currentOptions.delegate,
        },
        outputCategoryMask: true,
        outputConfidenceMasks: false,
        runningMode: currentOptions.runningMode,
      });
    } catch (finalError) {
      console.error('ImageSegmenter initialization failed:', finalError);
      if (currentOptions.delegate === 'GPU') {
        currentOptions.delegate = 'CPU';
        self.postMessage({ type: 'DELEGATE_FALLBACK', newDelegate: 'CPU' });

        const vision = await FilesetResolver.forVisionTasks(wasmPath);
        imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: currentOptions.modelAssetPath,
            delegate: 'CPU',
          },
          outputCategoryMask: true,
          outputConfidenceMasks: false,
          runningMode: currentOptions.runningMode,
        });
      } else {
        throw finalError;
      }
    }
  } finally {
    isInitializing = false;
  }
}
