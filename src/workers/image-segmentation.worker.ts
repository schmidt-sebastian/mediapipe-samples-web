import {
  ImageSegmenter,
  FilesetResolver,
  ImageSegmenterResult,
  DrawingUtils
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
let drawingUtils: DrawingUtils | undefined = undefined;
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

      const callback = (result: ImageSegmenterResult) => {
        try {
          console.log('Worker callback fired!');
          const inferenceTime = performance.now() - startTimeMs;
          bitmap.close();

          let maskBitmap: ImageBitmap | null = null;
          let width = 0;
          let height = 0;

          if (result.categoryMask && renderCanvas && drawingUtils && colors) {
            width = result.categoryMask.width;
            height = result.categoryMask.height;

            // Resize offscreen canvas to match mask bounds natively
            if (renderCanvas.width !== width || renderCanvas.height !== height) {
              renderCanvas.width = width;
              renderCanvas.height = height;
            }

            // Let the GPU bind the Texture and write directly onto our Canvas Buffer
            drawingUtils.drawCategoryMask(
              result.categoryMask,
              colors,
              [0, 0, 0, 0]
            );

            // Zero-copy extract the final Colored bitmap to jump the thread wall
            maskBitmap = renderCanvas.transferToImageBitmap();

            // Free WASM WebGL memory immediately
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

    // Provide an Offscreen Canvas for DrawingUtils to inherently map the GPU texture without stalling
    renderCanvas = new OffscreenCanvas(256, 256);
    drawingUtils = new DrawingUtils(renderCanvas.getContext('webgl2') as WebGL2RenderingContext);

    try {
      imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: currentOptions.modelAssetPath,
          delegate: currentOptions.delegate,
        },
        outputCategoryMask: true,
        outputConfidenceMasks: false,
        runningMode: currentOptions.runningMode,
        canvas: renderCanvas
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
