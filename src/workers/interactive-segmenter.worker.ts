/// <reference types="vite/client" />
import {
  InteractiveSegmenter,
  FilesetResolver
} from '@mediapipe/tasks-vision';
import { loadWasmModule } from '../utils/wasm-loader';

const basePath = import.meta.env.BASE_URL;

let interactiveSegmenter: InteractiveSegmenter | undefined;
let isInitializing = false;

self.onmessage = async (event) => {
  const { type } = event.data;

  if (type === 'INIT') {
    await initSegmenter(event.data);
  } else if (type === 'SEGMENT') {
    await segment(event.data);
  }
};

async function initSegmenter(data: any) {
  if (isInitializing) return;
  isInitializing = true;

  try {
    const wasmPath = new URL(`${basePath}wasm`, self.location.origin).href;
    const wasmLoaderUrl = `${wasmPath}/vision_wasm_internal.js`;

    await loadWasmModule(wasmLoaderUrl, ';ModuleFactory;');

    const vision = await FilesetResolver.forVisionTasks(wasmPath);

    interactiveSegmenter = await InteractiveSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: data.modelAssetPath,
        delegate: data.delegate || 'GPU',
      },
      outputCategoryMask: true,
      outputConfidenceMasks: false
    });

    self.postMessage({ type: 'INIT_DONE' });

  } catch (error: any) {
    console.error("Interactive Segmenter Init Error:", error);
    self.postMessage({ type: 'ERROR', error: error.message });
    // Fallback or retry logic could go here
  } finally {
    isInitializing = false;
  }
}

async function segment(data: any) {
  if (!interactiveSegmenter) {
    self.postMessage({ type: 'ERROR', error: 'Segmenter not initialized' });
    return;
  }

  try {
    const { bitmap, pt } = data; // pt is { x: 0..1, y: 0..1 } normalized? or relative?
    // InteractiveSegmenter expects RegionOfInterest.
    // We can use keypoint for click.

    // According to docs, we pass the image and the ROI.
    // The ROI for click is usually a keypoint.

    const timestampMs = performance.now();

    // Convert normalized or pixel coordinates?
    // InteractiveSegmenter.segment(image, roi, rotation)
    // ROI: { keypoint: { x, y } } - x, y are normalized [0, 1]???
    // Docs say: "The RegionOfInterest to segment. This is required for interactive segmentation."

    const result = interactiveSegmenter.segment(bitmap, {
      keypoint: { x: pt.x, y: pt.y }
    });

    // Loop over result.categoryMasks -> convert to ImageBitmap or Uint8Array to send back?
    // result.categoryMask matches the input image size.
    // We can send the Float32Array or Uint8Array back.

    // categoryMask is MPMask.
    const categoryMask = result.categoryMask;
    let maskData: Uint8Array | Float32Array | null = null;
    let width = 0;
    let height = 0;

    if (categoryMask) {
      width = categoryMask.width;
      height = categoryMask.height;
      // We need to export it. 
      // As Uint8Array?
      maskData = categoryMask.getAsUint8Array();
      // Or getAsFloat32Array() if confidence? Category is usually uint8.
    }

    const transfer = maskData ? [maskData.buffer] : [];
    (self as any).postMessage({
      type: 'SEGMENT_RESULT',
      maskData,
      width,
      height,
      inferenceTime: performance.now() - timestampMs
    }, transfer);

  } catch (error: any) {
    console.error("Segmentation Error:", error);
    self.postMessage({ type: 'ERROR', error: error.message });
  }
}
