import {
  HolisticLandmarker,
  FilesetResolver
} from '@mediapipe/tasks-vision';
import { loadWasmModule } from '../utils/wasm-loader';

const basePath = import.meta.env.BASE_URL;

let holisticLandmarker: HolisticLandmarker | undefined;
let isInitializing = false;

self.onmessage = async (event) => {
  const { type } = event.data;

  if (type === 'INIT') {
    await initDetector(event.data);
  } else if (type === 'DETECT_IMAGE') {
    await detectImage(event.data);
  } else if (type === 'DETECT_VIDEO') {
    await detectVideo(event.data);
  } else if (type === 'SET_OPTIONS') {
    await setOptions(event.data);
  } else if (type === 'CLEANUP') {
    holisticLandmarker?.close();
    holisticLandmarker = undefined;
  }
};

async function initDetector(data: any) {
  if (isInitializing) return;
  isInitializing = true;

  try {
    const wasmPath = new URL(`${basePath}wasm`, self.location.origin).href;
    const wasmLoaderUrl = `${wasmPath}/vision_wasm_internal.js`;

    await loadWasmModule(wasmLoaderUrl, ';ModuleFactory;');

    const vision = await FilesetResolver.forVisionTasks(wasmPath);

    holisticLandmarker = await HolisticLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: data.modelAssetPath,
        delegate: data.delegate || 'GPU',
      },
      runningMode: data.runningMode || 'IMAGE',
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minFaceSuppressionThreshold: 0.5,
      minHandLandmarksConfidence: 0.5,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minPoseSuppressionThreshold: 0.5
    });

    self.postMessage({ type: 'INIT_DONE' });

  } catch (error: any) {
    console.error("Holistic Landmarker Init Error:", error);
    self.postMessage({ type: 'ERROR', error: error.message });
  } finally {
    isInitializing = false;
  }
}

async function detectImage(data: any) {
  if (!holisticLandmarker) return;
  try {
    const result = holisticLandmarker.detect(data.bitmap);
    self.postMessage({
      type: 'DETECT_RESULT',
      result,
      mode: 'IMAGE',
      inferenceTime: performance.now() - data.timestampMs
    });
  } catch (error: any) {
    self.postMessage({ type: 'ERROR', error: error.message });
  }
}

async function detectVideo(data: any) {
  if (!holisticLandmarker) return;
  try {
    const result = holisticLandmarker.detectForVideo(data.bitmap, data.timestampMs);
    self.postMessage({
      type: 'DETECT_RESULT',
      result,
      mode: 'VIDEO',
      inferenceTime: performance.now() - data.timestampMs
    });
  } catch (error: any) {
    // Ignore frame errors in video loop
    console.warn("Video detection error", error);
  }
}

async function setOptions(data: any) {
  if (!holisticLandmarker) return;
  try {
    await holisticLandmarker.setOptions(data);
    if (data.delegate) {
      // Delegate change requires recreation usually, but if just options:
      // HolisticLandmarker might not support changing delegate via setOptions dynamically without recreation.
      // We'll assume simple options for now.
    }
  } catch (error: any) {
    console.error("Set Options Error", error);
  }
}
