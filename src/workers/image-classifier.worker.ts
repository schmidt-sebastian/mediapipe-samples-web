import {
  ImageClassifier,
  FilesetResolver
} from '@mediapipe/tasks-vision';
import { loadWasmModule } from '../utils/wasm-loader';

const basePath = import.meta.env.BASE_URL;

let imageClassifier: ImageClassifier | undefined;
let isInitializing = false;

self.onmessage = async (event) => {
  const { type } = event.data;

  if (type === 'INIT') {
    await initClassifier(event.data);
  } else if (type === 'CLASSIFY_IMAGE') {
    await classifyImage(event.data);
  } else if (type === 'CLASSIFY_VIDEO') {
    await classifyVideo(event.data);
  } else if (type === 'SET_OPTIONS') {
    await setOptions(event.data);
  } else if (type === 'CLEANUP') {
    imageClassifier?.close();
    imageClassifier = undefined;
  }
};

async function initClassifier(data: any) {
  if (isInitializing) return;
  isInitializing = true;

  try {
    const wasmPath = new URL(`${basePath}wasm`, self.location.origin).href;
    const wasmLoaderUrl = `${wasmPath}/vision_wasm_internal.js`;

    await loadWasmModule(wasmLoaderUrl, ';ModuleFactory;');

    const vision = await FilesetResolver.forVisionTasks(wasmPath);

    imageClassifier = await ImageClassifier.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: data.modelAssetPath,
        delegate: data.delegate || 'GPU',
      },
      runningMode: data.runningMode || 'IMAGE',
      maxResults: data.maxResults || 5,
      scoreThreshold: data.scoreThreshold || 0.0,
    });

    self.postMessage({ type: 'INIT_DONE' });

  } catch (error: any) {
    console.error("Image Classifier Init Error:", error);
    self.postMessage({ type: 'ERROR', error: error.message });
  } finally {
    isInitializing = false;
  }
}

async function classifyImage(data: any) {
  if (!imageClassifier) return;
  try {
    const result = imageClassifier.classify(data.bitmap);
    self.postMessage({
      type: 'CLASSIFY_RESULT',
      result,
      mode: 'IMAGE',
      inferenceTime: performance.now() - data.timestampMs
    });
  } catch (error: any) {
    self.postMessage({ type: 'ERROR', error: error.message });
  }
}

async function classifyVideo(data: any) {
  if (!imageClassifier) return;
  try {
    const startTimeMs = performance.now();
    const result = imageClassifier.classifyForVideo(data.bitmap, data.timestampMs);
    const inferenceTime = performance.now() - startTimeMs;

    self.postMessage({
      type: 'CLASSIFY_RESULT',
      result,
      mode: 'VIDEO',
      inferenceTime
    });
  } catch (error: any) {
    console.warn("Video classification error", error);
  }
}

async function setOptions(data: any) {
  if (!imageClassifier) return;
  try {
    await imageClassifier.setOptions(data);
  } catch (error: any) {
    console.error("Set Options Error", error);
  }
}
