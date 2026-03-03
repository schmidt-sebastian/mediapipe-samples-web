import { ImageClassifierResult } from '@mediapipe/tasks-vision';
import { ClassificationResult, ClassificationItem } from '../components/classification-result';
import { ModelSelector } from '../components/model-selector';
import { MediaManager } from '../components/media-manager';
// @ts-ignore
import template from '../templates/image-classifier.html?raw';
// @ts-ignore
import ImageClassifierWorker from '../workers/image-classifier.worker.ts?worker';

let worker: Worker | undefined;
let mediaManager: MediaManager;

// Options
let currentModel = 'efficientnet_lite0';
let modelSelector: ModelSelector;
let currentDelegate: 'CPU' | 'GPU' = 'GPU';
let maxResults = 3;
let scoreThreshold = 0.0;

const models: Record<string, string> = {
  'efficientnet_lite0': 'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite',
  'efficientnet_lite2': 'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite2/float32/1/efficientnet_lite2.tflite'
};

let classificationResultUI: ClassificationResult;

export async function setupImageClassifier(container: HTMLElement) {
  container.innerHTML = template;

  classificationResultUI = new ClassificationResult('classification-results');

  initWorker();
  setupUI();

  mediaManager = new MediaManager({
    onModeChange: (mode) => {
      worker?.postMessage({ type: 'SET_OPTIONS', runningMode: mode });
    },
    onImageUpload: (image) => {
      classifyImage(image);
    },
    onWebcamFrame: (video, timestampMs) => {
      if (!worker || !mediaManager.isWorkerReady) {
        mediaManager.requestNextFrame();
        return;
      }
      try {
        let bitmapPromise: Promise<ImageBitmap>;
        if (navigator.webdriver) {
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = video.videoWidth || 640;
          tempCanvas.height = video.videoHeight || 480;
          const ctx = tempCanvas.getContext('2d', { willReadFrequently: true });
          ctx?.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
          bitmapPromise = window.createImageBitmap(tempCanvas);
        } else {
          bitmapPromise = window.createImageBitmap(video);
        }

        bitmapPromise.then(bitmap => {
          worker?.postMessage({
            type: 'CLASSIFY_VIDEO',
            bitmap: bitmap,
            timestampMs: timestampMs
          }, [bitmap]);
        }).catch(err => {
          console.error("Failed to create ImageBitmap", err);
          mediaManager.requestNextFrame();
        });
      } catch (e) {
        console.error("Failed to process webcam frame", e);
        mediaManager.requestNextFrame();
      }
    }
  });

  await initializeClassifier();
}

function initWorker() {
  if (!worker) {
    worker = new ImageClassifierWorker();
  }
  if (worker) {
    worker.onmessage = handleWorkerMessage;
  }
}

function handleWorkerMessage(event: MessageEvent) {
  const { type } = event.data;

  switch (type) {
    case 'LOAD_PROGRESS':
      const { progress, loaded, total } = event.data;
      if (progress !== undefined) {
        modelSelector?.showProgress(progress * 100, 100);
        if (progress >= 1) {
          setTimeout(() => modelSelector?.hideProgress(), 500);
        }
      } else if (loaded !== undefined && total !== undefined) {
        modelSelector?.showProgress(loaded, total);
        if (loaded >= total) {
          setTimeout(() => modelSelector?.hideProgress(), 500);
        }
      }
      break;

    case 'INIT_DONE':
      modelSelector?.hideProgress();
      document.querySelector('.viewport')?.classList.remove('loading-model');
      mediaManager?.setWorkerReady(true);
      updateStatus('Ready');

      if (mediaManager?.getRunningMode() === 'VIDEO') {
        mediaManager.enableCam();
      } else {
        mediaManager.triggerImageAction();
      }
      break;

    case 'CLASSIFY_RESULT':
      const { mode, result, inferenceTime } = event.data;
      updateStatus(`Done in ${Math.round(inferenceTime)}ms`);
      updateInferenceTime(inferenceTime);

      displayResult(result);
      if (mode === 'VIDEO') {
        mediaManager?.requestNextFrame();
      }
      break;

    case 'ERROR':
      console.error('Worker error:', event.data.error);
      updateStatus(`Error: ${event.data.error}`);
      break;
  }
}

async function initializeClassifier() {
  mediaManager?.setWorkerReady(false);
  document.querySelector('.viewport')?.classList.add('loading-model');
  updateStatus('Loading Model...');

  // @ts-ignore
  const baseUrl = import.meta.env.BASE_URL;
  let modelPath = models[currentModel];

  worker?.postMessage({
    type: 'INIT',
    modelAssetPath: modelPath,
    delegate: currentDelegate,
    runningMode: mediaManager?.getRunningMode() || 'IMAGE',
    maxResults,
    scoreThreshold,
    baseUrl
  });
}

function setupUI() {
  modelSelector = new ModelSelector(
    'model-selector-container',
    [
      { label: 'EfficientNet-Lite0', value: 'efficientnet_lite0', isDefault: true },
      { label: 'EfficientNet-Lite2', value: 'efficientnet_lite2' }
    ],
    async (selection) => {
      if (selection.type === 'standard') {
        currentModel = selection.value;
      } else if (selection.type === 'custom') {
        models['custom'] = URL.createObjectURL(selection.file);
        currentModel = 'custom';
      }
      await initializeClassifier();
    }
  );

  const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
  delegateSelect.addEventListener('change', async () => {
    currentDelegate = delegateSelect.value as 'GPU' | 'CPU';
    await initializeClassifier();
  });
  delegateSelect.value = currentDelegate;

  const maxResultsInput = document.getElementById('max-results') as HTMLInputElement;
  const maxResultsValue = document.getElementById('max-results-value')!;
  maxResultsInput.addEventListener('input', () => {
    maxResults = parseInt(maxResultsInput.value);
    maxResultsValue.innerText = maxResults.toString();
    worker?.postMessage({ type: 'SET_OPTIONS', maxResults });
    if (mediaManager?.getRunningMode() === 'IMAGE') reRunImageClassification();
  });

  const scoreThresholdInput = document.getElementById('score-threshold') as HTMLInputElement;
  const scoreThresholdValue = document.getElementById('score-threshold-value')!;
  scoreThresholdInput.addEventListener('input', () => {
    scoreThreshold = parseInt(scoreThresholdInput.value) / 100;
    scoreThresholdValue.innerText = `${parseInt(scoreThresholdInput.value)}%`;
    worker?.postMessage({ type: 'SET_OPTIONS', scoreThreshold });
    if (mediaManager?.getRunningMode() === 'IMAGE') reRunImageClassification();
  });
}

function reRunImageClassification() {
  mediaManager?.triggerImageAction();
}

async function classifyImage(image: HTMLImageElement) {
  if (!worker || !mediaManager?.isWorkerReady) return;

  const bitmap = await createImageBitmap(image);
  updateStatus(`Processing image...`);
  worker.postMessage({
    type: 'CLASSIFY_IMAGE',
    bitmap: bitmap,
    timestampMs: performance.now()
  }, [bitmap]);
}

function displayResult(result: ImageClassifierResult) {
  if (!classificationResultUI) return;

  if (result.classifications && result.classifications.length > 0) {
    const categories = result.classifications[0].categories;
    const items: ClassificationItem[] = categories.map(c => ({
      label: c.categoryName,
      score: c.score
    }));
    classificationResultUI.updateResults(items);
  } else {
    classificationResultUI.clear();
  }
}

function updateStatus(msg: string) {
  const el = document.getElementById('status-message');
  if (el) el.innerText = msg;
}

function updateInferenceTime(time: number) {
  const el = document.getElementById('inference-time');
  if (el) el.innerText = `Inference Time: ${time.toFixed(2)} ms`;
}

export function cleanupImageClassifier() {
  if (mediaManager) mediaManager.cleanup();
  if (worker) {
    worker.postMessage({ type: 'CLEANUP' });
    worker.terminate();
    worker = undefined;
  }
}

