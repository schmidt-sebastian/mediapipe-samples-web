import { GestureRecognizerResult, DrawingUtils, HandLandmarker } from '@mediapipe/tasks-vision';
import { ModelSelector } from '../components/model-selector';
import { ClassificationResult, ClassificationItem } from '../components/classification-result';
import { MediaManager } from '../components/media-manager';
// @ts-ignore
import template from '../templates/gesture-recognizer.html?raw';
// @ts-ignore
import GestureRecognizerWorker from '../workers/gesture-recognizer.worker.ts?worker';

let worker: Worker | undefined;
let mediaManager: MediaManager;
let canvasElement: HTMLCanvasElement;
let canvasCtx: CanvasRenderingContext2D;
let drawingUtils: DrawingUtils | undefined;

// Options
let currentModel = 'gesture_recognizer';
let modelSelector: ModelSelector;
let numHands = 2;
let minHandDetectionConfidence = 0.5;
let minHandPresenceConfidence = 0.5;
let minTrackingConfidence = 0.5;
let currentDelegate: 'CPU' | 'GPU' = 'GPU';

const models: Record<string, string> = {
  'gesture_recognizer': 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'
};

let classificationResultUI: ClassificationResult;

export async function setupGestureRecognizer(container: HTMLElement) {
  container.innerHTML = template;

  canvasElement = document.getElementById('output_canvas') as HTMLCanvasElement;
  canvasCtx = canvasElement.getContext('2d')!;
  drawingUtils = new DrawingUtils(canvasCtx);

  classificationResultUI = new ClassificationResult('classification-results'); 

  initWorker();
  setupUI();

  mediaManager = new MediaManager({
    onModeChange: (mode) => {
      worker?.postMessage({ type: 'SET_OPTIONS', runningMode: mode });
    },
    onImageUpload: (image) => {
      detectImage(image);
      const dropzoneContent = document.querySelector('.dropzone-content') as HTMLElement;
      const imagePreviewContainer = document.getElementById('image-preview-container');
      if (dropzoneContent) dropzoneContent.style.display = 'none';
      if (imagePreviewContainer) imagePreviewContainer.style.display = '';
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
            type: 'DETECT_VIDEO',
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

  await initializeRecognizer();
}

function initWorker() {
  if (!worker) {
    worker = new GestureRecognizerWorker();
  }
  if (worker) {
    worker.onmessage = handleWorkerMessage;
  }
}

function handleWorkerMessage(event: MessageEvent) {
  const { type } = event.data;

  switch (type) {
    case 'LOAD_PROGRESS':
      const { loaded, total } = event.data;
      modelSelector?.showProgress(loaded, total);
      if (loaded >= total) {
        setTimeout(() => {
          modelSelector?.hideProgress();
        }, 500);
      }
      break;

    case 'INIT_DONE':
      document.querySelector('.viewport')?.classList.remove('loading-model');
      mediaManager?.setWorkerReady(true);
      updateStatus('Ready');

      modelSelector?.hideProgress();

      if (mediaManager?.getRunningMode() === 'VIDEO') {
        mediaManager.enableCam();
      } else {
        mediaManager.triggerImageAction();
      }
      break;

    case 'DELEGATE_FALLBACK':
      console.warn('Worker fell back to CPU delegate.');
      currentDelegate = 'CPU';
      const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
      if (delegateSelect) delegateSelect.value = 'CPU';
      break;

    case 'DETECT_RESULT':
      const { mode, result, inferenceTime } = event.data;
      updateStatus(`Done in ${Math.round(inferenceTime)}ms`);
      updateInferenceTime(inferenceTime);

      if (mode === 'IMAGE') {
        displayImageResult(result);
      } else if (mode === 'VIDEO') {
        displayVideoResult(result);
        mediaManager?.requestNextFrame();
      }
      break;

    case 'ERROR':
    case 'DETECT_ERROR':
      console.error('Worker error:', event.data.error);
      updateStatus(`Error: ${event.data.error}`);
      break;
  }
}

async function initializeRecognizer() {
  mediaManager?.setWorkerReady(false);
  document.querySelector('.viewport')?.classList.add('loading-model');
  updateStatus('Loading Model...');

  // @ts-ignore
  const baseUrl = import.meta.env.BASE_URL;
  let modelPath = models[currentModel];
  if (currentModel === 'custom' && models['custom']) {
    modelPath = models['custom'];
  } else if (!modelPath.startsWith('http')) {
    modelPath = new URL(modelPath, new URL(baseUrl, window.location.origin)).href;
  }

  worker?.postMessage({
    type: 'INIT',
    modelAssetPath: modelPath,
    delegate: currentDelegate,
    minHandDetectionConfidence,
    minHandPresenceConfidence,
    minTrackingConfidence,
    numHands,
    runningMode: mediaManager?.getRunningMode() || 'IMAGE',
    baseUrl
  });
}

function setupUI() {
  modelSelector = new ModelSelector(
    'model-selector-container',
    [
      { label: 'Gesture Recognizer (Default)', value: 'gesture_recognizer', isDefault: true }
    ],
    async (selection) => {
      if (selection.type === 'standard') {
        currentModel = selection.value;
      } else if (selection.type === 'custom') {
        models['custom'] = URL.createObjectURL(selection.file);
        currentModel = 'custom';
      }
      await initializeRecognizer();
    }
  );

  // Sliders
  const setupSlider = (id: string, onChange: (val: number) => void) => {
    const input = document.getElementById(id) as HTMLInputElement;
    const valueDisplay = document.getElementById(`${id}-value`)!;
    if (input) {
      input.addEventListener('input', () => {
        const val = parseFloat(input.value);
        valueDisplay.innerText = val.toString();
        onChange(val);
      });
    }
  };

  setupSlider('num-hands', (val) => {
    numHands = val;
    worker?.postMessage({ type: 'SET_OPTIONS', numHands: numHands });
    if (mediaManager?.getRunningMode() === 'IMAGE') reRunImageDetection();
  });

  setupSlider('min-hand-detection-confidence', (val) => {
    minHandDetectionConfidence = val;
    worker?.postMessage({ type: 'SET_OPTIONS', minHandDetectionConfidence: minHandDetectionConfidence });
    if (mediaManager?.getRunningMode() === 'IMAGE') reRunImageDetection();
  });

  setupSlider('min-hand-presence-confidence', (val) => {
    minHandPresenceConfidence = val;
    worker?.postMessage({ type: 'SET_OPTIONS', minHandPresenceConfidence: minHandPresenceConfidence });
    if (mediaManager?.getRunningMode() === 'IMAGE') reRunImageDetection();
  });

  setupSlider('min-tracking-confidence', (val) => {
    minTrackingConfidence = val;
    worker?.postMessage({ type: 'SET_OPTIONS', minTrackingConfidence: minTrackingConfidence });
    if (mediaManager?.getRunningMode() === 'IMAGE') reRunImageDetection();
  });

  const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
  delegateSelect.addEventListener('change', async () => {
    currentDelegate = delegateSelect.value as 'GPU' | 'CPU';
    await initializeRecognizer();
  });
  delegateSelect.value = currentDelegate;
}

function reRunImageDetection() {
  mediaManager?.triggerImageAction();
}

async function detectImage(image: HTMLImageElement) {
  if (!worker || !mediaManager?.isWorkerReady) return;

  const bitmap = await createImageBitmap(image);
  updateStatus(`Processing image...`);
  worker.postMessage({
    type: 'DETECT_IMAGE',
    bitmap: bitmap,
    timestampMs: performance.now()
  }, [bitmap]);
}

function displayImageResult(result: GestureRecognizerResult) {
  const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
  const testImage = document.getElementById('test-image') as HTMLImageElement;
  const ctx = imageCanvas.getContext('2d')!;

  imageCanvas.width = testImage.naturalWidth;
  imageCanvas.height = testImage.naturalHeight;
  imageCanvas.style.width = '100%';
  imageCanvas.style.height = 'auto';

  ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);

  if (result.landmarks) {
    const drawingUtils = new DrawingUtils(ctx);
    for (const landmark of result.landmarks) {
      drawingUtils.drawConnectors(landmark, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
      drawingUtils.drawLandmarks(landmark, { color: "#FF0000", lineWidth: 2 });
    }
  }

  displayGestureText(result);
}

function displayVideoResult(result: GestureRecognizerResult) {
  const video = document.getElementById('webcam') as HTMLVideoElement;
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  if (result.landmarks) {
    if (!drawingUtils) drawingUtils = new DrawingUtils(canvasCtx);
    for (const landmark of result.landmarks) {
      drawingUtils.drawConnectors(landmark, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
      drawingUtils.drawLandmarks(landmark, { color: "#FF0000", lineWidth: 2 });
    }
  }
  canvasCtx.restore();

  displayGestureText(result);
}

function displayGestureText(result: GestureRecognizerResult) {
  if (!classificationResultUI) return;

  if (result.gestures && result.gestures.length > 0) {
    const items: ClassificationItem[] = [];
    result.gestures.forEach((gestures, index) => {
      const handedness = result.handedness && result.handedness[index] ? result.handedness[index][0].displayName : `Hand ${index + 1}`;
      const topGesture = gestures[0];
      if (topGesture) {
        items.push({
          label: `${handedness}: ${topGesture.categoryName}`,
          score: topGesture.score
        });
      }
    });
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

export function cleanupGestureRecognizer() {
  if (mediaManager) mediaManager.cleanup();
  if (worker) {
    worker.postMessage({ type: 'CLEANUP' });
    worker.terminate();
    worker = undefined;
  }
  if (canvasCtx && canvasElement) canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
}
