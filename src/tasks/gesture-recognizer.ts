import { GestureRecognizerResult, DrawingUtils, HandLandmarker } from '@mediapipe/tasks-vision';

let worker: Worker | undefined;
let runningMode: 'IMAGE' | 'VIDEO' = 'IMAGE';
let video: HTMLVideoElement;
let canvasElement: HTMLCanvasElement;
let canvasCtx: CanvasRenderingContext2D;
let enableWebcamButton: HTMLButtonElement;
let lastVideoTimeSeconds = -1;
let lastTimestampMs = -1;
let animationFrameId: number;
let isWorkerReady = false;
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

// @ts-ignore
// @ts-ignore
import template from '../templates/gesture-recognizer.html?raw';
import { ViewToggle } from '../components/view-toggle';
import { ModelSelector } from '../components/model-selector';

export async function setupGestureRecognizer(container: HTMLElement) {
  container.innerHTML = template;

  video = document.getElementById('webcam') as HTMLVideoElement;
  canvasElement = document.getElementById('output_canvas') as HTMLCanvasElement;
  canvasCtx = canvasElement.getContext('2d')!;
  enableWebcamButton = document.getElementById('webcamButton') as HTMLButtonElement;
  drawingUtils = new DrawingUtils(canvasCtx);

  initWorker();
  setupUI();
  await initializeRecognizer();
}

// @ts-ignore
import GestureRecognizerWorker from '../workers/gesture-recognizer.worker.ts?worker';

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
      isWorkerReady = true;
      enableWebcamButton.disabled = false;
      enableWebcamButton.innerText = 'Enable Webcam';
      updateStatus('Ready');

      modelSelector?.hideProgress();

      if (runningMode === 'VIDEO') {
        if (video.srcObject) {
          enableCam();
        }
      } else if (runningMode === 'IMAGE') {
        const testImage = document.getElementById('test-image') as HTMLImageElement;
        if (testImage.style.display !== 'none' && testImage.src) {
          triggerImageDetection(testImage);
        }
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
        if (video.srcObject && !video.paused) {
          animationFrameId = window.requestAnimationFrame(predictWebcam);
        }
      }
      break;

    case 'ERROR':
    case 'DETECT_ERROR':
      console.error('Worker error:', event.data.error);
      updateStatus(`Error: ${event.data.error}`);
      break;
  }
}

function triggerImageDetection(image: HTMLImageElement) {
  if (image.complete && image.naturalWidth > 0) {
    detectImage(image);
    hideDropzone();
  } else {
    image.onload = () => {
      if (image.naturalWidth > 0) {
        detectImage(image);
        hideDropzone();
      }
    };
  }
}

function hideDropzone() {
  const dropzoneContent = document.querySelector('.dropzone-content') as HTMLElement;
  const imagePreviewContainer = document.getElementById('image-preview-container');
  if (dropzoneContent) dropzoneContent.style.display = 'none';
  if (imagePreviewContainer) imagePreviewContainer.style.display = '';
}

async function initializeRecognizer() {
  enableWebcamButton.disabled = true;
  if (!video.srcObject) {
    enableWebcamButton.innerText = 'Initializing...';
  }
  document.querySelector('.viewport')?.classList.add('loading-model');
  isWorkerReady = false;
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
    runningMode,
    baseUrl
  });
}

function setupUI() {
  const viewWebcam = document.getElementById('view-webcam')!;
  const viewImage = document.getElementById('view-image')!;

  const switchView = (mode: 'VIDEO' | 'IMAGE') => {
    localStorage.setItem('mediapipe-running-mode', mode);
    if (mode === 'VIDEO') {
      viewWebcam.classList.add('active');
      viewImage.classList.remove('active');
      runningMode = 'VIDEO';
      worker?.postMessage({ type: 'SET_OPTIONS', runningMode: 'VIDEO' });
      enableCam();
    } else {
      viewWebcam.classList.remove('active');
      viewImage.classList.add('active');
      runningMode = 'IMAGE';
      worker?.postMessage({ type: 'SET_OPTIONS', runningMode: 'IMAGE' });
      stopCam();

      if (isWorkerReady) {
        const testImage = document.getElementById('test-image') as HTMLImageElement;
        if (testImage && testImage.src) triggerImageDetection(testImage);
      }
    }
  };

  const storedMode = localStorage.getItem('mediapipe-running-mode') as 'VIDEO' | 'IMAGE';
  const initialMode = storedMode || 'IMAGE';

  new ViewToggle(
    'view-mode-toggle',
    [
      { label: 'Webcam', value: 'video' },
      { label: 'Image', value: 'image' }
    ],
    initialMode.toLowerCase(),
    (value) => {
      switchView(value === 'video' ? 'VIDEO' : 'IMAGE');
    }
  );

  switchView(initialMode);

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
      enableWebcamButton.innerText = 'Loading...';
      enableWebcamButton.disabled = true;
      await initializeRecognizer();
    }
  );

  enableWebcamButton.addEventListener('click', toggleCam);

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
    if (runningMode === 'IMAGE') reRunImageDetection();
  });

  setupSlider('min-hand-detection-confidence', (val) => {
    minHandDetectionConfidence = val;
    worker?.postMessage({ type: 'SET_OPTIONS', minHandDetectionConfidence: minHandDetectionConfidence });
    if (runningMode === 'IMAGE') reRunImageDetection();
  });

  setupSlider('min-hand-presence-confidence', (val) => {
    minHandPresenceConfidence = val;
    worker?.postMessage({ type: 'SET_OPTIONS', minHandPresenceConfidence: minHandPresenceConfidence });
    if (runningMode === 'IMAGE') reRunImageDetection();
  });

  setupSlider('min-tracking-confidence', (val) => {
    minTrackingConfidence = val;
    worker?.postMessage({ type: 'SET_OPTIONS', minTrackingConfidence: minTrackingConfidence });
    if (runningMode === 'IMAGE') reRunImageDetection();
  });

  const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
  delegateSelect.addEventListener('change', async () => {
    currentDelegate = delegateSelect.value as 'GPU' | 'CPU';
    await initializeRecognizer();
  });
  delegateSelect.value = currentDelegate;

  // Image Upload
  const imageUpload = document.getElementById('image-upload') as HTMLInputElement;
  const imagePreviewContainer = document.getElementById('image-preview-container')!;
  const testImage = document.getElementById('test-image') as HTMLImageElement;
  const dropzone = document.querySelector('.upload-dropzone') as HTMLElement;
  const dropzoneContent = document.querySelector('.dropzone-content') as HTMLElement;

  // Hide dropzone content if we have a default image
  if (testImage.src && dropzoneContent) {
    dropzoneContent.style.display = 'none';
  }

  if (dropzone) dropzone.addEventListener('click', () => imageUpload.click());

  imageUpload.addEventListener('change', (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        testImage.src = e.target?.result as string;
        imagePreviewContainer.style.display = '';
        const dropzoneContent = document.querySelector('.dropzone-content') as HTMLElement;
        if (dropzoneContent) dropzoneContent.style.display = 'none';

        triggerImageDetection(testImage);
      };
      reader.readAsDataURL(file);
    }
  });


}

function reRunImageDetection() {
  const testImage = document.getElementById('test-image') as HTMLImageElement;
  if (testImage && testImage.src && testImage.naturalWidth > 0) {
    detectImage(testImage);
  }
}

async function detectImage(image: HTMLImageElement) {
  if (!worker || !isWorkerReady) return;
  if (runningMode !== 'IMAGE') runningMode = 'IMAGE';

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
  const outputDiv = document.getElementById('gesture-output');
  if (!outputDiv) return;

  if (result.gestures && result.gestures.length > 0) {
    let html = '';
    result.gestures.forEach((gestures, index) => {
      const handedness = result.handedness && result.handedness[index] ? result.handedness[index][0].displayName : `Hand ${index + 1}`;
      const topGesture = gestures[0];
      if (topGesture) {
        html += `<div class="gesture-item">
                           <strong>${handedness}:</strong> ${topGesture.categoryName} (${(topGesture.score * 100).toFixed(1)}%)
                         </div>`;
      }
    });
    outputDiv.innerHTML = html;
  } else {
    outputDiv.innerHTML = '<p>No gestures detected.</p>';
  }
}

async function enableCam() {
  if (!worker || video.srcObject) return;

  enableWebcamButton.innerText = 'Starting...';
  enableWebcamButton.disabled = true;
  const constraints = { video: true };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    document.getElementById('webcam-placeholder')?.classList.add('hidden');

    const playAndPredict = () => {
      video.play().catch(console.error);
      predictWebcam();
    };

    if (video.readyState >= 2) {
      playAndPredict();
    } else {
      video.addEventListener('loadeddata', playAndPredict, { once: true });
    }

    runningMode = 'VIDEO';
    worker.postMessage({ type: 'SET_OPTIONS', runningMode: 'VIDEO' });
    updateStatus('Webcam running...');
    enableWebcamButton.innerText = 'Disable Webcam';
    enableWebcamButton.disabled = false;
  } catch (err) {
    console.error(err);
    updateStatus('Camera error!');
    enableWebcamButton.innerText = 'Enable Webcam';
    enableWebcamButton.disabled = false;
  }
}

function toggleCam() {
  if (video.srcObject) stopCam();
  else enableCam();
}

function stopCam() {
  if (video.srcObject) {
    const stream = video.srcObject as MediaStream;
    stream.getTracks().forEach(t => t.stop());
    video.srcObject = null;
    document.getElementById('webcam-placeholder')?.classList.remove('hidden');
    enableWebcamButton.innerText = 'Enable Webcam';
    cancelAnimationFrame(animationFrameId);
  }
}

async function predictWebcam() {
  if (runningMode === 'IMAGE') runningMode = 'VIDEO';
  if (!isWorkerReady || !worker) {
    animationFrameId = window.requestAnimationFrame(predictWebcam);
    return;
  }

  if (video.currentTime !== lastVideoTimeSeconds) {
    lastVideoTimeSeconds = video.currentTime;
    try {
      let bitmap: ImageBitmap;
      if (navigator.webdriver) {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth || 640;
        tempCanvas.height = video.videoHeight || 480;
        const ctx = tempCanvas.getContext('2d', { willReadFrequently: true });
        ctx?.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
        bitmap = await window.createImageBitmap(tempCanvas);
      } else {
        bitmap = await window.createImageBitmap(video);
      }

      const now = performance.now();
      const timestampMs = now > lastTimestampMs ? now : lastTimestampMs + 1;
      lastTimestampMs = timestampMs;

      worker.postMessage({
        type: 'DETECT_VIDEO',
        bitmap: bitmap,
        timestampMs: timestampMs
      }, [bitmap]);
    } catch (e) {
      console.error("Failed to create ImageBitmap", e);
      animationFrameId = window.requestAnimationFrame(predictWebcam);
    }
  } else {
    animationFrameId = window.requestAnimationFrame(predictWebcam);
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
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  stopCam();
  if (worker) {
    worker.postMessage({ type: 'CLEANUP' });
    worker.terminate();
    worker = undefined;
  }
  isWorkerReady = false;
  if (canvasCtx && canvasElement) canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
}
