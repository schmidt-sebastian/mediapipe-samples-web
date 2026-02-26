import { ImageClassifierResult } from '@mediapipe/tasks-vision';

let worker: Worker | undefined;
let runningMode: 'IMAGE' | 'VIDEO' = 'IMAGE';
let video: HTMLVideoElement;
let enableWebcamButton: HTMLButtonElement;
let lastVideoTimeSeconds = -1;
let lastTimestampMs = -1;
let animationFrameId: number;
let isWorkerReady = false;

// Options
let currentModel = 'efficientnet_lite0';
let currentDelegate: 'CPU' | 'GPU' = 'GPU';
let maxResults = 3;
let scoreThreshold = 0.0;

const models: Record<string, string> = {
  'efficientnet_lite0': 'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite',
  'efficientnet_lite2': 'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite2/float32/1/efficientnet_lite2.tflite'
};

// @ts-ignore
// @ts-ignore
import template from '../templates/image-classifier.html?raw';
import { ViewToggle } from '../components/view-toggle';

export async function setupImageClassifier(container: HTMLElement) {
  container.innerHTML = template;

  video = document.getElementById('webcam') as HTMLVideoElement;
  enableWebcamButton = document.getElementById('webcamButton') as HTMLButtonElement;

  initWorker();
  setupUI();
  await initializeClassifier();
}

// @ts-ignore
import ImageClassifierWorker from '../workers/image-classifier.worker.ts?worker';

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
    case 'INIT_DONE':
      document.querySelector('.viewport')?.classList.remove('loading-model');
      isWorkerReady = true;
      enableWebcamButton.disabled = false;
      enableWebcamButton.innerText = 'Enable Webcam';
      updateStatus('Ready');

      if (runningMode === 'VIDEO') {
        if (video.srcObject) {
          enableCam();
        }
      } else if (runningMode === 'IMAGE') {
        const testImage = document.getElementById('test-image') as HTMLImageElement;
        if (testImage.style.display !== 'none' && testImage.src) {
          triggerImageClassification(testImage);
        }
      }
      break;

    case 'CLASSIFY_RESULT':
      const { mode, result, inferenceTime } = event.data;
      updateStatus(`Done in ${Math.round(inferenceTime)}ms`);
      updateInferenceTime(inferenceTime);

      if (mode === 'IMAGE') {
        displayResult(result);
      } else if (mode === 'VIDEO') {
        displayResult(result);
        if (video.srcObject && !video.paused) {
          animationFrameId = window.requestAnimationFrame(predictWebcam);
        }
      }
      break;

    case 'ERROR':
      console.error('Worker error:', event.data.error);
      updateStatus(`Error: ${event.data.error}`);
      break;
  }
}

function triggerImageClassification(image: HTMLImageElement) {
  if (image.complete && image.naturalWidth > 0) {
    classifyImage(image);
  } else {
    image.onload = () => {
      if (image.naturalWidth > 0) {
        classifyImage(image);
      }
    };
  }
}

async function initializeClassifier() {
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

  worker?.postMessage({
    type: 'INIT',
    modelAssetPath: modelPath,
    delegate: currentDelegate,
    runningMode,
    maxResults,
    scoreThreshold,
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
        if (testImage && testImage.src) triggerImageClassification(testImage);
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

  enableWebcamButton.addEventListener('click', toggleCam);

  const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
  modelSelect.addEventListener('change', async () => {
    currentModel = modelSelect.value;
    await initializeClassifier();
  });

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
    if (runningMode === 'IMAGE') reRunImageClassification();
  });

  const scoreThresholdInput = document.getElementById('score-threshold') as HTMLInputElement;
  const scoreThresholdValue = document.getElementById('score-threshold-value')!;
  scoreThresholdInput.addEventListener('input', () => {
    scoreThreshold = parseInt(scoreThresholdInput.value) / 100;
    scoreThresholdValue.innerText = `${parseInt(scoreThresholdInput.value)}%`;
    worker?.postMessage({ type: 'SET_OPTIONS', scoreThreshold });
    if (runningMode === 'IMAGE') reRunImageClassification();
  });

  const imageUpload = document.getElementById('image-upload') as HTMLInputElement;
  const imagePreviewContainer = document.getElementById('image-preview-container')!;
  const testImage = document.getElementById('test-image') as HTMLImageElement;
  const dropzone = document.querySelector('.upload-dropzone') as HTMLElement;
  const dropzoneContent = document.querySelector('.dropzone-content') as HTMLElement;

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

        triggerImageClassification(testImage);
      };
      reader.readAsDataURL(file);
    }
  });
}

function reRunImageClassification() {
  const testImage = document.getElementById('test-image') as HTMLImageElement;
  if (testImage && testImage.src && testImage.naturalWidth > 0) {
    classifyImage(testImage);
  }
}

async function classifyImage(image: HTMLImageElement) {
  if (!worker || !isWorkerReady) return;
  if (runningMode !== 'IMAGE') runningMode = 'IMAGE';

  const bitmap = await createImageBitmap(image);
  updateStatus(`Processing image...`);
  worker.postMessage({
    type: 'CLASSIFY_IMAGE',
    bitmap: bitmap,
    timestampMs: performance.now()
  }, [bitmap]);
}

function displayResult(result: ImageClassifierResult) {
  const resultsContainer = document.getElementById('classification-results');
  if (!resultsContainer) return;

  resultsContainer.innerHTML = '';
  
  if (result.classifications && result.classifications.length > 0) {
    const categories = result.classifications[0].categories;
    
    if (categories.length === 0) {
      resultsContainer.innerHTML = '<div class="no-results">No categories found matching criteria</div>';
      return;
    }

    categories.forEach(category => {
      const scorePercent = Math.round(category.score * 100);
      const row = document.createElement('div');
      row.className = 'classification-row';
      row.innerHTML = `
        <div class="category-name">${category.categoryName || 'Unknown'}</div>
        <div class="score-container">
          <div class="score-track">
            <div class="score-fill" style="width: ${scorePercent}%"></div>
          </div>
          <div class="score-text">${scorePercent}%</div>
        </div>
      `;
      resultsContainer.appendChild(row);
    });
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
        type: 'CLASSIFY_VIDEO',
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

export function cleanupImageClassifier() {
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  stopCam();
  if (worker) {
    worker.postMessage({ type: 'CLEANUP' });
    worker.terminate();
    worker = undefined;
  }
  isWorkerReady = false;
}
