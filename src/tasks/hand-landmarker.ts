import { HandLandmarkerResult, DrawingUtils, HandLandmarker } from '@mediapipe/tasks-vision';

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
let currentModel = 'hand_landmarker';
let numHands = 2;
let minHandDetectionConfidence = 0.5;
let minHandPresenceConfidence = 0.5;
let minTrackingConfidence = 0.5;
let currentDelegate: 'CPU' | 'GPU' = 'CPU';

const models: Record<string, string> = {
  'hand_landmarker': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
};

// @ts-ignore
import template from '../templates/hand-landmarker.html?raw';

export async function setupHandLandmarker(container: HTMLElement) {
  container.innerHTML = template;

  video = document.getElementById('webcam') as HTMLVideoElement;
  canvasElement = document.getElementById('output_canvas') as HTMLCanvasElement;
  canvasCtx = canvasElement.getContext('2d')!;
  enableWebcamButton = document.getElementById('webcamButton') as HTMLButtonElement;

  if (canvasCtx) {
    drawingUtils = new DrawingUtils(canvasCtx);
  }

  initWorker();
  setupUI();
  await initializeLandmarker();
}

// @ts-ignore
import HandLandmarkerWorker from '../workers/hand-landmarker.worker.ts?worker';

function initWorker() {
  if (!worker) {
    worker = new HandLandmarkerWorker();
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
      const progressContainer = document.getElementById('model-loading-progress');
      const progressBar = progressContainer?.querySelector('.progress-bar') as HTMLElement;
      const progressText = progressContainer?.querySelector('.progress-text') as HTMLElement;

      if (progressContainer && progressBar && progressText) {
        progressContainer.style.display = 'block';
        const percent = Math.round((loaded / total) * 100);
        progressBar.style.width = `${percent}%`;
        progressText.innerText = `Loading Model... ${percent}%`;

        if (percent >= 100) {
          setTimeout(() => {
            progressContainer.style.display = 'none';
          }, 500);
        }
      }
      break;

    case 'INIT_DONE':
      document.querySelector('.viewport')?.classList.remove('loading-model');
      isWorkerReady = true;
      enableWebcamButton.disabled = false;
      enableWebcamButton.innerText = 'Enable Webcam';

      const pContainer = document.getElementById('model-loading-progress');
      if (pContainer) pContainer.style.display = 'none';

      if (runningMode === 'VIDEO') {
        if (video.srcObject) {
          enableCam();
        }
      } else if (runningMode === 'IMAGE') {
        const testImage = document.getElementById('test-image') as HTMLImageElement;
        if (testImage.style.display !== 'none' && testImage.src) {
          if (testImage.complete && testImage.naturalWidth > 0) {
            detectImage(testImage);
            hideDropzone();
          }
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
        const testImage = document.getElementById('test-image') as HTMLImageElement;
        displayImageResult(result, testImage);
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

function hideDropzone() {
  const dropzoneContent = document.querySelector('.dropzone-content') as HTMLElement;
  const imagePreviewContainer = document.getElementById('image-preview-container');
  const reUploadBtn = document.getElementById('re-upload-btn');
  if (dropzoneContent) dropzoneContent.style.display = 'none';
  if (imagePreviewContainer) imagePreviewContainer.style.display = '';
  if (reUploadBtn) reUploadBtn.style.display = 'flex';
}

async function initializeLandmarker() {
  enableWebcamButton.disabled = true;
  if (!video.srcObject) {
    enableWebcamButton.innerText = 'Initializing...';
  }
  document.querySelector('.viewport')?.classList.add('loading-model');
  isWorkerReady = false;

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
    numHands: numHands,
    minHandDetectionConfidence: minHandDetectionConfidence,
    minHandPresenceConfidence: minHandPresenceConfidence,
    minTrackingConfidence: minTrackingConfidence,
    runningMode: runningMode,
    baseUrl: baseUrl
  });
}

function setupUI() {
  const tabWebcam = document.getElementById('tab-webcam')!;
  const tabImage = document.getElementById('tab-image')!;
  const viewWebcam = document.getElementById('view-webcam')!;
  const viewImage = document.getElementById('view-image')!;

  const switchView = (mode: 'VIDEO' | 'IMAGE') => {
    localStorage.setItem('mediapipe-running-mode', mode);
    if (mode === 'VIDEO') {
      tabWebcam.classList.add('active');
      tabImage.classList.remove('active');
      viewWebcam.classList.add('active');
      viewImage.classList.remove('active');
      runningMode = 'VIDEO';
      worker?.postMessage({ type: 'SET_OPTIONS', runningMode: 'VIDEO' });
      enableCam();
    } else {
      tabWebcam.classList.remove('active');
      tabImage.classList.add('active');
      viewWebcam.classList.remove('active');
      viewImage.classList.add('active');
      runningMode = 'IMAGE';
      worker?.postMessage({ type: 'SET_OPTIONS', runningMode: 'IMAGE' });
      stopCam();

      if (isWorkerReady) {
        const testImage = document.getElementById('test-image') as HTMLImageElement;
        if (testImage && testImage.src) detectImage(testImage);
      }
    }
  };

  const storedMode = localStorage.getItem('mediapipe-running-mode') as 'VIDEO' | 'IMAGE';
  if (storedMode === 'VIDEO') switchView('VIDEO');
  else switchView('IMAGE');

  tabWebcam.addEventListener('click', () => { if (runningMode !== 'VIDEO') switchView('VIDEO'); });
  tabImage.addEventListener('click', () => { if (runningMode !== 'IMAGE') switchView('IMAGE'); });

  enableWebcamButton.addEventListener('click', toggleCam);

  // Model Upload Logic
  const tabModelList = document.getElementById('tab-model-list')!;
  const tabModelUpload = document.getElementById('tab-model-upload')!;
  const viewModelList = document.getElementById('view-model-list')!;
  const viewModelUpload = document.getElementById('view-model-upload')!;

  const switchModelTab = (tab: 'LIST' | 'UPLOAD') => {
    if (tab === 'LIST') {
      tabModelList.classList.add('active');
      tabModelUpload.classList.remove('active');
      viewModelList.classList.add('active');
      viewModelUpload.classList.remove('active');
      const select = document.getElementById('model-select') as HTMLSelectElement;
      currentModel = select.value;
      initializeLandmarker();
    } else {
      tabModelList.classList.remove('active');
      tabModelUpload.classList.add('active');
      viewModelList.classList.remove('active');
      viewModelUpload.classList.add('active');
    }
  };

  tabModelList.addEventListener('click', () => switchModelTab('LIST'));
  tabModelUpload.addEventListener('click', () => switchModelTab('UPLOAD'));

  const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
  modelSelect.addEventListener('change', (e) => {
    currentModel = (e.target as HTMLSelectElement).value;
    initializeLandmarker();
  });

  // Confidence Sliders
  const minHandDetectionConfidenceInput = document.getElementById('min-hand-detection-confidence') as HTMLInputElement;
  const minHandDetectionConfidenceValue = document.getElementById('min-hand-detection-confidence-value')!;
  if (minHandDetectionConfidenceInput) {
    minHandDetectionConfidenceInput.addEventListener('input', () => {
      minHandDetectionConfidence = parseFloat(minHandDetectionConfidenceInput.value);
      minHandDetectionConfidenceValue.innerText = minHandDetectionConfidence.toString();
      worker?.postMessage({ type: 'SET_OPTIONS', minHandDetectionConfidence: minHandDetectionConfidence });
      if (runningMode === 'IMAGE') reRunImageDetection();
    });
  }

  const minHandPresenceConfidenceInput = document.getElementById('min-hand-presence-confidence') as HTMLInputElement;
  const minHandPresenceConfidenceValue = document.getElementById('min-hand-presence-confidence-value')!;
  if (minHandPresenceConfidenceInput) {
    minHandPresenceConfidenceInput.addEventListener('input', () => {
      minHandPresenceConfidence = parseFloat(minHandPresenceConfidenceInput.value);
      minHandPresenceConfidenceValue.innerText = minHandPresenceConfidence.toString();
      worker?.postMessage({ type: 'SET_OPTIONS', minHandPresenceConfidence: minHandPresenceConfidence });
      if (runningMode === 'IMAGE') reRunImageDetection();
    });
  }

  const minTrackingConfidenceInput = document.getElementById('min-tracking-confidence') as HTMLInputElement;
  const minTrackingConfidenceValue = document.getElementById('min-tracking-confidence-value')!;
  if (minTrackingConfidenceInput) {
    minTrackingConfidenceInput.addEventListener('input', () => {
      minTrackingConfidence = parseFloat(minTrackingConfidenceInput.value);
      minTrackingConfidenceValue.innerText = minTrackingConfidence.toString();
      worker?.postMessage({ type: 'SET_OPTIONS', minTrackingConfidence: minTrackingConfidence });
      if (runningMode === 'IMAGE') reRunImageDetection();
    });
  }

  const numHandsInput = document.getElementById('num-hands') as HTMLInputElement;
  const numHandsValue = document.getElementById('num-hands-value')!;
  if (numHandsInput) {
    numHandsInput.addEventListener('input', () => {
      numHands = parseInt(numHandsInput.value);
      numHandsValue.innerText = numHands.toString();
      worker?.postMessage({ type: 'SET_OPTIONS', numHands: numHands });
      if (runningMode === 'IMAGE') reRunImageDetection();
    });
  }

  const imageUpload = document.getElementById('image-upload') as HTMLInputElement;
  const imagePreviewContainer = document.getElementById('image-preview-container')!;
  const testImage = document.getElementById('test-image') as HTMLImageElement;
  const dropzoneContent = document.querySelector('.dropzone-content') as HTMLElement;
  const dropzone = document.querySelector('.upload-dropzone') as HTMLElement;
  if (dropzone) dropzone.addEventListener('click', () => imageUpload.click());

  const reUploadBtn = document.getElementById('re-upload-btn');
  if (reUploadBtn) reUploadBtn.addEventListener('click', (e) => { e.stopPropagation(); imageUpload.click(); });

  imageUpload.addEventListener('change', (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        testImage.src = e.target?.result as string;
        imagePreviewContainer.style.display = '';
        dropzoneContent.style.display = 'none';
        if (reUploadBtn) reUploadBtn.style.display = 'flex';
        testImage.onload = () => { if (testImage.naturalWidth > 0) detectImage(testImage); };
      };
      reader.readAsDataURL(file);
    }
  });

  const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
  delegateSelect.addEventListener('change', async () => {
    currentDelegate = delegateSelect.value as 'GPU' | 'CPU';
    await initializeLandmarker();
  });
  if (currentDelegate) delegateSelect.value = currentDelegate;
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

function displayImageResult(result: HandLandmarkerResult, image: HTMLImageElement) {
  const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
  const ctx = imageCanvas.getContext('2d')!;

  imageCanvas.width = image.naturalWidth;
  imageCanvas.height = image.naturalHeight;
  imageCanvas.style.width = `${image.width}px`;
  imageCanvas.style.height = `${image.height}px`;

  ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);

  if (result.landmarks) {
    console.log('HandLandmarker: drawing landmarks', result.landmarks.length);
    const drawingUtilsImage = new DrawingUtils(ctx);
    for (const landmarks of result.landmarks) {
      console.log('HandLandmarker: drawing landmark set', landmarks);
      drawLandmarks(drawingUtilsImage, landmarks);
    }
  } else {
    console.log('HandLandmarker: no landmarks found');
  }
}

function displayVideoResult(result: HandLandmarkerResult) {
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  if (result.landmarks) {
    for (const landmarks of result.landmarks) {
      if (drawingUtils) {
        drawLandmarks(drawingUtils, landmarks);
      }
    }
  }
}

function drawLandmarks(drawingUtils: DrawingUtils, landmarks: any[]) {
  drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
    color: "#00FF00",
    lineWidth: 5
  });
  drawingUtils.drawLandmarks(landmarks, { color: "#FF0000", lineWidth: 2 });
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
        // Workaround for some environments
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

export function cleanupHandLandmarker() {
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
