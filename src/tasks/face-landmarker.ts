import { FaceLandmarkerResult, DrawingUtils, FaceLandmarker } from '@mediapipe/tasks-vision';

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
let currentModel = 'face_landmarker';
let numFaces = 1;
let minFaceDetectionConfidence = 0.5;
let minFacePresenceConfidence = 0.5;
let minTrackingConfidence = 0.5;
let currentDelegate: 'CPU' | 'GPU' = 'CPU';

const models: Record<string, string> = {
  'face_landmarker': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
};

// @ts-ignore
import template from '../templates/face-landmarker.html?raw';

export async function setupFaceLandmarker(container: HTMLElement) {
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
import FaceLandmarkerWorker from '../workers/face-landmarker.worker.ts?worker';

function initWorker() {
  if (!worker) {
    worker = new FaceLandmarkerWorker();
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
    numFaces: numFaces,
    minFaceDetectionConfidence: minFaceDetectionConfidence,
    minFacePresenceConfidence: minFacePresenceConfidence,
    minTrackingConfidence: minTrackingConfidence,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true,
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

  // Model Upload Logic (Reuse from existing tasks pattern)
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
  const minFaceDetectionConfidenceInput = document.getElementById('min-face-detection-confidence') as HTMLInputElement;
  const minFaceDetectionConfidenceValue = document.getElementById('min-face-detection-confidence-value')!;
  if (minFaceDetectionConfidenceInput) {
    minFaceDetectionConfidenceInput.addEventListener('input', () => {
      minFaceDetectionConfidence = parseFloat(minFaceDetectionConfidenceInput.value);
      minFaceDetectionConfidenceValue.innerText = minFaceDetectionConfidence.toString();
      worker?.postMessage({ type: 'SET_OPTIONS', minFaceDetectionConfidence: minFaceDetectionConfidence });
      if (runningMode === 'IMAGE') reRunImageDetection();
    });
  }

  const minFacePresenceConfidenceInput = document.getElementById('min-face-presence-confidence') as HTMLInputElement;
  const minFacePresenceConfidenceValue = document.getElementById('min-face-presence-confidence-value')!;
  if (minFacePresenceConfidenceInput) {
    minFacePresenceConfidenceInput.addEventListener('input', () => {
      minFacePresenceConfidence = parseFloat(minFacePresenceConfidenceInput.value);
      minFacePresenceConfidenceValue.innerText = minFacePresenceConfidence.toString();
      worker?.postMessage({ type: 'SET_OPTIONS', minFacePresenceConfidence: minFacePresenceConfidence });
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

  const numFacesInput = document.getElementById('num-faces') as HTMLInputElement;
  const numFacesValue = document.getElementById('num-faces-value')!;
  if (numFacesInput) {
    numFacesInput.addEventListener('input', () => {
      numFaces = parseInt(numFacesInput.value);
      numFacesValue.innerText = numFaces.toString();
      worker?.postMessage({ type: 'SET_OPTIONS', numFaces: numFaces });
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

function displayImageResult(result: FaceLandmarkerResult, image: HTMLImageElement) {
  const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
  const ctx = imageCanvas.getContext('2d')!;

  imageCanvas.width = image.naturalWidth;
  imageCanvas.height = image.naturalHeight;
  imageCanvas.style.width = `${image.width}px`;
  imageCanvas.style.height = `${image.height}px`;

  ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);

  if (result.faceLandmarks) {
    // Re-initialize drawing utils for this context if needed, or just use drawingUtils with this context?
    // DrawingUtils takes a context in constructor.
    const drawingUtilsImage = new DrawingUtils(ctx);
    for (const landmarks of result.faceLandmarks) {
      drawLandmarks(drawingUtilsImage, landmarks);
    }
  }
}

function displayVideoResult(result: FaceLandmarkerResult) {
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  if (result.faceLandmarks) {
    for (const landmarks of result.faceLandmarks) {
      if (drawingUtils) {
        drawLandmarks(drawingUtils, landmarks);
      }
    }
  }
}

function drawLandmarks(drawingUtils: DrawingUtils, landmarks: any[]) {
  drawingUtils.drawConnectors(
    landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION,
    { color: '#C0C0C070', lineWidth: 1 });
  drawingUtils.drawConnectors(
    landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: '#FF3030' });
  drawingUtils.drawConnectors(
    landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
    { color: '#FF3030' });
  drawingUtils.drawConnectors(
    landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: '#30FF30' });
  drawingUtils.drawConnectors(
    landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
    { color: '#30FF30' });
  drawingUtils.drawConnectors(
    landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: '#E0E0E0' });
  drawingUtils.drawConnectors(
    landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: '#E0E0E0' });
  drawingUtils.drawConnectors(
    landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS, { color: '#FF3030' });
  drawingUtils.drawConnectors(
    landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS, { color: '#30FF30' });
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

export function cleanupFaceLandmarker() {
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  stopCam();
  if (worker) worker.postMessage({ type: 'CLEANUP' });
  if (canvasCtx && canvasElement) canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
}
