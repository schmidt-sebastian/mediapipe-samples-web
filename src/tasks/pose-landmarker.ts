import { PoseLandmarker, PoseLandmarkerResult, DrawingUtils } from '@mediapipe/tasks-vision';

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
let currentModel = 'pose_landmarker_lite';
let minPoseDetectionConfidence = 0.5;
let minPosePresenceConfidence = 0.5;
let minTrackingConfidence = 0.5;
let numPoses = 1;
let outputSegmentationMasks = false;
let currentDelegate: 'CPU' | 'GPU' = 'GPU';

const models: Record<string, string> = {
  'pose_landmarker_lite': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
  'pose_landmarker_full': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
  'pose_landmarker_heavy': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
};

// @ts-ignore
import template from '../templates/pose-landmarker.html?raw';

export async function setupPoseLandmarker(container: HTMLElement) {
  container.innerHTML = template;

  video = document.getElementById('webcam') as HTMLVideoElement;
  canvasElement = document.getElementById('output_canvas') as HTMLCanvasElement;
  canvasCtx = canvasElement.getContext('2d')!;
  enableWebcamButton = document.getElementById('webcamButton') as HTMLButtonElement;
  drawingUtils = new DrawingUtils(canvasCtx);

  initWorker();

  // Setup UI event listeners
  setupUI();

  // Initialize landmarker via worker
  await initializeLandmarker();
}

// @ts-ignore
import PoseLandmarkerWorker from '../workers/pose-landmarker.worker.ts?worker';

function initWorker() {
  if (!worker) {
    worker = new PoseLandmarkerWorker();
  }
  if (worker) {
    worker.onmessage = handleWorkerMessage;
  }
}

function handleWorkerMessage(event: MessageEvent) {
  const { type } = event.data;

  switch (type) {
    case 'LOAD_PROGRESS':
      const { progress } = event.data;
      const progressContainer = document.getElementById('model-loading-progress');
      const progressBar = progressContainer?.querySelector('.progress-bar') as HTMLElement;
      const progressText = progressContainer?.querySelector('.progress-text') as HTMLElement;

      if (progressContainer && progressBar && progressText) {
        progressContainer.style.display = 'block';
        const percent = Math.round(progress * 100);
        progressBar.style.width = `${percent}%`;
        progressText.innerText = `Loading Model... ${percent}%`;

        if (progress >= 1) {
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
      updateStatus('Ready');

      // Hide progress
      const pContainer = document.getElementById('model-loading-progress');
      if (pContainer) pContainer.style.display = 'none';

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
  updateStatus('Loading Model...');

  // @ts-ignore
  const baseUrl = import.meta.env.BASE_URL;
  let modelPath = models[currentModel];
  if (!modelPath.startsWith('http')) {
    modelPath = new URL(modelPath, new URL(baseUrl, window.location.origin)).href;
  }

  worker?.postMessage({
    type: 'INIT',
    modelAssetPath: modelPath,
    delegate: currentDelegate,
    minPoseDetectionConfidence,
    minPosePresenceConfidence,
    minTrackingConfidence,
    numPoses,
    outputSegmentationMasks,
    runningMode,
    baseUrl
  });
}

function setupUI() {
  // Tabs
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
      worker?.postMessage({
        type: 'SET_OPTIONS',
        runningMode: 'VIDEO'
      });
      enableCam();
    } else {
      tabWebcam.classList.remove('active');
      tabImage.classList.add('active');
      viewWebcam.classList.remove('active');
      viewImage.classList.add('active');
      runningMode = 'IMAGE';
      worker?.postMessage({
        type: 'SET_OPTIONS',
        runningMode: 'IMAGE'
      });
      stopCam();

      if (isWorkerReady) {
        const testImage = document.getElementById('test-image') as HTMLImageElement;
        if (testImage && testImage.src) {
           triggerImageDetection(testImage);
        }
      }
    }
  };

  const storedMode = localStorage.getItem('mediapipe-running-mode') as 'VIDEO' | 'IMAGE';
  if (storedMode === 'VIDEO') switchView('VIDEO');
  else switchView('IMAGE');

  tabWebcam.addEventListener('click', () => {
    if (runningMode !== 'VIDEO') switchView('VIDEO');
  });
  tabImage.addEventListener('click', () => {
    if (runningMode !== 'IMAGE') switchView('IMAGE');
  });

  enableWebcamButton.addEventListener('click', toggleCam);

  // Model Selection
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
  modelSelect.addEventListener('change', async (e) => {
    currentModel = (e.target as HTMLSelectElement).value;
    await initializeLandmarker();
  });

  const modelUpload = document.getElementById('model-upload') as HTMLInputElement;
  const uploadStatus = document.getElementById('upload-status')!;
  modelUpload.addEventListener('change', async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file) {
      uploadStatus.innerText = file.name;
      const tempModelUrl = URL.createObjectURL(file);
      models['custom'] = tempModelUrl;
      currentModel = 'custom';
      await initializeLandmarker();
    }
  });

  // Settings
  const setupSlider = (id: string, onChange: (val: number) => void) => {
    const input = document.getElementById(id) as HTMLInputElement;
    const valueDisplay = document.getElementById(`${id}-value`)!;
    input.addEventListener('input', () => {
      const val = parseFloat(input.value);
      valueDisplay.innerText = val.toString();
      onChange(val);
    });
  };

  setupSlider('num-poses', (val) => {
    numPoses = val;
    worker?.postMessage({ type: 'SET_OPTIONS', numPoses: numPoses });
    triggerRedetection();
  });

  const outputSegmentationMasksInput = document.getElementById('output-segmentation-masks') as HTMLInputElement;
  if (outputSegmentationMasksInput) {
    outputSegmentationMasksInput.addEventListener('change', () => {
      outputSegmentationMasks = outputSegmentationMasksInput.checked;
      worker?.postMessage({ type: 'SET_OPTIONS', outputSegmentationMasks: outputSegmentationMasks });
      triggerRedetection();
    });
  }

  setupSlider('min-pose-detection-confidence', (val) => {
    minPoseDetectionConfidence = val;
    worker?.postMessage({ type: 'SET_OPTIONS', minPoseDetectionConfidence: minPoseDetectionConfidence });
    triggerRedetection();
  });

  setupSlider('min-pose-presence-confidence', (val) => {
    minPosePresenceConfidence = val;
    worker?.postMessage({ type: 'SET_OPTIONS', minPosePresenceConfidence: minPosePresenceConfidence });
    triggerRedetection();
  });

  setupSlider('min-tracking-confidence', (val) => {
    minTrackingConfidence = val;
    worker?.postMessage({ type: 'SET_OPTIONS', minTrackingConfidence: minTrackingConfidence });
    triggerRedetection();
  });

  const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
  delegateSelect.addEventListener('change', async () => {
    currentDelegate = delegateSelect.value as 'GPU' | 'CPU';
    await initializeLandmarker();
  });

  // Image Upload
  const imageUpload = document.getElementById('image-upload') as HTMLInputElement;
  const imagePreviewContainer = document.getElementById('image-preview-container')!;
  const testImage = document.getElementById('test-image') as HTMLImageElement;
  const dropzone = document.querySelector('.upload-dropzone') as HTMLElement;
  
  if (dropzone) {
    dropzone.addEventListener('click', (e) => {
       // Avoid clicking if clicking button
       if ((e.target as HTMLElement).closest('#re-upload-btn')) return;
       imageUpload.click();
    });
  }

  const reUploadBtn = document.getElementById('re-upload-btn');
  if (reUploadBtn) {
    reUploadBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      imageUpload.click();
    });
  }

  imageUpload.addEventListener('change', (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        testImage.src = e.target?.result as string;
        imagePreviewContainer.style.display = '';
        const dropzoneContent = document.querySelector('.dropzone-content') as HTMLElement;
        if (dropzoneContent) dropzoneContent.style.display = 'none';
        if (reUploadBtn) reUploadBtn.style.display = 'flex';
        
        triggerImageDetection(testImage);
      };
      reader.readAsDataURL(file);
    }
  });
}

function triggerRedetection() {
    if (runningMode === 'IMAGE') {
      const testImage = document.getElementById('test-image') as HTMLImageElement;
      if (testImage && testImage.src) {
        detectImage(testImage);
      }
    }
}

async function detectImage(image: HTMLImageElement) {
  if (!worker || !isWorkerReady) return;

  if (runningMode !== 'IMAGE') {
    runningMode = 'IMAGE';
  }

  const bitmap = await createImageBitmap(image);
  updateStatus(`Processing image...`);
  worker.postMessage({
    type: 'DETECT_IMAGE',
    bitmap: bitmap,
    timestampMs: performance.now()
  }, [bitmap]);
}

function displayImageResult(result: PoseLandmarkerResult) {
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
      drawingUtils.drawLandmarks(landmark, {
        radius: (data) => DrawingUtils.lerp(data.from!.z, -0.15, 0.1, 5, 1)
      });
      drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
    }
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
  if (video.srcObject) {
    stopCam();
  } else {
    enableCam();
  }
}

function stopCam() {
  if (video.srcObject) {
    const stream = video.srcObject as MediaStream;
    const tracks = stream.getTracks();
    tracks.forEach((track) => track.stop());
    video.srcObject = null;
    enableWebcamButton.innerText = 'Enable Webcam';
    cancelAnimationFrame(animationFrameId);
  }
}

async function predictWebcam() {
  if (runningMode === 'IMAGE') {
    runningMode = 'VIDEO';
  }

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

      worker?.postMessage({
        type: 'DETECT_VIDEO',
        bitmap: bitmap,
        timestampMs: timestampMs
      }, [bitmap]);
    } catch (e) {
      console.error("Failed to create ImageBitmap from video", e);
      animationFrameId = window.requestAnimationFrame(predictWebcam);
    }
  } else {
    animationFrameId = window.requestAnimationFrame(predictWebcam);
  }
}

function displayVideoResult(result: PoseLandmarkerResult) {
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  if (result.landmarks) {
    // Re-create DrawingUtils if context changes (resizing), or just use the global one
    // But specific one for video canvas is safer
    if (!drawingUtils) drawingUtils = new DrawingUtils(canvasCtx);
    
    for (const landmark of result.landmarks) {
       drawingUtils.drawLandmarks(landmark, {
        radius: (data) => DrawingUtils.lerp(data.from!.z, -0.15, 0.1, 5, 1)
      });
      drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
    }
  }
  canvasCtx.restore();
}

function updateStatus(msg: string) {
  const el = document.getElementById('status-message');
  if (el) el.innerText = msg;
}

function updateInferenceTime(time: number) {
  const el = document.getElementById('inference-time');
  if (el) el.innerText = `Inference Time: ${time.toFixed(2)} ms`;
}

export function cleanupPoseLandmarker() {
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  stopCam();
  if (worker) {
    worker.postMessage({ type: 'CLEANUP' });
    worker.terminate();
    worker = undefined;
  }
  isWorkerReady = false;
  if (canvasCtx && canvasElement) {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  }
}
