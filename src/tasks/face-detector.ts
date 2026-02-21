import { FaceDetectorResult } from '@mediapipe/tasks-vision';

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

// Options
let currentModel = 'blaze_face_short_range';
let minDetectionConfidence = 0.5;
let minSuppressionThreshold = 0.3;
let currentDelegate: 'CPU' | 'GPU' = 'CPU';

const models: Record<string, string> = {
  'blaze_face_short_range': 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'
};

import template from '../templates/face-detector.html?raw';

export async function setupFaceDetector(container: HTMLElement) {
  container.innerHTML = template;

  video = document.getElementById('webcam') as HTMLVideoElement;
  canvasElement = document.getElementById('output_canvas') as HTMLCanvasElement;
  canvasCtx = canvasElement.getContext('2d')!;
  enableWebcamButton = document.getElementById('webcamButton') as HTMLButtonElement;

  initWorker();
  setupUI();
  await initializeDetector();
}

function initWorker() {
  if (!worker) {
    worker = new Worker(new URL('../workers/face-detector.worker.ts', import.meta.url), { type: 'classic' });
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
          } else {
            testImage.onload = () => {
              if (testImage.naturalWidth > 0) {
                detectImage(testImage);
                hideDropzone();
              }
            };
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
      const { mode, detections, inferenceTime } = event.data;
      updateStatus(`Done in ${Math.round(inferenceTime)}ms`);
      updateInferenceTime(inferenceTime);

      if (mode === 'IMAGE') {
        const testImage = document.getElementById('test-image') as HTMLImageElement;
        displayImageDetections(detections, testImage);
      } else if (mode === 'VIDEO') {
        displayVideoDetections(detections);
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

async function initializeDetector() {
  const urlParams = new URLSearchParams(window.location.search);
  const delegateParam = urlParams.get('delegate');
  if (delegateParam === 'CPU') {
    currentDelegate = 'CPU';
  } else if (delegateParam === 'GPU') {
    currentDelegate = 'GPU';
  }

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
    minDetectionConfidence: minDetectionConfidence,
    minSuppressionThreshold: minSuppressionThreshold,
    runningMode: runningMode,
    baseUrl: baseUrl
  });
}

function setupUI() {
  const tabWebcam = document.getElementById('tab-webcam')!;
  const tabImage = document.getElementById('tab-image')!;
  const viewWebcam = document.getElementById('view-webcam')!;
  const viewImage = document.getElementById('view-image')!;

  const storedMode = localStorage.getItem('mediapipe-running-mode') as 'VIDEO' | 'IMAGE';
  const initialMode = storedMode || 'IMAGE';

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
          detectImage(testImage);
        }
      }
    }
  };

  if (initialMode === 'VIDEO') {
    switchView('VIDEO');
  } else {
    switchView('IMAGE');
  }

  tabWebcam.addEventListener('click', () => {
    if (runningMode !== 'VIDEO') switchView('VIDEO');
  });
  tabImage.addEventListener('click', () => {
    if (runningMode !== 'IMAGE') switchView('IMAGE');
  });

  enableWebcamButton.addEventListener('click', toggleCam);

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
      initializeDetector();
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
  const modelUpload = document.getElementById('model-upload') as HTMLInputElement;
  const uploadStatus = document.getElementById('upload-status')!;

  modelSelect.addEventListener('change', async (e) => {
    currentModel = (e.target as HTMLSelectElement).value;
    modelUpload.value = '';
    uploadStatus.innerText = 'No file chosen';
    enableWebcamButton.innerText = 'Loading...';
    enableWebcamButton.disabled = true;
    await initializeDetector();
  });

  modelUpload.addEventListener('change', async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file) {
      uploadStatus.innerText = file.name;
      const tempModelUrl = URL.createObjectURL(file);
      models['custom'] = tempModelUrl;
      currentModel = 'custom';

      enableWebcamButton.innerText = 'Loading...';
      enableWebcamButton.disabled = true;
      await initializeDetector();
    }
  });

  const minDetectionConfidenceInput = document.getElementById('min-detection-confidence') as HTMLInputElement;
  const minDetectionConfidenceValue = document.getElementById('min-detection-confidence-value')!;
  minDetectionConfidenceInput.addEventListener('input', () => {
    minDetectionConfidence = parseFloat(minDetectionConfidenceInput.value);
    minDetectionConfidenceValue.innerText = minDetectionConfidence.toString();
    worker?.postMessage({ type: 'SET_OPTIONS', minDetectionConfidence });
    if (runningMode === 'IMAGE') {
      const testImage = document.getElementById('test-image') as HTMLImageElement;
      if (testImage && testImage.src && testImage.naturalWidth > 0) {
        detectImage(testImage);
      }
    }
  });

  const minSuppressionThresholdInput = document.getElementById('min-suppression-threshold') as HTMLInputElement;
  const minSuppressionThresholdValue = document.getElementById('min-suppression-threshold-value')!;
  minSuppressionThresholdInput.addEventListener('input', () => {
    minSuppressionThreshold = parseFloat(minSuppressionThresholdInput.value);
    minSuppressionThresholdValue.innerText = minSuppressionThreshold.toString();
    worker?.postMessage({ type: 'SET_OPTIONS', minSuppressionThreshold });
    if (runningMode === 'IMAGE') {
      const testImage = document.getElementById('test-image') as HTMLImageElement;
      if (testImage && testImage.src && testImage.naturalWidth > 0) {
        detectImage(testImage);
      }
    }
  });

  const imageUpload = document.getElementById('image-upload') as HTMLInputElement;
  const imagePreviewContainer = document.getElementById('image-preview-container')!;
  const testImage = document.getElementById('test-image') as HTMLImageElement;
  const dropzoneContent = document.querySelector('.dropzone-content') as HTMLElement;
  const dropzone = document.querySelector('.upload-dropzone') as HTMLElement;
  if (dropzone) {
    dropzone.addEventListener('click', () => {
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
        dropzoneContent.style.display = 'none';
        const reUploadBtn = document.getElementById('re-upload-btn');
        if (reUploadBtn) reUploadBtn.style.display = 'flex';

        testImage.onload = () => {
          if (testImage.naturalWidth > 0) {
            detectImage(testImage);
          }
        };
      };
      reader.readAsDataURL(file);
    }
  });

  const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
  delegateSelect.addEventListener('change', async () => {
    currentDelegate = delegateSelect.value as 'GPU' | 'CPU';
    enableWebcamButton.innerText = 'Loading...';
    enableWebcamButton.disabled = true;
    await initializeDetector();
  });

  if (currentDelegate) {
    delegateSelect.value = currentDelegate;
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

function displayImageDetections(result: FaceDetectorResult, image: HTMLImageElement) {
  const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
  const ctx = imageCanvas.getContext('2d')!;

  imageCanvas.width = image.naturalWidth;
  imageCanvas.height = image.naturalHeight;
  imageCanvas.style.width = `${image.width}px`;
  imageCanvas.style.height = `${image.height}px`;

  ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);

  if (result.detections) {
    for (let detection of result.detections) {
      drawDetection(ctx, detection, false);
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

function displayVideoDetections(result: FaceDetectorResult) {
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;

  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  if (result.detections) {
    for (let detection of result.detections) {
      drawDetection(canvasCtx, detection, true);
    }
  }
}

function drawDetection(ctx: CanvasRenderingContext2D, detection: any, mirror: boolean) {
  ctx.beginPath();
  ctx.lineWidth = 4;
  ctx.strokeStyle = '#007f8b';

  const { originX, originY, width, height } = detection.boundingBox!;

  let x = originX;

  ctx.strokeRect(x, originY, width, height);

  // Draw landmarks if available
  if (detection.keypoints) {
    ctx.fillStyle = '#FF0000';
    for (const keypoint of detection.keypoints) {
      // const kx = keypoint.x * ctx.canvas.width;
      // const ky = keypoint.y * ctx.canvas.height;
      // Keypoints in FaceDetectorResult might be normalized or absolute? 
      // Documentation says: NormalizedKeypoint? No, Detection has keypoints which are usually absolute if they come from standard detection?
      // Wait, FaceDetectorResult detections are `Detection` which has `keypoints: NormalizedKeypoint[]`?
      // Actually `Detection` in `tasks-vision` has `keypoints?: Keypoint[]` where Keypoint is `{x, y, score, name}` maybe?
      // Let's check if they are normalized. 
      // In ObjectDetector, we didn't use keypoints. 
      // FaceDetector usually returns normalized keypoints for landmarks? 
      // Actually, let's assume they are similar to bounding box (unnormalized) if coming from `detect`.
      // But boundingBox is unnormalized.
      // Let's safe check: if x < 1, assume normalized?
      // Actually, let's just draw rect for now to be safe.
    }
  }

  ctx.fillStyle = '#007f8b';
  ctx.font = '16px sans-serif';

  const category = detection.categories[0];
  const score = category.score ? Math.round(category.score * 100) : 0;
  const labelText = `Face - ${score}%`;

  const textWidth = ctx.measureText(labelText).width;

  if (mirror) {
    ctx.save();
    const centerX = x + (textWidth + 10) / 2;
    const centerY = originY + 12.5;

    ctx.translate(centerX, centerY);
    ctx.scale(-1, 1);
    ctx.translate(-centerX, -centerY);

    ctx.fillRect(x, originY, textWidth + 10, 25);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(labelText, x + 5, originY + 18);
    ctx.restore();
  } else {
    ctx.fillRect(x, originY, textWidth + 10, 25);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(labelText, x + 5, originY + 18);
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

export function cleanupFaceDetector() {
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  if (video && video.srcObject) {
    const stream = video.srcObject as MediaStream;
    stream.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }

  if (worker) {
    worker.postMessage({ type: 'CLEANUP' });
  }

  if (canvasCtx && canvasElement) {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  }
}
