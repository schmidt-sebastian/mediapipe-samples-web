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
let modelSelector: ModelSelector;
let minDetectionConfidence = 0.5;
let minSuppressionThreshold = 0.3;
let currentDelegate: 'CPU' | 'GPU' = 'CPU';

const models: Record<string, string> = {
  'blaze_face_short_range': 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'
};

// @ts-ignore
import template from '../templates/face-detector.html?raw';
import { ViewToggle } from '../components/view-toggle';
import { ModelSelector } from '../components/model-selector';
// @ts-ignore
import FaceDetectorWorker from '../workers/face-detector.worker.ts?worker';

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
    worker = new FaceDetectorWorker();
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
      if (video && video.srcObject) {
        enableWebcamButton.innerText = 'Disable Webcam';
        enableWebcamButton.disabled = false;
      } else if (enableWebcamButton.innerText !== 'Starting...') {
        enableWebcamButton.innerText = 'Enable Webcam';
        enableWebcamButton.disabled = false;
      }

      modelSelector?.hideProgress();

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
  const viewWebcam = document.getElementById('view-webcam')!;
  const viewImage = document.getElementById('view-image')!;

  const storedMode = localStorage.getItem('mediapipe-running-mode') as 'VIDEO' | 'IMAGE';
  const initialMode = storedMode || 'IMAGE';

  const switchView = (mode: 'VIDEO' | 'IMAGE') => {
    localStorage.setItem('mediapipe-running-mode', mode);
    if (mode === 'VIDEO') {
      viewWebcam.classList.add('active');
      viewImage.classList.remove('active');
      runningMode = 'VIDEO';
      worker?.postMessage({
        type: 'SET_OPTIONS',
        runningMode: 'VIDEO'
      });
      const isWebcamActive = localStorage.getItem('mediapipe-webcam-active') === 'true';
      if (isWebcamActive) {
        enableCam();
      }
    } else {
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

  // View Toggle
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

  // Initialize logic (ViewToggle handles the buttons, we just need to trigger the logic if needed, 
  // but ViewToggle doesn't trigger callback on init, so we manually call switchView or let it be if defaults match)
  // Actually switchView handles both UI visibility and Worker logic.
  // We should match the initial state.
  switchView(initialMode);

  enableWebcamButton.addEventListener('click', toggleCam);

  modelSelector = new ModelSelector(
    'model-selector-container',
    [
      { label: 'BlazeFace (Short Range)', value: 'blaze_face_short_range', isDefault: true }
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
      await initializeDetector();
    }
  );

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
  // Canvas size logic handled by CSS wrapper
  // imageCanvas.style.width = '${image.width}px';
  // imageCanvas.style.height = '${image.height}px';

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
    localStorage.setItem('mediapipe-webcam-active', 'true');
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
    document.getElementById('webcam-placeholder')?.classList.remove('hidden');
    enableWebcamButton.innerText = 'Enable Webcam';
    cancelAnimationFrame(animationFrameId);
    localStorage.setItem('mediapipe-webcam-active', 'false');
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
  // if (detection.keypoints) {
  //   ctx.fillStyle = '#FF0000';
  //   for (const keypoint of detection.keypoints) {
  //      // TODO: Implement keypoint drawing if needed
  //   }
  // }

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
    document.getElementById('webcam-placeholder')?.classList.remove('hidden');
  }

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
