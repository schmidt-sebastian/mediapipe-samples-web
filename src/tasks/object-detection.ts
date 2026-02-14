import { ObjectDetectorResult } from '@mediapipe/tasks-vision';

let worker: Worker | undefined;
let runningMode: 'IMAGE' | 'VIDEO' = 'IMAGE';
let video: HTMLVideoElement;
let canvasElement: HTMLCanvasElement;
let canvasCtx: CanvasRenderingContext2D;
let enableWebcamButton: HTMLButtonElement;
let lastVideoTime = -1;
let animationFrameId: number;
let isWorkerReady = false;

// Options
let currentModel = 'efficientdet_lite0';
let scoreThreshold = 0.5;
let maxResults = 3;
let currentDelegate: 'CPU' | 'GPU' = 'CPU';

const models: Record<string, string> = {
  'efficientdet_lite0': 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite',
  'efficientdet_lite2': 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float32/1/efficientdet_lite2.tflite',
  'ssd_mobilenet_v2': 'https://storage.googleapis.com/mediapipe-models/object_detector/ssd_mobilenet_v2/float32/1/ssd_mobilenet_v2.tflite'
};

import template from '../templates/object-detection.html?raw';

export async function setupObjectDetection(container: HTMLElement) {
  container.innerHTML = template;

  video = document.getElementById('webcam') as HTMLVideoElement;
  canvasElement = document.getElementById('output_canvas') as HTMLCanvasElement;
  canvasCtx = canvasElement.getContext('2d')!;
  enableWebcamButton = document.getElementById('webcamButton') as HTMLButtonElement;

  initWorker();

  // Setup UI event listeners
  setupUI();

  // Initialize detector via worker
  await initializeDetector();
}

// @ts-ignore
import ObjectDetectionWorker from '../workers/object-detection.worker.ts?worker';

function initWorker() {
  if (!worker) {
    worker = new ObjectDetectionWorker();
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
              detectImage(testImage);
              hideDropzone();
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
        // Render current video frame detections
        displayVideoDetections(detections);

        // Request next frame if still running video
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
  // Check for delegate override in URL
  const urlParams = new URLSearchParams(window.location.search);
  const delegateParam = urlParams.get('delegate');
  if (delegateParam === 'CPU') {
    currentDelegate = 'CPU';
  } else if (delegateParam === 'GPU') {
    console.warn('GPU requested via URL, but Object Detection Worker only supports CPU. Forcing CPU.');
    currentDelegate = 'CPU';
  }

  enableWebcamButton.disabled = true;
  if (!video.srcObject) {
    enableWebcamButton.innerText = 'Initializing...';
  }
  document.querySelector('.viewport')?.classList.add('loading-model');
  isWorkerReady = false;

  // Tell worker to init
  // @ts-ignore
  const baseUrl = import.meta.env.BASE_URL;
  worker?.postMessage({
    type: 'INIT',
    modelAssetPath: models[currentModel],
    delegate: currentDelegate,
    scoreThreshold: scoreThreshold,
    maxResults: maxResults,
    runningMode: runningMode,
    baseUrl: baseUrl
  });
}

function setupUI() {
  // Top-level View Tabs (Webcam vs Image)
  const tabWebcam = document.getElementById('tab-webcam')!;
  const tabImage = document.getElementById('tab-image')!;
  const viewWebcam = document.getElementById('view-webcam')!;
  const viewImage = document.getElementById('view-image')!;

  // Determine initial mode from localStorage or default to IMAGE (since it's lighter)
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

      // Trigger detection on current image immediately
      if (isWorkerReady) {
        const testImage = document.getElementById('test-image') as HTMLImageElement;
        if (testImage && testImage.src) {
          detectImage(testImage);
        }
      }
    }
  };

  // Set initial state
  if (initialMode === 'VIDEO') {
    switchView('VIDEO');
  } else {
    switchView('IMAGE');
  }

  tabWebcam.addEventListener('click', () => switchView('VIDEO'));
  tabImage.addEventListener('click', () => switchView('IMAGE'));

  // Define enableCam before using it if possible, or hoist it?
  // enableCam is defined below so it's hoisted.

  // Webcam Button
  enableWebcamButton.addEventListener('click', enableCam);

  // Auto-Start Check

  // Model Selection Tabs
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
      // Re-select standard model if needed or just leave as is
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

  // Options
  const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
  const modelUpload = document.getElementById('model-upload') as HTMLInputElement;
  const uploadStatus = document.getElementById('upload-status')!;

  modelSelect.addEventListener('change', async (e) => {
    currentModel = (e.target as HTMLSelectElement).value;
    modelUpload.value = ''; // Clear upload
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

  const maxResultsInput = document.getElementById('max-results') as HTMLInputElement;
  const maxResultsValue = document.getElementById('max-results-value')!;
  maxResultsInput.addEventListener('input', () => {
    maxResults = parseInt(maxResultsInput.value);
    maxResultsValue.innerText = maxResults.toString();
    worker?.postMessage({ type: 'SET_OPTIONS', maxResults });
    // Trigger re-detection if in image mode
    if (runningMode === 'IMAGE') {
      const testImage = document.getElementById('test-image') as HTMLImageElement;
      if (testImage.src) detectImage(testImage);
    }
  });

  const scoreThresholdInput = document.getElementById('score-threshold') as HTMLInputElement;
  const scoreThresholdValue = document.getElementById('score-threshold-value')!;
  scoreThresholdInput.addEventListener('input', () => {
    scoreThreshold = parseFloat(scoreThresholdInput.value);
    scoreThresholdValue.innerText = scoreThreshold.toString();
    worker?.postMessage({ type: 'SET_OPTIONS', scoreThreshold });
    if (runningMode === 'IMAGE') {
      const testImage = document.getElementById('test-image') as HTMLImageElement;
      if (testImage.src) detectImage(testImage);
    }
  });

  // Webcam
  enableWebcamButton.addEventListener('click', enableCam);

  // Image Upload
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
        imagePreviewContainer.style.display = ''; // Let CSS grid handle it
        dropzoneContent.style.display = 'none';
        const reUploadBtn = document.getElementById('re-upload-btn');
        if (reUploadBtn) reUploadBtn.style.display = 'flex';

        testImage.onload = () => {
          detectImage(testImage);
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
    enableWebcamButton.innerText = 'Loading...';
    enableWebcamButton.disabled = true;
    await initializeDetector();
  });

  // Sync initial value from state (e.g. URL override)
  if (currentDelegate) {
    delegateSelect.value = currentDelegate;
  }
}

async function detectImage(image: HTMLImageElement) {
  if (!worker) return;

  // Ensure running mode is IMAGE
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

function displayImageDetections(result: ObjectDetectorResult, image: HTMLImageElement) {
  const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
  const ctx = imageCanvas.getContext('2d')!;

  imageCanvas.width = image.naturalWidth;
  imageCanvas.height = image.naturalHeight;
  // Scale canvas to match image display size? 
  // Actually the image is styled with max-width: 100%. 
  // We should probably rely on the canvas having the same dimensions as the natural image
  // and let CSS scale it down if needed, OR we match the displayed size.
  // For simplicity, let's match natural size and scale with CSS.

  imageCanvas.style.width = `${image.width}px`;
  imageCanvas.style.height = `${image.height}px`;

  ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);

  if (result.detections) {
    for (let detection of result.detections) {
      drawDetection(ctx, detection, false); // No mirroring for images
    }
    // Expose results for testing
    const resultsEl = document.createElement('div');
    resultsEl.id = 'test-results';
    resultsEl.style.display = 'none';
    resultsEl.textContent = JSON.stringify(result.detections);

    // Remove old results if exist
    const oldResults = document.getElementById('test-results');
    if (oldResults) oldResults.remove();

    document.body.appendChild(resultsEl);
  }
}

async function enableCam() {
  if (!worker) return;

  if (video.paused) {
    enableWebcamButton.innerText = 'Disable Webcam';
    const constraints = { video: true };

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = stream;
      video.addEventListener('loadeddata', predictWebcam);

      // Make sure mode is VIDEO
      runningMode = 'VIDEO';
      worker.postMessage({ type: 'SET_OPTIONS', runningMode: 'VIDEO' });

    } catch (err) {
      console.error(err);
    }
  } else {
    stopCam();
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

  // Wait for worker to finish initializing (e.g., during delegate or model switch)
  if (!isWorkerReady) {
    animationFrameId = window.requestAnimationFrame(predictWebcam);
    return;
  }

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    // We send the current frame to the worker. 
    // The worker will request the next frame by telling us it's done via `DETECT_RESULT`
    try {
      const bitmap = await createImageBitmap(video);
      worker?.postMessage({
        type: 'DETECT_VIDEO',
        bitmap: bitmap,
        timestampMs: performance.now()
      }, [bitmap]);
    } catch (e) {
      console.error("Failed to create ImageBitmap from video", e);
      // Try again next frame
      animationFrameId = window.requestAnimationFrame(predictWebcam);
    }
  } else {
    // If video hasn't advanced, just wait for next frame
    animationFrameId = window.requestAnimationFrame(predictWebcam);
  }
}

function displayVideoDetections(result: ObjectDetectorResult) {
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
  // if (mirror && videoWidth) {
  //   x = videoWidth - width - originX;
  // }

  ctx.strokeRect(x, originY, width, height);

  ctx.fillStyle = '#007f8b';
  ctx.font = '16px sans-serif';

  const category = detection.categories[0];
  const score = category.score ? Math.round(category.score * 100) : 0;
  const labelText = `${category.categoryName} - ${score}%`;

  const textWidth = ctx.measureText(labelText).width;

  if (mirror) {
    ctx.save();
    // Translate to where we want the text box start to be (flipped)
    // We want the left edge of the box at x
    // If we simply draw at x, it will be flipped around the origin? No.
    // We need to flip the text locally.
    // Center of the text box:
    const centerX = x + (textWidth + 10) / 2;
    const centerY = originY + 12.5; // Half of 25 height

    ctx.translate(centerX, centerY);
    ctx.scale(-1, 1);
    ctx.translate(-centerX, -centerY);

    // Now draw normally relative to x
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

export function cleanupObjectDetection() {
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  if (video && video.srcObject) {
    const stream = video.srcObject as MediaStream;
    stream.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }

  if (worker) {
    worker.postMessage({ type: 'CLEANUP' });
    // We optionally terminate the worker or keep it alive, let's keep it alive for faster resumes, 
    // but tell it to close the internal mediapipe instance
  }

  // Also clear canvas
  if (canvasCtx && canvasElement) {
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  }
}