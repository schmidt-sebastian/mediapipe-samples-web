/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {
  HolisticLandmarkerResult,
  DrawingUtils,
  FaceLandmarker,
  PoseLandmarker,
  HandLandmarker
} from '@mediapipe/tasks-vision';

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
let currentModel = 'holistic_landmarker_lite';
let modelSelector: ModelSelector;
let currentDelegate: 'CPU' | 'GPU' = 'GPU';

const models: Record<string, string> = {
  'holistic_landmarker_lite': 'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task',
};

// @ts-ignore
// @ts-ignore
import template from '../templates/holistic-landmarker.html?raw';
import { ViewToggle } from '../components/view-toggle';
import { ModelSelector } from '../components/model-selector';

export async function setupHolisticLandmarker(container: HTMLElement) {
  container.innerHTML = template;

  video = document.getElementById('webcam') as HTMLVideoElement;
  canvasElement = document.getElementById('output_canvas') as HTMLCanvasElement;
  canvasCtx = canvasElement.getContext('2d')!;
  enableWebcamButton = document.getElementById('webcamButton') as HTMLButtonElement;

  initWorker();
  setupUI();
  await initializeDetector();
}

// @ts-ignore
import HolisticLandmarkerWorker from '../workers/holistic-landmarker.worker.ts?worker';

function initWorker() {
  if (!worker) {
    worker = new HolisticLandmarkerWorker();
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
      modelSelector?.hideProgress();
      document.querySelector('.viewport')?.classList.remove('loading-model');
      isWorkerReady = true;
      if (video && video.srcObject) {
        enableWebcamButton.innerText = 'Disable Webcam';
        enableWebcamButton.disabled = false;
      } else if (enableWebcamButton.innerText !== 'Starting...') {
        enableWebcamButton.innerText = 'Enable Webcam';
        enableWebcamButton.disabled = false;
      }
      updateStatus('Ready');

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
      console.error('Worker error:', event.data.error);
      updateStatus(`Error: ${event.data.error}`);
      break;
  }
}

function triggerImageDetection(image: HTMLImageElement) {
  if (image.complete && image.naturalWidth > 0) {
    detectImage(image);
  } else {
    image.onload = () => {
      if (image.naturalWidth > 0) {
        detectImage(image);
      }
    };
  }
}

async function initializeDetector() {
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
      const isWebcamActive = localStorage.getItem('mediapipe-webcam-active') === 'true';
      if (isWebcamActive) {
        enableCam();
      }
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

  enableWebcamButton.addEventListener('click', toggleCam);

  modelSelector = new ModelSelector(
    'model-selector-container',
    [
      { label: 'Holistic Landmarker', value: 'holistic_landmarker_lite', isDefault: true }
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

  const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
  delegateSelect.addEventListener('change', async () => {
    currentDelegate = delegateSelect.value as 'GPU' | 'CPU';
    await initializeDetector();
  });
  delegateSelect.value = currentDelegate;

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

        triggerImageDetection(testImage);
      };
      reader.readAsDataURL(file);
    }
  });
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

function displayImageResult(result: HolisticLandmarkerResult) {
  const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
  const testImage = document.getElementById('test-image') as HTMLImageElement;
  const ctx = imageCanvas.getContext('2d')!;

  imageCanvas.width = testImage.naturalWidth;
  imageCanvas.height = testImage.naturalHeight;
  // imageCanvas.style.width = '100%';
  // imageCanvas.style.height = 'auto';

  ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);

  drawResults(ctx, result);
}

function displayVideoResult(result: HolisticLandmarkerResult) {
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  drawResults(canvasCtx, result);
  canvasCtx.restore();
}

function drawResults(ctx: CanvasRenderingContext2D, result: HolisticLandmarkerResult) {
  const drawingUtils = new DrawingUtils(ctx);

  // Face Landmarks
  if (result.faceLandmarks && result.faceLandmarks.length > 0) {
    for (const landmarks of result.faceLandmarks) {
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: '#C0C0C070', lineWidth: 1 });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE, { color: '#FF3030' });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW, { color: '#FF3030' });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYE, { color: '#30FF30' });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW, { color: '#30FF30' });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: '#E0E0E0' });
      drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: '#E0E0E0' });
    }
  }

  // Pose Landmarks
  if (result.poseLandmarks && result.poseLandmarks.length > 0) {
    for (const landmarks of result.poseLandmarks) {
      drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: '#FFFFFF' });
      drawingUtils.drawLandmarks(landmarks, { color: '#FF0000', radius: 1 });
    }
  }

  // Hand Landmarks
  if (result.leftHandLandmarks && result.leftHandLandmarks.length > 0) {
    for (const landmarks of result.leftHandLandmarks) {
      drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: '#CC0000', lineWidth: 5 });
      drawingUtils.drawLandmarks(landmarks, { color: '#00FF00', lineWidth: 2 });
    }
  }

  if (result.rightHandLandmarks && result.rightHandLandmarks.length > 0) {
    for (const landmarks of result.rightHandLandmarks) {
      drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, { color: '#00CC00', lineWidth: 5 });
      drawingUtils.drawLandmarks(landmarks, { color: '#FF0000', lineWidth: 2 });
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
  if (video.srcObject) stopCam();
  else enableCam();
}

function stopCam(persistState = true) {
  if (video.srcObject) {
    const stream = video.srcObject as MediaStream;
    stream.getTracks().forEach(t => t.stop());
    video.srcObject = null;
    document.getElementById('webcam-placeholder')?.classList.remove('hidden');
    enableWebcamButton.innerText = 'Enable Webcam';
    cancelAnimationFrame(animationFrameId);
    if (persistState) {
      localStorage.setItem('mediapipe-webcam-active', 'false');
    }
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

export function cleanupHolisticLandmarker() {
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  stopCam(false);
  if (worker) {
    worker.postMessage({ type: 'CLEANUP' });
    worker.terminate();
    worker = undefined;
  }
  isWorkerReady = false;
  if (canvasCtx && canvasElement) canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
}
