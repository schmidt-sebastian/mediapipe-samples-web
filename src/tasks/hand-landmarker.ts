import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import {
  bindCameraCapture,
  bindImageUpload,
  drawBaseImage,
  drawLandmarkSets,
  ensureImageReady,
  getVisionWasmBasePath,
  resolveAssetPath,
  setInferenceTime,
  setStatus,
} from './vision-task-utils';

type HandResult = ReturnType<HandLandmarker['detect']>;

const MODELS: Record<string, string> = {
  hand_landmarker:
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
};

let landmarker: HandLandmarker | undefined;
let disposeUpload: (() => void) | undefined;
let disposeCameraCapture: (() => void) | undefined;
let disposed = false;

let runningMode: 'IMAGE' | 'VIDEO' = 'IMAGE';
let video: HTMLVideoElement;
let webcamCanvas: HTMLCanvasElement;
let webcamCtx: CanvasRenderingContext2D;
let enableWebcamButton: HTMLButtonElement;
let animationFrameId = 0;
let lastVideoTime = -1;
let lastStableVideoResult: HandResult | null = null;
let lastStableVideoResultMs = 0;
let smoothedLandmarks: HandResult['landmarks'] = [];
let lastResultsRenderMs = 0;
let lastResultsSignature = '';
const FLICKER_HOLD_MS = 600;
const RESULTS_RENDER_INTERVAL_MS = 180;
const LANDMARK_NEW_WEIGHT = 0.32;

function template() {
  return `
    <div class="task-container">
      <div class="controls-panel">
        <div class="section-title">Model Selection</div>
        <div class="select-wrapper">
          <select id="model-select"><option value="hand_landmarker">Hand Landmarker</option></select>
        </div>

        <div class="divider"></div>

        <div class="section-title">Settings</div>
        <div class="control-group">
          <div class="control-label"><span>Delegate</span></div>
          <div class="select-wrapper">
            <select id="delegate-select"><option value="GPU">GPU</option><option value="CPU">CPU</option></select>
          </div>
        </div>

        <div class="control-group">
          <div class="control-label"><span>Image Source</span></div>
          <input id="image-upload" type="file" accept="image/*" style="display:none;" />
          <button id="re-upload-btn" class="action-button" type="button" style="width:100%; margin-bottom:8px;">
            <span class="material-icons">upload</span> Upload Image
          </button>
          <button id="camera-capture-btn" class="action-button" type="button" style="width:100%;">
            <span class="material-icons">photo_camera</span> Use Built-in Camera
          </button>
        </div>

        <div class="divider"></div>

        <div id="status-message" class="status-message">Initializing...</div>
        <div id="inference-time" class="inference-time">Inference Time: - ms</div>

        <div class="divider"></div>
        <div class="section-title">Results</div>
        <div id="task-results" class="status-message">No hands yet</div>
      </div>

      <div class="output-panel">
        <div class="output-header">
          <h2>Hand Landmark Detection</h2>
          <div class="view-tabs">
            <button id="tab-webcam" class="view-tab">Webcam</button>
            <button id="tab-image" class="view-tab active">Image</button>
          </div>
        </div>
        <div class="viewport">
          <div id="view-webcam" class="view-content">
            <div class="cam-container">
              <button id="webcamButton" class="action-button">
                <span class="material-icons">videocam</span> Enable Webcam
              </button>
              <div class="video-wrapper">
                <video id="webcam" autoplay playsinline muted></video>
                <canvas id="output_canvas"></canvas>
              </div>
            </div>
          </div>

          <div id="view-image" class="view-content active">
            <div class="upload-dropzone">
              <div class="dropzone-content" style="display:none;">
                <span class="material-icons large-icon">image</span>
                <p>Upload or capture an image</p>
              </div>
              <div id="image-preview-container" class="preview-container">
                <img id="test-image" src="dog.jpg" crossorigin="anonymous" />
                <canvas id="image-canvas"></canvas>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  `;
}

async function initLandmarker(modelPath: string, delegate: 'CPU' | 'GPU') {
  const vision = await FilesetResolver.forVisionTasks(getVisionWasmBasePath());
  if (landmarker) {
    landmarker.close();
    landmarker = undefined;
  }

  try {
    landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: resolveAssetPath(modelPath), delegate },
      runningMode,
      numHands: 2,
    });
  } catch (err) {
    if (delegate !== 'GPU') throw err;
    const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement | null;
    if (delegateSelect) delegateSelect.value = 'CPU';
    landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: resolveAssetPath(modelPath), delegate: 'CPU' },
      runningMode,
      numHands: 2,
    });
    setStatus('GPU unavailable, using CPU');
    return;
  }

  setStatus('Ready');
}

function renderHandedness(result: HandResult, resultsEl: HTMLElement) {
  const blocks: string[] = [];
  for (let i = 0; i < result.handedness.length; i++) {
    const handed = result.handedness[i]?.[0];
    blocks.push(`<div>Hand ${i + 1}: ${handed?.categoryName ?? 'Unknown'} (${Math.round((handed?.score ?? 0) * 100)}%)</div>`);
  }
  resultsEl.innerHTML = blocks.join('') || '<div>No hands detected</div>';
}

function maybeRenderHandedness(result: HandResult, resultsEl: HTMLElement, force = false) {
  const now = performance.now();
  const signature = JSON.stringify(
    result.handedness.map((hands, i) => ({
      i,
      h: hands[0]?.categoryName ?? '',
      hs: Math.round((hands[0]?.score ?? 0) * 20) * 5,
    })),
  );

  if (!force && now - lastResultsRenderMs < RESULTS_RENDER_INTERVAL_MS && signature === lastResultsSignature) {
    return;
  }

  lastResultsRenderMs = now;
  lastResultsSignature = signature;
  renderHandedness(result, resultsEl);
}

function normalizeHandResultOrder(result: HandResult): HandResult {
  const indexed = result.landmarks.map((landmarks, i) => {
    const handed = result.handedness[i]?.[0];
    const label = handed?.categoryName ?? '';
    const centerX =
      landmarks.length > 0 ? landmarks.reduce((sum, point) => sum + point.x, 0) / landmarks.length : 0;
    return { i, label, centerX };
  });

  indexed.sort((a, b) => {
    if (a.label && b.label && a.label !== b.label) {
      return a.label.localeCompare(b.label);
    }
    return a.centerX - b.centerX;
  });

  return {
    ...result,
    landmarks: indexed.map(({ i }) => result.landmarks[i] ?? []),
    handedness: indexed.map(({ i }) => result.handedness[i] ?? []),
    worldLandmarks: indexed.map(({ i }) => result.worldLandmarks[i] ?? []),
  };
}

function smoothHandLandmarks(current: HandResult['landmarks']): HandResult['landmarks'] {
  if (smoothedLandmarks.length === 0 || smoothedLandmarks.length !== current.length) {
    smoothedLandmarks = current.map(hand => hand.map(landmark => ({ ...landmark })));
    return smoothedLandmarks;
  }

  smoothedLandmarks = current.map((hand, handIndex) => {
    const prevHand = smoothedLandmarks[handIndex];
    if (!prevHand || prevHand.length !== hand.length) {
      return hand.map(landmark => ({ ...landmark }));
    }

    return hand.map((landmark, landmarkIndex) => {
      const prev = prevHand[landmarkIndex];
      if (!prev) return { ...landmark };
      return {
        ...landmark,
        x: prev.x * (1 - LANDMARK_NEW_WEIGHT) + landmark.x * LANDMARK_NEW_WEIGHT,
        y: prev.y * (1 - LANDMARK_NEW_WEIGHT) + landmark.y * LANDMARK_NEW_WEIGHT,
        z: prev.z * (1 - LANDMARK_NEW_WEIGHT) + landmark.z * LANDMARK_NEW_WEIGHT,
      };
    });
  });

  return smoothedLandmarks;
}

async function runImageDetection(image: HTMLImageElement, canvas: HTMLCanvasElement, resultsEl: HTMLElement) {
  if (!landmarker || disposed) return;
  await ensureImageReady(image);

  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  drawBaseImage(image, canvas, ctx);

  const start = performance.now();
  const result = normalizeHandResultOrder(landmarker.detect(image));
  setInferenceTime(start);

  if (result.landmarks.length > 0) {
    drawLandmarkSets(ctx, result.landmarks, HandLandmarker.HAND_CONNECTIONS);
  }

  maybeRenderHandedness(result, resultsEl, true);
  setStatus('Done');
}

async function enableCam(resultsEl: HTMLElement) {
  if (video.srcObject) return;

  enableWebcamButton.innerText = 'Starting...';
  enableWebcamButton.disabled = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    const playAndPredict = async () => {
      await video.play();
      runningMode = 'VIDEO';
      await landmarker?.setOptions({ runningMode: 'VIDEO' });
      predictWebcam(resultsEl);
    };

    if (video.readyState >= 2) {
      await playAndPredict();
    } else {
      video.addEventListener('loadeddata', () => {
        void playAndPredict();
      }, { once: true });
    }

    setStatus('Webcam running...');
    enableWebcamButton.innerText = 'Disable Webcam';
    enableWebcamButton.disabled = false;
  } catch (err) {
    console.error(err);
    setStatus('Camera error!');
    enableWebcamButton.innerText = 'Enable Webcam';
    enableWebcamButton.disabled = false;
  }
}

function stopCam() {
  if (!video.srcObject) return;
  const stream = video.srcObject as MediaStream;
  stream.getTracks().forEach((track) => track.stop());
  video.srcObject = null;
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  enableWebcamButton.innerText = 'Enable Webcam';
}

async function predictWebcam(resultsEl: HTMLElement) {
  if (!landmarker || !video.srcObject || runningMode !== 'VIDEO') return;

  if (webcamCanvas.width !== video.videoWidth || webcamCanvas.height !== video.videoHeight) {
    webcamCanvas.width = video.videoWidth;
    webcamCanvas.height = video.videoHeight;
  }

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    webcamCtx.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);
    webcamCtx.drawImage(video, 0, 0, webcamCanvas.width, webcamCanvas.height);

    const start = performance.now();
    const nowMs = performance.now();
    const result = normalizeHandResultOrder(landmarker.detectForVideo(video, nowMs));
    setInferenceTime(start);

    let resultToRender = result;
    if (result.landmarks.length > 0) {
      lastStableVideoResult = result;
      lastStableVideoResultMs = nowMs;
      smoothedLandmarks = smoothHandLandmarks(result.landmarks);
    } else if (lastStableVideoResult && nowMs - lastStableVideoResultMs < FLICKER_HOLD_MS) {
      resultToRender = lastStableVideoResult;
      if (smoothedLandmarks.length === 0 && resultToRender.landmarks.length > 0) {
        smoothedLandmarks = resultToRender.landmarks.map(hand => hand.map(landmark => ({ ...landmark })));
      }
    } else {
      smoothedLandmarks = [];
    }

    if (smoothedLandmarks.length > 0) {
      drawLandmarkSets(webcamCtx, smoothedLandmarks, HandLandmarker.HAND_CONNECTIONS);
    }

    maybeRenderHandedness(resultToRender, resultsEl);
  }

  animationFrameId = window.requestAnimationFrame(() => {
    void predictWebcam(resultsEl);
  });
}

export async function setupHandLandmarker(container: HTMLElement) {
  disposed = false;
  container.innerHTML = template();

  const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
  const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
  const imageUpload = document.getElementById('image-upload') as HTMLInputElement;
  const reUploadBtn = document.getElementById('re-upload-btn') as HTMLButtonElement;
  const cameraCaptureBtn = document.getElementById('camera-capture-btn') as HTMLButtonElement;
  const image = document.getElementById('test-image') as HTMLImageElement;
  const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
  const resultsEl = document.getElementById('task-results') as HTMLElement;

  const tabWebcam = document.getElementById('tab-webcam') as HTMLButtonElement;
  const tabImage = document.getElementById('tab-image') as HTMLButtonElement;
  const viewWebcam = document.getElementById('view-webcam') as HTMLElement;
  const viewImage = document.getElementById('view-image') as HTMLElement;

  video = document.getElementById('webcam') as HTMLVideoElement;
  webcamCanvas = document.getElementById('output_canvas') as HTMLCanvasElement;
  webcamCtx = webcamCanvas.getContext('2d') as CanvasRenderingContext2D;
  enableWebcamButton = document.getElementById('webcamButton') as HTMLButtonElement;

  const refresh = async () => {
    setStatus('Loading model...');
    await initLandmarker(MODELS[modelSelect.value], delegateSelect.value as 'CPU' | 'GPU');
    if (runningMode === 'IMAGE') {
      await runImageDetection(image, imageCanvas, resultsEl);
    }
  };

  const switchView = async (mode: 'IMAGE' | 'VIDEO') => {
    localStorage.setItem('mediapipe-running-mode', mode);
    runningMode = mode;

    if (mode === 'VIDEO') {
      tabWebcam.classList.add('active');
      tabImage.classList.remove('active');
      viewWebcam.classList.add('active');
      viewImage.classList.remove('active');
      await landmarker?.setOptions({ runningMode: 'VIDEO' });
      await enableCam(resultsEl);
    } else {
      tabWebcam.classList.remove('active');
      tabImage.classList.add('active');
      viewWebcam.classList.remove('active');
      viewImage.classList.add('active');
      await landmarker?.setOptions({ runningMode: 'IMAGE' });
      stopCam();
      await runImageDetection(image, imageCanvas, resultsEl);
    }
  };

  modelSelect.addEventListener('change', () => void refresh());
  delegateSelect.addEventListener('change', () => void refresh());
  reUploadBtn.addEventListener('click', () => imageUpload.click());

  tabWebcam.addEventListener('click', () => {
    if (runningMode !== 'VIDEO') void switchView('VIDEO');
  });
  tabImage.addEventListener('click', () => {
    if (runningMode !== 'IMAGE') void switchView('IMAGE');
  });

  enableWebcamButton.addEventListener('click', () => {
    if (video.srcObject) stopCam();
    else void enableCam(resultsEl);
  });

  disposeUpload = bindImageUpload(imageUpload, image, async () => {
    await runImageDetection(image, imageCanvas, resultsEl);
  });
  disposeCameraCapture = bindCameraCapture(cameraCaptureBtn, image, async () => {
    await runImageDetection(image, imageCanvas, resultsEl);
  });

  image.addEventListener('load', () => {
    if (runningMode === 'IMAGE') {
      void runImageDetection(image, imageCanvas, resultsEl);
    }
  });

  await refresh();

  const initialMode = (localStorage.getItem('mediapipe-running-mode') as 'IMAGE' | 'VIDEO') || 'IMAGE';
  await switchView(initialMode);
}

export function cleanupHandLandmarker() {
  disposed = true;
  stopCam();
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  disposeUpload?.();
  disposeUpload = undefined;
  disposeCameraCapture?.();
  disposeCameraCapture = undefined;
  if (landmarker) {
    landmarker.close();
    landmarker = undefined;
  }
  lastStableVideoResult = null;
  lastStableVideoResultMs = 0;
  smoothedLandmarks = [];
  lastResultsRenderMs = 0;
  lastResultsSignature = '';
}
