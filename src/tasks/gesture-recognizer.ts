import { FilesetResolver, GestureRecognizer } from '@mediapipe/tasks-vision';
import {
  bindCameraCapture,
  bindImageUpload,
  drawBaseImage,
  ensureImageReady,
  getVisionWasmBasePath,
  resolveAssetPath,
  setInferenceTime,
  setStatus,
} from './vision-task-utils';

type GestureResult = ReturnType<GestureRecognizer['recognize']>;

const MODELS: Record<string, string> = {
  gesture_recognizer:
    'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task',
};

let numHands = 2;
let minHandDetectionConfidence = 0.5;
let minHandPresenceConfidence = 0.5;
let minTrackingConfidence = 0.5;

let recognizer: GestureRecognizer | undefined;
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
let lastStableVideoResult: GestureResult | null = null;
let lastStableVideoResultMs = 0;
let smoothedLandmarks: GestureResult['landmarks'] = [];
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
          <select id="model-select"><option value="gesture_recognizer">Gesture Recognizer</option></select>
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
          <div class="control-label">
            <span>Demo Num Hands</span>
            <span id="num-hands-value" class="value-badge">2</span>
          </div>
          <input type="range" id="num-hands" min="1" max="2" step="1" value="2" class="range-slider">
        </div>

        <div class="control-group">
          <div class="control-label">
            <span>Minimum Hand Detection Confidence</span>
            <span id="min-detection-value" class="value-badge">50%</span>
          </div>
          <input type="range" id="min-detection" min="0.01" max="0.99" step="0.01" value="0.50" class="range-slider">
        </div>

        <div class="control-group">
          <div class="control-label">
            <span>Minimum Hand Presence Confidence</span>
            <span id="min-presence-value" class="value-badge">50%</span>
          </div>
          <input type="range" id="min-presence" min="0.01" max="0.99" step="0.01" value="0.50" class="range-slider">
        </div>

        <div class="control-group">
          <div class="control-label">
            <span>Minimum Tracking Confidence</span>
            <span id="min-tracking-value" class="value-badge">50%</span>
          </div>
          <input type="range" id="min-tracking" min="0.01" max="0.99" step="0.01" value="0.50" class="range-slider">
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
        <div id="task-results" class="status-message">No gestures yet</div>
      </div>

      <div class="output-panel">
        <div class="output-header">
          <h2>Gesture Recognition</h2>
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

async function initRecognizer(modelPath: string, delegate: 'CPU' | 'GPU') {
  const vision = await FilesetResolver.forVisionTasks(getVisionWasmBasePath());
  if (recognizer) {
    recognizer.close();
    recognizer = undefined;
  }

  try {
    recognizer = await GestureRecognizer.createFromOptions(vision, {
      baseOptions: { modelAssetPath: resolveAssetPath(modelPath), delegate },
      runningMode,
      numHands,
      minHandDetectionConfidence,
      minHandPresenceConfidence,
      minTrackingConfidence,
    });
  } catch (err) {
    if (delegate !== 'GPU') throw err;
    const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement | null;
    if (delegateSelect) delegateSelect.value = 'CPU';
    recognizer = await GestureRecognizer.createFromOptions(vision, {
      baseOptions: { modelAssetPath: resolveAssetPath(modelPath), delegate: 'CPU' },
      runningMode,
      numHands,
      minHandDetectionConfidence,
      minHandPresenceConfidence,
      minTrackingConfidence,
    });
    setStatus('GPU unavailable, using CPU');
    return;
  }

  setStatus('Ready');
}

function drawGestureLandmarks(
  ctx: CanvasRenderingContext2D,
  landmarkSets: GestureResult['landmarks'],
) {
  for (const landmarks of landmarkSets) {
    // Bright connector outline
    ctx.save();
    ctx.strokeStyle = '#00f5ff';
    ctx.lineWidth = 5;
    ctx.shadowColor = '#00f5ff';
    ctx.shadowBlur = 12;
    for (const connection of GestureRecognizer.HAND_CONNECTIONS) {
      const start = landmarks[connection.start];
      const end = landmarks[connection.end];
      if (!start || !end) continue;
      ctx.beginPath();
      ctx.moveTo(start.x * ctx.canvas.width, start.y * ctx.canvas.height);
      ctx.lineTo(end.x * ctx.canvas.width, end.y * ctx.canvas.height);
      ctx.stroke();
    }
    ctx.restore();

    // High contrast joints
    for (const landmark of landmarks) {
      const x = landmark.x * ctx.canvas.width;
      const y = landmark.y * ctx.canvas.height;

      ctx.beginPath();
      ctx.fillStyle = '#ffffff';
      ctx.arc(x, y, 4.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.beginPath();
      ctx.fillStyle = '#ff1e00';
      ctx.arc(x, y, 2.6, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function renderGestureResult(result: GestureResult, resultsEl: HTMLElement) {
  const blocks: string[] = [];
  for (let i = 0; i < result.gestures.length; i++) {
    const gesture = result.gestures[i]?.[0];
    const handed = result.handedness[i]?.[0];
    blocks.push(`
      <div style="margin-bottom:10px;">
        <div><strong>Hand ${i + 1}</strong></div>
        <div>Gesture: ${gesture?.categoryName ?? 'Unknown'} (${Math.round((gesture?.score ?? 0) * 100)}%)</div>
        <div>Handedness: ${handed?.categoryName ?? 'Unknown'} (${Math.round((handed?.score ?? 0) * 100)}%)</div>
      </div>
    `);
  }
  const html = blocks.join('') || '<div>No hands detected</div>';
  resultsEl.innerHTML = html;
}

function maybeRenderGestureResult(
  result: GestureResult,
  resultsEl: HTMLElement,
  force = false,
) {
  const now = performance.now();
  const signature = JSON.stringify(
    result.gestures.map((hands, i) => ({
      i,
      g: hands[0]?.categoryName ?? '',
      gs: Math.round((hands[0]?.score ?? 0) * 20) * 5,
      h: result.handedness[i]?.[0]?.categoryName ?? '',
      hs: Math.round((result.handedness[i]?.[0]?.score ?? 0) * 20) * 5,
    })),
  );

  if (!force && now - lastResultsRenderMs < RESULTS_RENDER_INTERVAL_MS && signature === lastResultsSignature) {
    return;
  }

  lastResultsRenderMs = now;
  lastResultsSignature = signature;
  renderGestureResult(result, resultsEl);
}

function normalizeResultOrder(result: GestureResult): GestureResult {
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
    gestures: indexed.map(({ i }) => result.gestures[i] ?? []),
    landmarks: indexed.map(({ i }) => result.landmarks[i] ?? []),
    handedness: indexed.map(({ i }) => result.handedness[i] ?? []),
    worldLandmarks: indexed.map(({ i }) => result.worldLandmarks[i] ?? []),
  };
}

function smoothLandmarks(
  current: GestureResult['landmarks'],
): GestureResult['landmarks'] {
  if (smoothedLandmarks.length === 0 || smoothedLandmarks.length !== current.length) {
    smoothedLandmarks = current.map(hand => hand.map(l => ({ ...l })));
    return smoothedLandmarks;
  }

  smoothedLandmarks = current.map((hand, handIndex) => {
    const previousHand = smoothedLandmarks[handIndex];
    if (!previousHand || previousHand.length !== hand.length) {
      return hand.map(l => ({ ...l }));
    }

    return hand.map((landmark, landmarkIndex) => {
      const prev = previousHand[landmarkIndex];
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

async function runImageRecognition(image: HTMLImageElement, canvas: HTMLCanvasElement, resultsEl: HTMLElement) {
  if (!recognizer || disposed) return;
  await ensureImageReady(image);

  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  drawBaseImage(image, canvas, ctx);

  const start = performance.now();
  const result = normalizeResultOrder(recognizer.recognize(image));
  setInferenceTime(start);

  if (result.landmarks.length > 0) {
    drawGestureLandmarks(ctx, result.landmarks);
  }

  maybeRenderGestureResult(result, resultsEl, true);
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
      await recognizer?.setOptions({ runningMode: 'VIDEO' });
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
  if (!recognizer || !video.srcObject || runningMode !== 'VIDEO') return;

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
    const result = normalizeResultOrder(recognizer.recognizeForVideo(video, nowMs));
    setInferenceTime(start);

    let resultToRender = result;
    if (result.landmarks.length > 0) {
      lastStableVideoResult = result;
      lastStableVideoResultMs = nowMs;
      smoothedLandmarks = smoothLandmarks(result.landmarks);
    } else if (lastStableVideoResult && nowMs - lastStableVideoResultMs < FLICKER_HOLD_MS) {
      resultToRender = lastStableVideoResult;
      if (smoothedLandmarks.length === 0 && resultToRender.landmarks.length > 0) {
        smoothedLandmarks = resultToRender.landmarks.map(hand => hand.map(l => ({ ...l })));
      }
    } else {
      smoothedLandmarks = [];
    }

    if (smoothedLandmarks.length > 0) {
      drawGestureLandmarks(webcamCtx, smoothedLandmarks);
    }
    maybeRenderGestureResult(resultToRender, resultsEl);
  }

  animationFrameId = window.requestAnimationFrame(() => {
    void predictWebcam(resultsEl);
  });
}

export async function setupGestureRecognizer(container: HTMLElement) {
  disposed = false;
  container.innerHTML = template();

  const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
  const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
  const imageUpload = document.getElementById('image-upload') as HTMLInputElement;
  const numHandsInput = document.getElementById('num-hands') as HTMLInputElement;
  const numHandsValue = document.getElementById('num-hands-value') as HTMLElement;
  const minDetectionInput = document.getElementById('min-detection') as HTMLInputElement;
  const minDetectionValue = document.getElementById('min-detection-value') as HTMLElement;
  const minPresenceInput = document.getElementById('min-presence') as HTMLInputElement;
  const minPresenceValue = document.getElementById('min-presence-value') as HTMLElement;
  const minTrackingInput = document.getElementById('min-tracking') as HTMLInputElement;
  const minTrackingValue = document.getElementById('min-tracking-value') as HTMLElement;
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

  const updateOptionLabels = () => {
    numHandsValue.textContent = `${numHands}`;
    minDetectionValue.textContent = `${Math.round(minHandDetectionConfidence * 100)}%`;
    minPresenceValue.textContent = `${Math.round(minHandPresenceConfidence * 100)}%`;
    minTrackingValue.textContent = `${Math.round(minTrackingConfidence * 100)}%`;
  };

  const applyRuntimeOptions = async () => {
    if (!recognizer) return;
    await recognizer.setOptions({
      numHands,
      minHandDetectionConfidence,
      minHandPresenceConfidence,
      minTrackingConfidence,
    });
    if (runningMode === 'IMAGE') {
      await runImageRecognition(image, imageCanvas, resultsEl);
    }
  };

  const refresh = async () => {
    setStatus('Loading model...');
    await initRecognizer(MODELS[modelSelect.value], delegateSelect.value as 'CPU' | 'GPU');
    if (runningMode === 'IMAGE') {
      await runImageRecognition(image, imageCanvas, resultsEl);
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
      await recognizer?.setOptions({ runningMode: 'VIDEO' });
      await enableCam(resultsEl);
    } else {
      tabWebcam.classList.remove('active');
      tabImage.classList.add('active');
      viewWebcam.classList.remove('active');
      viewImage.classList.add('active');
      await recognizer?.setOptions({ runningMode: 'IMAGE' });
      stopCam();
      await runImageRecognition(image, imageCanvas, resultsEl);
    }
  };

  modelSelect.addEventListener('change', () => void refresh());
  delegateSelect.addEventListener('change', () => void refresh());
  numHandsInput.addEventListener('input', () => {
    numHands = parseInt(numHandsInput.value, 10);
    updateOptionLabels();
    void applyRuntimeOptions();
  });
  minDetectionInput.addEventListener('input', () => {
    minHandDetectionConfidence = parseFloat(minDetectionInput.value);
    updateOptionLabels();
    void applyRuntimeOptions();
  });
  minPresenceInput.addEventListener('input', () => {
    minHandPresenceConfidence = parseFloat(minPresenceInput.value);
    updateOptionLabels();
    void applyRuntimeOptions();
  });
  minTrackingInput.addEventListener('input', () => {
    minTrackingConfidence = parseFloat(minTrackingInput.value);
    updateOptionLabels();
    void applyRuntimeOptions();
  });
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
    await runImageRecognition(image, imageCanvas, resultsEl);
  });
  disposeCameraCapture = bindCameraCapture(cameraCaptureBtn, image, async () => {
    await runImageRecognition(image, imageCanvas, resultsEl);
  });

  image.addEventListener('load', () => {
    if (runningMode === 'IMAGE') {
      void runImageRecognition(image, imageCanvas, resultsEl);
    }
  });

  await refresh();
  updateOptionLabels();

  const initialMode = (localStorage.getItem('mediapipe-running-mode') as 'IMAGE' | 'VIDEO') || 'IMAGE';
  await switchView(initialMode);
}

export function cleanupGestureRecognizer() {
  disposed = true;
  stopCam();
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  disposeUpload?.();
  disposeUpload = undefined;
  disposeCameraCapture?.();
  disposeCameraCapture = undefined;
  if (recognizer) {
    recognizer.close();
    recognizer = undefined;
  }
  smoothedLandmarks = [];
  lastStableVideoResult = null;
  lastStableVideoResultMs = 0;
  lastResultsRenderMs = 0;
  lastResultsSignature = '';
}
