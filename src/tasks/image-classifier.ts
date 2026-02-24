import { FilesetResolver, ImageClassifier } from '@mediapipe/tasks-vision';
import {
  bindCameraCapture,
  bindImageUpload,
  drawBaseImage,
  ensureImageReady,
  getVisionWasmBasePath,
  renderSimpleCategories,
  resolveAssetPath,
  setInferenceTime,
  setStatus,
} from './vision-task-utils';

type ImageVideoResult = ReturnType<ImageClassifier['classifyForVideo']>;

const MODELS: Record<string, string> = {
  efficientnet_lite0:
    'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite',
  efficientnet_lite2:
    'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite2/float32/1/efficientnet_lite2.tflite',
};

let classifier: ImageClassifier | undefined;
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
let lastStableVideoResult: ImageVideoResult | null = null;
let lastStableVideoResultMs = 0;
let lastResultsRenderMs = 0;
let lastResultsSignature = '';
const FLICKER_HOLD_MS = 600;
const RESULTS_RENDER_INTERVAL_MS = 180;

function template() {
  return `
    <div class="task-container">
      <div class="controls-panel">
        <div class="section-title">Model Selection</div>
        <div class="select-wrapper">
          <select id="model-select">
            <option value="efficientnet_lite0">EfficientNet Lite0</option>
            <option value="efficientnet_lite2">EfficientNet Lite2</option>
          </select>
        </div>

        <div class="divider"></div>

        <div class="section-title">Settings</div>
        <div class="control-group">
          <div class="control-label"><span>Delegate</span></div>
          <div class="select-wrapper">
            <select id="delegate-select">
              <option value="GPU">GPU</option>
              <option value="CPU">CPU</option>
            </select>
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
        <div id="task-results" class="status-message">No results</div>
      </div>

      <div class="output-panel">
        <div class="output-header">
          <h2>Image Classification</h2>
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

async function initClassifier(modelPath: string, delegate: 'CPU' | 'GPU'): Promise<void> {
  const vision = await FilesetResolver.forVisionTasks(getVisionWasmBasePath());

  if (classifier) {
    classifier.close();
    classifier = undefined;
  }

  try {
    classifier = await ImageClassifier.createFromOptions(vision, {
      baseOptions: { modelAssetPath: resolveAssetPath(modelPath), delegate },
      runningMode,
      maxResults: 5,
    });
  } catch (err) {
    if (delegate !== 'GPU') throw err;
    const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement | null;
    if (delegateSelect) delegateSelect.value = 'CPU';
    classifier = await ImageClassifier.createFromOptions(vision, {
      baseOptions: { modelAssetPath: resolveAssetPath(modelPath), delegate: 'CPU' },
      runningMode,
      maxResults: 5,
    });
    setStatus('GPU unavailable, using CPU');
    return;
  }

  setStatus('Ready');
}

async function runImageClassification(image: HTMLImageElement, canvas: HTMLCanvasElement, resultsEl: HTMLElement): Promise<void> {
  if (!classifier || disposed) return;

  await ensureImageReady(image);
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  drawBaseImage(image, canvas, ctx);

  const start = performance.now();
  const result = classifier.classify(image);
  setInferenceTime(start);
  const categories = result.classifications[0]?.categories ?? [];
  renderSimpleCategories(resultsEl, categories, 5);
  setStatus('Done');
}

function renderVideoResults(result: ReturnType<ImageClassifier['classifyForVideo']>, resultsEl: HTMLElement) {
  const categories = result.classifications[0]?.categories ?? [];
  renderSimpleCategories(resultsEl, categories, 5);
}

function maybeRenderVideoResults(result: ImageVideoResult, resultsEl: HTMLElement, force = false) {
  const now = performance.now();
  const categories = result.classifications[0]?.categories ?? [];
  const signature = JSON.stringify(
    categories.slice(0, 5).map(category => ({
      n: category.categoryName,
      s: Math.round((category.score ?? 0) * 20) * 5,
    })),
  );

  if (!force && now - lastResultsRenderMs < RESULTS_RENDER_INTERVAL_MS && signature === lastResultsSignature) {
    return;
  }

  lastResultsRenderMs = now;
  lastResultsSignature = signature;
  renderVideoResults(result, resultsEl);
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
      await classifier?.setOptions({ runningMode: 'VIDEO' });
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
  if (!classifier || !video.srcObject || runningMode !== 'VIDEO') return;

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
    const result = classifier.classifyForVideo(video, nowMs);
    setInferenceTime(start);

    let resultToRender = result;
    const categories = result.classifications[0]?.categories ?? [];
    if (categories.length > 0) {
      lastStableVideoResult = result;
      lastStableVideoResultMs = nowMs;
    } else if (lastStableVideoResult && nowMs - lastStableVideoResultMs < FLICKER_HOLD_MS) {
      resultToRender = lastStableVideoResult;
    }

    maybeRenderVideoResults(resultToRender, resultsEl);
  }

  animationFrameId = window.requestAnimationFrame(() => {
    void predictWebcam(resultsEl);
  });
}

export async function setupImageClassifier(container: HTMLElement) {
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
    if (disposed) return;
    setStatus('Loading model...');
    await initClassifier(MODELS[modelSelect.value], delegateSelect.value as 'CPU' | 'GPU');
    if (runningMode === 'IMAGE') {
      await runImageClassification(image, imageCanvas, resultsEl);
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
      await classifier?.setOptions({ runningMode: 'VIDEO' });
      await enableCam(resultsEl);
    } else {
      tabWebcam.classList.remove('active');
      tabImage.classList.add('active');
      viewWebcam.classList.remove('active');
      viewImage.classList.add('active');
      await classifier?.setOptions({ runningMode: 'IMAGE' });
      stopCam();
      await runImageClassification(image, imageCanvas, resultsEl);
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
    if (video.srcObject) {
      stopCam();
    } else {
      void enableCam(resultsEl);
    }
  });

  disposeUpload = bindImageUpload(imageUpload, image, async () => {
    await runImageClassification(image, imageCanvas, resultsEl);
  });
  disposeCameraCapture = bindCameraCapture(cameraCaptureBtn, image, async () => {
    await runImageClassification(image, imageCanvas, resultsEl);
  });

  image.addEventListener('load', () => {
    if (runningMode === 'IMAGE') {
      void runImageClassification(image, imageCanvas, resultsEl);
    }
  });

  await refresh();

  const initialMode = (localStorage.getItem('mediapipe-running-mode') as 'IMAGE' | 'VIDEO') || 'IMAGE';
  await switchView(initialMode);
}

export function cleanupImageClassifier() {
  disposed = true;
  stopCam();
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  disposeUpload?.();
  disposeUpload = undefined;
  disposeCameraCapture?.();
  disposeCameraCapture = undefined;
  if (classifier) {
    classifier.close();
    classifier = undefined;
  }
  lastStableVideoResult = null;
  lastStableVideoResultMs = 0;
  lastResultsRenderMs = 0;
  lastResultsSignature = '';
}
