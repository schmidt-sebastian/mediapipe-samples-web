import { FilesetResolver, InteractiveSegmenter, RegionOfInterest } from '@mediapipe/tasks-vision';
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

const MODELS: Record<string, string> = {
  magic_touch:
    'https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite',
};

let segmenter: InteractiveSegmenter | undefined;
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
let lastRoi: RegionOfInterest = { keypoint: { x: 0.5, y: 0.5 } };

function template() {
  return `
    <div class="task-container">
      <div class="controls-panel">
        <div class="section-title">Model Selection</div>
        <div class="select-wrapper">
          <select id="model-select">
            <option value="magic_touch">Magic Touch</option>
          </select>
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

        <div id="status-message" class="status-message">Click on image/video to pick foreground point.</div>
        <div id="inference-time" class="inference-time">Inference Time: - ms</div>

        <div class="divider"></div>
        <div class="section-title">Results</div>
        <div id="task-results" class="status-message">Tip: click on object to segment.</div>
      </div>

      <div class="output-panel">
        <div class="output-header">
          <h2>Interactive Segmentation</h2>
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

async function initSegmenter(modelPath: string, delegate: 'CPU' | 'GPU') {
  const vision = await FilesetResolver.forVisionTasks(getVisionWasmBasePath());
  if (segmenter) {
    segmenter.close();
    segmenter = undefined;
  }

  try {
    segmenter = await InteractiveSegmenter.createFromOptions(vision, {
      baseOptions: { modelAssetPath: resolveAssetPath(modelPath), delegate },
      outputCategoryMask: true,
      outputConfidenceMasks: false,
    });
  } catch (err) {
    if (delegate !== 'GPU') throw err;
    const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement | null;
    if (delegateSelect) delegateSelect.value = 'CPU';
    segmenter = await InteractiveSegmenter.createFromOptions(vision, {
      baseOptions: { modelAssetPath: resolveAssetPath(modelPath), delegate: 'CPU' },
      outputCategoryMask: true,
      outputConfidenceMasks: false,
    });
    setStatus('GPU unavailable, using CPU');
    return;
  }

  setStatus('Ready. Click image/video to segment.');
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function renderRoiInfo() {
  const resultsEl = document.getElementById('task-results');
  if (resultsEl && lastRoi.keypoint) {
    resultsEl.innerHTML = `<div>ROI: (${lastRoi.keypoint.x.toFixed(2)}, ${lastRoi.keypoint.y.toFixed(2)})</div>`;
  }
}

function drawMaskOnCanvas(
  result: ReturnType<InteractiveSegmenter['segment']>,
  ctx: CanvasRenderingContext2D,
  targetWidth: number,
  targetHeight: number,
) {
  const mask = result.categoryMask;
  if (!mask) {
    result.close();
    return;
  }

  const maskData = mask.getAsUint8Array();
  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = mask.width;
  maskCanvas.height = mask.height;
  const maskCtx = maskCanvas.getContext('2d');
  if (!maskCtx) {
    result.close();
    return;
  }

  const imageData = maskCtx.createImageData(mask.width, mask.height);
  for (let i = 0; i < maskData.length; i++) {
    const active = maskData[i] > 0;
    const offset = i * 4;
    imageData.data[offset] = 34;
    imageData.data[offset + 1] = 197;
    imageData.data[offset + 2] = 94;
    imageData.data[offset + 3] = active ? 150 : 0;
  }

  maskCtx.putImageData(imageData, 0, 0);
  ctx.drawImage(maskCanvas, 0, 0, targetWidth, targetHeight);

  if (lastRoi.keypoint) {
    ctx.beginPath();
    ctx.arc(lastRoi.keypoint.x * targetWidth, lastRoi.keypoint.y * targetHeight, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#ef4444';
    ctx.fill();
  }

  result.close();
  renderRoiInfo();
}

async function runImageSegmentation(image: HTMLImageElement, canvas: HTMLCanvasElement): Promise<void> {
  if (!segmenter || disposed) return;

  await ensureImageReady(image);
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  drawBaseImage(image, canvas, ctx);

  const start = performance.now();
  const result = segmenter.segment(image, lastRoi);
  setInferenceTime(start);
  drawMaskOnCanvas(result, ctx, canvas.width, canvas.height);
  setStatus('Done');
}

async function enableCam() {
  if (video.srcObject) return;

  enableWebcamButton.innerText = 'Starting...';
  enableWebcamButton.disabled = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;

    const playAndPredict = async () => {
      await video.play();
      runningMode = 'VIDEO';
      predictWebcam();
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

async function predictWebcam() {
  if (!segmenter || !video.srcObject || runningMode !== 'VIDEO') return;

  webcamCanvas.width = video.videoWidth;
  webcamCanvas.height = video.videoHeight;

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    webcamCtx.clearRect(0, 0, webcamCanvas.width, webcamCanvas.height);
    webcamCtx.drawImage(video, 0, 0, webcamCanvas.width, webcamCanvas.height);

    const start = performance.now();
    const result = segmenter.segment(video, lastRoi);
    setInferenceTime(start);
    drawMaskOnCanvas(result, webcamCtx, webcamCanvas.width, webcamCanvas.height);
  }

  animationFrameId = window.requestAnimationFrame(() => {
    void predictWebcam();
  });
}

export async function setupInteractiveSegmentation(container: HTMLElement) {
  disposed = false;
  container.innerHTML = template();

  const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
  const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
  const imageUpload = document.getElementById('image-upload') as HTMLInputElement;
  const reUploadBtn = document.getElementById('re-upload-btn') as HTMLButtonElement;
  const cameraCaptureBtn = document.getElementById('camera-capture-btn') as HTMLButtonElement;
  const image = document.getElementById('test-image') as HTMLImageElement;
  const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;

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
    await initSegmenter(MODELS[modelSelect.value], delegateSelect.value as 'CPU' | 'GPU');
    if (runningMode === 'IMAGE') {
      await runImageSegmentation(image, imageCanvas);
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
      await enableCam();
    } else {
      tabWebcam.classList.remove('active');
      tabImage.classList.add('active');
      viewWebcam.classList.remove('active');
      viewImage.classList.add('active');
      stopCam();
      await runImageSegmentation(image, imageCanvas);
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
    else void enableCam();
  });

  disposeUpload = bindImageUpload(imageUpload, image, async () => {
    await runImageSegmentation(image, imageCanvas);
  });
  disposeCameraCapture = bindCameraCapture(cameraCaptureBtn, image, async () => {
    await runImageSegmentation(image, imageCanvas);
  });

  imageCanvas.addEventListener('click', (evt) => {
    const rect = imageCanvas.getBoundingClientRect();
    const x = clamp01((evt.clientX - rect.left) / rect.width);
    const y = clamp01((evt.clientY - rect.top) / rect.height);
    lastRoi = { keypoint: { x, y } };
    if (runningMode === 'IMAGE') {
      void runImageSegmentation(image, imageCanvas);
    }
  });

  webcamCanvas.addEventListener('click', (evt) => {
    const rect = webcamCanvas.getBoundingClientRect();
    const x = clamp01((evt.clientX - rect.left) / rect.width);
    const y = clamp01((evt.clientY - rect.top) / rect.height);
    lastRoi = { keypoint: { x, y } };
  });

  image.addEventListener('load', () => {
    if (runningMode === 'IMAGE') {
      void runImageSegmentation(image, imageCanvas);
    }
  });

  await refresh();

  const initialMode = (localStorage.getItem('mediapipe-running-mode') as 'IMAGE' | 'VIDEO') || 'IMAGE';
  await switchView(initialMode);
}

export function cleanupInteractiveSegmentation() {
  disposed = true;
  stopCam();
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  disposeUpload?.();
  disposeUpload = undefined;
  disposeCameraCapture?.();
  disposeCameraCapture = undefined;
  if (segmenter) {
    segmenter.close();
    segmenter = undefined;
  }
}
