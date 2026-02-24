import { DrawingUtils, NormalizedLandmark } from '@mediapipe/tasks-vision';

export function getVisionWasmBasePath(): string {
  const baseUrl = (import.meta as { env: { BASE_URL: string } }).env.BASE_URL;
  return new URL(`${baseUrl}wasm`, window.location.origin).href;
}

export function resolveAssetPath(path: string): string {
  if (/^(https?:|blob:|data:)/.test(path)) {
    return path;
  }
  const baseUrl = (import.meta as { env: { BASE_URL: string } }).env.BASE_URL;
  return new URL(path, new URL(baseUrl, window.location.origin)).href;
}

export async function ensureImageReady(image: HTMLImageElement): Promise<void> {
  if (image.complete && image.naturalWidth > 0) {
    return;
  }

  await new Promise<void>((resolve, reject) => {
    const onLoad = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error('Failed to load image'));
    };
    const cleanup = () => {
      image.removeEventListener('load', onLoad);
      image.removeEventListener('error', onError);
    };

    image.addEventListener('load', onLoad, { once: true });
    image.addEventListener('error', onError, { once: true });
  });
}

export function drawBaseImage(
  image: HTMLImageElement,
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
): void {
  canvas.width = image.naturalWidth || image.width;
  canvas.height = image.naturalHeight || image.height;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
}

export function drawLandmarkSets(
  ctx: CanvasRenderingContext2D,
  landmarkSets: NormalizedLandmark[][],
  connections: Array<{ start: number; end: number }>,
): void {
  const drawingUtils = new DrawingUtils(ctx);
  for (const landmarks of landmarkSets) {
    drawingUtils.drawConnectors(landmarks, connections, { color: '#22c55e', lineWidth: 2 });
    drawingUtils.drawLandmarks(landmarks, { color: '#0ea5e9', radius: 2 });
  }
}

export function bindImageUpload(
  fileInput: HTMLInputElement,
  image: HTMLImageElement,
  onImageReady: () => Promise<void> | void,
): () => void {
  const listener = () => {
    const file = fileInput.files?.[0];
    if (!file) return;

    const objectUrl = URL.createObjectURL(file);
    image.src = objectUrl;

    const onLoad = () => {
      URL.revokeObjectURL(objectUrl);
      void onImageReady();
      image.removeEventListener('load', onLoad);
    };
    image.addEventListener('load', onLoad);
  };

  fileInput.addEventListener('change', listener);
  return () => fileInput.removeEventListener('change', listener);
}

let cameraModalRoot: HTMLDivElement | undefined;
let cameraModalVideo: HTMLVideoElement | undefined;
let cameraCaptureBtn: HTMLButtonElement | undefined;
let cameraCancelBtn: HTMLButtonElement | undefined;
let activeCameraStream: MediaStream | undefined;

function ensureCameraModal(): void {
  if (cameraModalRoot) return;

  cameraModalRoot = document.createElement('div');
  cameraModalRoot.style.position = 'fixed';
  cameraModalRoot.style.inset = '0';
  cameraModalRoot.style.background = 'rgba(15, 23, 42, 0.75)';
  cameraModalRoot.style.display = 'none';
  cameraModalRoot.style.alignItems = 'center';
  cameraModalRoot.style.justifyContent = 'center';
  cameraModalRoot.style.zIndex = '9999';
  cameraModalRoot.innerHTML = `
    <div style="background:#fff; border-radius:12px; width:min(92vw,720px); padding:16px;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
        <strong>Built-in Camera</strong>
        <small style="color:#64748b;">Capture a frame to use as input</small>
      </div>
      <video id="camera-modal-video" autoplay playsinline muted style="width:100%; border-radius:8px; background:#0f172a;"></video>
      <div style="display:flex; gap:10px; justify-content:flex-end; margin-top:12px;">
        <button id="camera-modal-cancel" style="padding:8px 12px;">Cancel</button>
        <button id="camera-modal-capture" style="padding:8px 12px;">Capture</button>
      </div>
    </div>
  `;
  document.body.appendChild(cameraModalRoot);

  cameraModalVideo = cameraModalRoot.querySelector('#camera-modal-video') as HTMLVideoElement;
  cameraCaptureBtn = cameraModalRoot.querySelector('#camera-modal-capture') as HTMLButtonElement;
  cameraCancelBtn = cameraModalRoot.querySelector('#camera-modal-cancel') as HTMLButtonElement;
}

function stopCameraStream(): void {
  if (!activeCameraStream) return;
  for (const track of activeCameraStream.getTracks()) {
    track.stop();
  }
  activeCameraStream = undefined;
}

async function captureImageFromCamera(): Promise<string | null> {
  ensureCameraModal();
  if (!cameraModalRoot || !cameraModalVideo || !cameraCaptureBtn || !cameraCancelBtn) {
    return null;
  }

  stopCameraStream();

  try {
    activeCameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user' },
      audio: false,
    });
  } catch {
    return null;
  }

  cameraModalVideo.srcObject = activeCameraStream;
  cameraModalRoot.style.display = 'flex';
  await cameraModalVideo.play();

  return await new Promise<string | null>((resolve) => {
    const close = (value: string | null) => {
      cameraCaptureBtn?.removeEventListener('click', onCapture);
      cameraCancelBtn?.removeEventListener('click', onCancel);
      cameraModalRoot!.style.display = 'none';
      cameraModalVideo!.srcObject = null;
      stopCameraStream();
      resolve(value);
    };

    const onCancel = () => close(null);
    const onCapture = () => {
      if (!cameraModalVideo || cameraModalVideo.videoWidth === 0 || cameraModalVideo.videoHeight === 0) {
        close(null);
        return;
      }
      const canvas = document.createElement('canvas');
      canvas.width = cameraModalVideo.videoWidth;
      canvas.height = cameraModalVideo.videoHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        close(null);
        return;
      }
      ctx.drawImage(cameraModalVideo, 0, 0, canvas.width, canvas.height);
      close(canvas.toDataURL('image/png'));
    };

    cameraCaptureBtn.addEventListener('click', onCapture);
    cameraCancelBtn.addEventListener('click', onCancel);
  });
}

export function bindCameraCapture(
  triggerButton: HTMLButtonElement,
  image: HTMLImageElement,
  onImageReady: () => Promise<void> | void,
): () => void {
  const listener = async () => {
    const dataUrl = await captureImageFromCamera();
    if (!dataUrl) return;
    image.src = dataUrl;
    await onImageReady();
  };

  triggerButton.addEventListener('click', listener);
  return () => triggerButton.removeEventListener('click', listener);
}

export function setStatus(text: string): void {
  const el = document.getElementById('status-message');
  if (el) el.textContent = text;
}

export function setInferenceTime(startTime: number): void {
  const el = document.getElementById('inference-time');
  if (el) {
    const duration = performance.now() - startTime;
    el.textContent = `Inference Time: ${duration.toFixed(1)} ms`;
  }
}

export function renderSimpleCategories(
  container: HTMLElement,
  categories: Array<{ categoryName: string; score: number }>,
  maxItems = 5,
): void {
  const top = [...categories].sort((a, b) => b.score - a.score).slice(0, maxItems);
  container.innerHTML = top
    .map((c) => {
      const pct = Math.round(c.score * 100);
      return `<div style="display:grid;grid-template-columns:1fr auto;gap:8px;margin-bottom:8px;"><span>${c.categoryName || 'Unknown'}</span><strong>${pct}%</strong></div>`;
    })
    .join('') || '<div>No results</div>';
}

export const sharedTaskStyles = `
  <style>
    .simple-task-wrap { padding: 20px; display: flex; flex-direction: column; gap: 16px; height: 100%; overflow: auto; }
    .simple-task-controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
    .simple-task-pane { background: var(--surface); border: 1px solid var(--border-color); border-radius: 10px; padding: 12px; }
    .simple-task-output { display: grid; grid-template-columns: minmax(0, 1fr) minmax(260px, 340px); gap: 12px; min-height: 0; }
    .simple-task-stage { position: relative; width: 100%; min-height: 320px; background: #f1f5f9; border-radius: 10px; overflow: hidden; }
    .simple-task-stage img, .simple-task-stage canvas { width: 100%; height: auto; display: block; }
    .simple-task-stage canvas { position: absolute; inset: 0; }
    .simple-task-results { font-size: 14px; line-height: 1.4; }
    @media (max-width: 900px) { .simple-task-output { grid-template-columns: 1fr; } }
  </style>
`;
