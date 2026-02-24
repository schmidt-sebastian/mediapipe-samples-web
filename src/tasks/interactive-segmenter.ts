// @ts-ignore
import template from '../templates/interactive-segmenter.html?raw';

let worker: Worker | undefined;
let isWorkerReady = false;
let outputCanvas: HTMLCanvasElement;
let outputCtx: CanvasRenderingContext2D;

const models: Record<string, string> = {
  'magic_touch': 'https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite'
};

export async function setupInteractiveSegmenter(container: HTMLElement) {
  container.innerHTML = template;

  outputCanvas = document.getElementById('output_canvas') as HTMLCanvasElement;
  outputCtx = outputCanvas.getContext('2d', { willReadFrequently: true })!;

  initWorker();
  setupUI();
  await initializeSegmenter();
}

// @ts-ignore
import InteractiveSegmenterWorker from '../workers/interactive-segmenter.worker.ts?worker';

function initWorker() {
  if (!worker) {
    worker = new InteractiveSegmenterWorker();
  }
  if (worker) {
    worker.onmessage = handleWorkerMessage;
  }
}

function handleWorkerMessage(event: MessageEvent) {
  const { type } = event.data;

  switch (type) {
    case 'INIT_DONE':
      isWorkerReady = true;
      updateStatus('Ready');
      break;

    case 'SEGMENT_RESULT':
      const { maskData, width, height, inferenceTime } = event.data;
      updateInferenceTime(inferenceTime);
      drawResult(maskData, width, height);
      updateStatus(`Done in ${Math.round(inferenceTime)}ms`);
      break;

    case 'ERROR':
      console.error('Worker error:', event.data.error);
      updateStatus(`Error: ${event.data.error}`);
      break;
  }
}

async function initializeSegmenter() {
  isWorkerReady = false;
  updateStatus('Loading Model...');

  // @ts-ignore
  const baseUrl = import.meta.env.BASE_URL;
  const modelPath = models['magic_touch']; // Fixed for now

  worker?.postMessage({
    type: 'INIT',
    modelAssetPath: modelPath,
    delegate: 'GPU',
    baseUrl
  });
}

function setupUI() {
  const imageUpload = document.getElementById('image-upload') as HTMLInputElement;
  const imagePreviewContainer = document.getElementById('image-preview-container')!;
  const testImage = document.getElementById('test-image') as HTMLImageElement;
  const dropzone = document.querySelector('.upload-dropzone') as HTMLElement;
  const dropzoneContent = document.querySelector('.dropzone-content') as HTMLElement;

  if (dropzone) dropzone.addEventListener('click', () => imageUpload.click());

  imageUpload.addEventListener('change', (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        testImage.src = e.target?.result as string;
        imagePreviewContainer.style.display = '';
        if (dropzoneContent) dropzoneContent.style.display = 'none';

        // Clear previous result
        outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
      };
      reader.readAsDataURL(file);
    }
  });

  // Handle clicks on image/canvas
  const handleInteraction = async (e: MouseEvent) => {
    if (!isWorkerReady || !testImage.src) return;

    // Get click coordinates relative to the image
    const rect = testImage.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;

    if (x >= 0 && x <= 1 && y >= 0 && y <= 1) {
      updateStatus('Segmenting...');
      const bitmap = await createImageBitmap(testImage);
      worker?.postMessage({
        type: 'SEGMENT',
        bitmap,
        pt: { x, y }
      }, [bitmap]);
    }
  };

  testImage.addEventListener('click', handleInteraction);
  outputCanvas.addEventListener('click', handleInteraction);

  // Resize canvas to match image when loaded
  testImage.onload = () => {
    outputCanvas.width = testImage.naturalWidth;
    outputCanvas.height = testImage.naturalHeight;
    outputCanvas.style.width = '100%';
    outputCanvas.style.height = 'auto';
  };
  // Trigger onload if already complete
  if (testImage.complete) {
    testImage.onload(new Event('load'));
  }
}

function drawResult(maskData: Uint8Array | null, width: number, height: number) {
  if (!maskData) return;

  // Create ImageData
  const imageData = outputCtx.createImageData(width, height);
  const data = imageData.data;

  // Overlay color (e.g., semi-transparent blue)
  // Mask is uint8, usually 0 or 1? Or category index?
  // Interactive segmenter usually returns category mask where 1 is the object.

  for (let i = 0; i < maskData.length; i++) {
    const category = maskData[i]; // 0 or 1 usually
    if (category > 0) {
      const offset = i * 4;
      data[offset] = 0;     // R
      data[offset + 1] = 0; // G
      data[offset + 2] = 255; // B
      data[offset + 3] = 128; // Alpha (semi-transparent)
    }
  }

  outputCtx.putImageData(imageData, 0, 0);
}

function updateStatus(msg: string) {
  const el = document.getElementById('status-message');
  if (el) el.innerText = msg;
}

function updateInferenceTime(time: number) {
  const el = document.getElementById('inference-time');
  if (el) el.innerText = `Inference Time: ${time.toFixed(2)} ms`;
}

export function cleanupInteractiveSegmenter() {
  if (worker) {
    worker.terminate();
    worker = undefined;
  }
  isWorkerReady = false;
}
