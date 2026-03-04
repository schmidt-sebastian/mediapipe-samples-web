// @ts-ignore
import template from '../templates/interactive-segmenter.html?raw';
import { ViewToggle } from '../components/view-toggle';
import { ModelSelector } from '../components/model-selector';

let worker: Worker | undefined;
let isWorkerReady = false;
let outputCanvas: HTMLCanvasElement;
let outputCtx: CanvasRenderingContext2D;
// Webcam elements
let video: HTMLVideoElement;
let webcamCapture: HTMLCanvasElement;
let webcamOverlay: HTMLCanvasElement;
let webcamCtx: CanvasRenderingContext2D;
let overlayCtx: CanvasRenderingContext2D;
let webcamButton: HTMLButtonElement;
let freezeButton: HTMLButtonElement;
let stream: MediaStream | null = null;
let isFrozen = false;

const models: Record<string, string> = {
  'magic_touch': 'https://storage.googleapis.com/mediapipe-models/interactive_segmenter/magic_touch/float32/1/magic_touch.tflite'
};

let currentModelUrl: string = models['magic_touch'];
let modelSelector: ModelSelector;

export async function setupInteractiveSegmenter(container: HTMLElement) {
  container.innerHTML = template;

  outputCanvas = document.getElementById('output_canvas') as HTMLCanvasElement;
  outputCtx = outputCanvas.getContext('2d', { willReadFrequently: true })!;

  // Initialize webcam elements
  video = document.getElementById('webcam') as HTMLVideoElement;
  webcamCapture = document.getElementById('webcam-capture') as HTMLCanvasElement;
  webcamOverlay = document.getElementById('webcam-overlay') as HTMLCanvasElement;
  webcamButton = document.getElementById('webcamButton') as HTMLButtonElement;
  freezeButton = document.getElementById('freezeButton') as HTMLButtonElement;

  webcamCtx = webcamCapture.getContext('2d', { willReadFrequently: true })!;
  overlayCtx = webcamOverlay.getContext('2d', { willReadFrequently: true })!;

  // Set initial visibility
  webcamCapture.style.display = 'none';
  webcamOverlay.style.display = 'none';
  webcamOverlay.style.position = 'absolute';
  webcamOverlay.style.top = '0';
  webcamOverlay.style.left = '0';
  webcamOverlay.style.pointerEvents = 'none'; // Let clicks pass through to capture canvas if needed, but we bind click to capture canvas

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
    case 'LOAD_PROGRESS':
      const { progress } = event.data;
      modelSelector?.showProgress(progress * 100, 100);
      if (progress >= 1) {
        setTimeout(() => {
          modelSelector?.hideProgress();
        }, 500);
      }
      break;

    case 'INIT_DONE':
      modelSelector?.hideProgress();
      isWorkerReady = true;
      updateStatus('Ready');
      webcamButton.disabled = false;
      break;

    case 'SEGMENT_RESULT':
      const { maskData, width, height, inferenceTime } = event.data;
      updateInferenceTime(inferenceTime);

      // Determine where to draw based on active view
      const activeTab = document.querySelector('.view-tab.active')?.getAttribute('data-value');
      if (activeTab === 'webcam') {
        drawResult(maskData, width, height, overlayCtx);
      } else {
        drawResult(maskData, width, height, outputCtx);
      }

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
  webcamButton.disabled = true;

  // @ts-ignore
  const baseUrl = import.meta.env.BASE_URL;

  worker?.postMessage({
    type: 'INIT',
    modelAssetPath: currentModelUrl,
    delegate: (document.getElementById('delegate-select') as HTMLSelectElement)?.value || 'GPU',
    baseUrl
  });
}

function setupUI() {
  const imageUpload = document.getElementById('image-upload') as HTMLInputElement;
  const imagePreviewContainer = document.getElementById('image-preview-container')!;
  const testImage = document.getElementById('test-image') as HTMLImageElement;
  const dropzone = document.querySelector('.upload-dropzone') as HTMLElement;

  // Fix: Stop propagation on image click to prevent opening file picker
  // The dropzone has a click listener that opens file picker
  if (dropzone) {
    dropzone.addEventListener('click', (e) => {
      if (e.target !== imageUpload) {
        imageUpload.click();
      }
    });
  }

  // Prevent clicks on the preview container from bubbling up to dropzone
  if (imagePreviewContainer) {
    imagePreviewContainer.addEventListener('click', (e) => {
      e.stopPropagation();
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
        if (dropzone) dropzone.classList.add('active'); // Use class instead of hiding content logic directly if we want css control
        // if (dropzoneContent) dropzoneContent.style.display = 'none'; // CSS handles this now

        // Clear previous result
        outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
      };
      reader.readAsDataURL(file);
    }
  });

  // Handle clicks on image/canvas
  const handleInteraction = async (e: MouseEvent, source: 'image' | 'webcam') => {
    console.log('Interaction detected:', source);
    if (!isWorkerReady) {
      console.log('Worker not ready');
      return;
    }

    let targetElement: HTMLImageElement | HTMLCanvasElement;
    let originalBitmapSource: HTMLImageElement | HTMLCanvasElement;

    if (source === 'image') {
      if (!testImage.src) return;
      targetElement = testImage;
      originalBitmapSource = testImage;
    } else {
      // Webcam source
      console.log('Webcam source. detection. isFrozen:', isFrozen);
      if (!isFrozen) return; // Only segment when frozen
      targetElement = webcamCapture;
      originalBitmapSource = webcamCapture;
    }

    const rect = targetElement.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;

    console.log('Click coordinates:', x, y);

    if (x >= 0 && x <= 1 && y >= 0 && y <= 1) {
      updateStatus('Segmenting...');
      try {
        const bitmap = await createImageBitmap(originalBitmapSource);
        worker?.postMessage({
          type: 'SEGMENT',
          bitmap,
          pt: { x, y }
        }, [bitmap]);
      } catch (err) {
        console.error('Error creating bitmap or sending message:', err);
      }
    }
  };

  testImage.addEventListener('click', (e) => handleInteraction(e, 'image'));
  outputCanvas.addEventListener('click', (e) => handleInteraction(e, 'image'));

  // Webcam interactions
  webcamCapture.addEventListener('click', (e) => handleInteraction(e, 'webcam'));
  webcamOverlay.addEventListener('click', (e) => {
    // Forward click to capture canvas logic since overlay is on top
    handleInteraction(e, 'webcam');
  });

  // Resize canvas to match image when loaded
  testImage.onload = () => {
    outputCanvas.width = testImage.naturalWidth;
    outputCanvas.height = testImage.naturalHeight;
    // outputCanvas.style.width = '100%';
    // outputCanvas.style.height = 'auto';
  };

  if (testImage.complete) {
    testImage.onload(new Event('load'));
  }

  const storedMode = localStorage.getItem('mediapipe-running-mode') as 'webcam' | 'image';
  const initialMode = storedMode || 'webcam';

  new ViewToggle(
    'view-mode-toggle',
    [
      { label: 'Webcam', value: 'webcam' },
      { label: 'Image', value: 'image' }
    ],
    initialMode,
    (value) => {
      localStorage.setItem('mediapipe-running-mode', value);
      const viewImage = document.getElementById('view-image')!;
      const viewWebcam = document.getElementById('view-webcam')!;

      if (value === 'webcam') {
        viewWebcam.style.display = '';
        viewImage.style.display = 'none';
        viewWebcam.classList.add('active');
        viewImage.classList.remove('active');
        const isWebcamActive = localStorage.getItem('mediapipe-webcam-active') === 'true';
        if (isWebcamActive && !stream) toggleWebcam();
      } else {
        viewImage.style.display = '';
        viewWebcam.style.display = 'none';
        viewImage.classList.add('active');
        viewWebcam.classList.remove('active');
        if (stream) toggleWebcam();
      }
    }
  );

  // Initialize view properly on load
  const viewToggleButtons = document.querySelectorAll('#view-mode-toggle button');
  viewToggleButtons.forEach(btn => {
    if ((btn as HTMLButtonElement).dataset.value === 'webcam') {
      btn.classList.add('active'); // Ensure initial state matches
    }
  });

  modelSelector = new ModelSelector(
    'model-selector-container',
    [
      { label: 'Magic Touch', value: 'magic_touch', isDefault: true }
    ],
    async (selection) => {
      if (selection.type === 'standard') {
        currentModelUrl = models[selection.value] || models['magic_touch'];
      } else if (selection.type === 'custom') {
        currentModelUrl = URL.createObjectURL(selection.file);
      }
      updateStatus('Loading Model...');
      webcamButton.disabled = true;
      freezeButton.disabled = true;
      await initializeSegmenter();
    }
  );

  webcamButton.addEventListener('click', toggleWebcam);
  freezeButton.addEventListener('click', toggleFreeze);
}

async function toggleWebcam() {
  if (!stream) {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
    document.getElementById('webcam-placeholder')?.classList.add('hidden');
      video.style.display = 'block';
      webcamCapture.style.display = 'none';
      webcamOverlay.style.display = 'none';
      webcamButton.innerText = 'Disable Webcam';
      localStorage.setItem('mediapipe-webcam-active', 'true');
      freezeButton.disabled = false;
      isFrozen = false;
      freezeButton.innerText = 'Freeze & Segment';
      // Reset logging
      console.log('Webcam started');
    } catch (err) {
      console.error('Error accessing webcam:', err);
      updateStatus('Error accessing webcam');
    }
  } else {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
    video.srcObject = null;
    document.getElementById('webcam-placeholder')?.classList.remove('hidden');
    webcamButton.innerText = 'Enable Webcam';
    localStorage.setItem('mediapipe-webcam-active', 'false');
    freezeButton.disabled = true;
    isFrozen = false;
    // Hide overlay
    webcamOverlay.style.display = 'none';
    webcamCapture.style.display = 'none';
    video.style.display = 'block';
  }
}

function toggleFreeze() {
  if (!stream) return;

  if (!isFrozen) {
    // Freeze
    webcamCapture.width = video.videoWidth;
    webcamCapture.height = video.videoHeight;
    webcamOverlay.width = video.videoWidth;
    webcamOverlay.height = video.videoHeight;

    webcamCtx.drawImage(video, 0, 0);

    video.style.display = 'none';
    webcamCapture.style.display = 'block';
    webcamOverlay.style.display = 'block';
    // EXPLICITLY ENABLE POINTER EVENTS
    webcamOverlay.style.pointerEvents = 'auto';
    webcamOverlay.classList.add('clickable');

    // Position overlay matches capture
    webcamOverlay.style.width = '100%'; // CSS handles layout

    isFrozen = true;
    freezeButton.innerText = 'Unfreeze';
    updateStatus('Click on object to segment');
    console.log('Webcam frozen. Clicks enabled on overlay.');
  } else {
    // Unfreeze
    isFrozen = false;
    freezeButton.innerText = 'Freeze & Segment';
    video.style.display = 'block';
    webcamCapture.style.display = 'none';
    webcamOverlay.style.display = 'none';
    webcamOverlay.style.pointerEvents = 'none';

    // Clear overlay
    overlayCtx.clearRect(0, 0, webcamOverlay.width, webcamOverlay.height);
    updateStatus('Ready to freeze');
  }
}

function drawResult(maskData: Uint8Array | null, width: number, height: number, ctx: CanvasRenderingContext2D) {
  if (!maskData) return;

  // Create ImageData
  const imageData = ctx.createImageData(width, height);
  const data = imageData.data;

  for (let i = 0; i < maskData.length; i++) {
    const category = maskData[i];
    if (category > 0) {
      const offset = i * 4;
      data[offset] = 0;     // R
      data[offset + 1] = 0; // G
      data[offset + 2] = 255; // B
      data[offset + 3] = 128; // Alpha
    }
  }

  ctx.putImageData(imageData, 0, 0);
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
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  if (worker) {
    worker.terminate();
    worker = undefined;
  }
  isWorkerReady = false;
}
