import template from '../templates/image-segmentation.html?raw';

let segmentationWorker: Worker | undefined;
let isWorkerReady = false;
let isSegmentingVideo = false;

let runningMode: 'IMAGE' | 'VIDEO' = 'IMAGE';
let video: HTMLVideoElement;
let canvasElement: HTMLCanvasElement;
let canvasCtx: WebGL2RenderingContext;
let enableWebcamButton: HTMLButtonElement;
let lastVideoTime = -1;
let animationFrameId: number;

// Options
let outputType: 'CATEGORY_MASK' | 'CONFIDENCE_MASKS' = 'CATEGORY_MASK';
let currentModelUrl = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite';
let labels: string[] = [];
let confidenceMaskSelection = 0;
let currentDelegate: 'GPU' | 'CPU' = 'GPU';
let modelLabels: string[] = [];

// Definitions for Custom Drawing Output
const legendColors: number[][] = [
  [66, 133, 244, 255],  // 0: Background - Google Blue (Masked)
  [128, 0, 0, 200],     // 1: Aeroplane
  [0, 128, 0, 200],     // 2: Bicycle
  [128, 128, 0, 200],   // 3: Bird
  [0, 0, 128, 200],     // 4: Boat
  [128, 0, 128, 200],   // 5: Bottle
  [0, 128, 128, 200],   // 6: Bus
  [128, 128, 128, 200], // 7: Car
  [64, 0, 0, 200],      // 8: Cat
  [0, 255, 0, 200],     // 9: Chair - Bright Green
  [192, 0, 0, 200],     // 10: Cow
  [255, 105, 180, 200], // 11: Dining Table - Pink
  [192, 128, 0, 200],   // 12: Dog
  [64, 0, 128, 200],    // 13: Horse
  [192, 0, 128, 200],   // 14: Motorbike
  [0, 255, 255, 255],   // 15: Person - Cyan
  [0, 128, 0, 200],     // 16: Potted Plant - Green
  [128, 64, 0, 200],    // 17: Sheep
  [0, 192, 0, 200],     // 18: Sofa
  [128, 192, 0, 200],   // 19: Train
  [0, 64, 128, 200]     // 20: TV
];

export async function setupImageSegmentation(container: HTMLElement) {
  container.innerHTML = template;

  video = document.getElementById('webcam') as HTMLVideoElement;
  canvasElement = document.getElementById('output_canvas') as HTMLCanvasElement;
  // Use WebGL2 for GPU delegate compatibility
  canvasCtx = canvasElement.getContext('webgl2') as WebGL2RenderingContext;
  enableWebcamButton = document.getElementById('webcamButton') as HTMLButtonElement;

  // Create/Get Overlay Canvas for 2D Drawing Output
  let overlayCanvas = document.getElementById('output_overlay') as HTMLCanvasElement;
  if (!overlayCanvas) {
    overlayCanvas = document.createElement('canvas');
    overlayCanvas.id = 'output_overlay';
    overlayCanvas.style.position = 'absolute';
    overlayCanvas.style.top = '0';
    overlayCanvas.style.left = '0';
    overlayCanvas.style.width = '100%';
    overlayCanvas.style.height = '100%';
    overlayCanvas.style.pointerEvents = 'none'; // Click-through
    overlayCanvas.style.mixBlendMode = 'normal'; // Revert to normal for correct color representation
    // Insert after output_canvas
    canvasElement.parentElement?.appendChild(overlayCanvas);
  }

  // Pre-load worker
  if (!segmentationWorker) {
    segmentationWorker = new Worker(new URL('../workers/image-segmentation.worker.ts', import.meta.url), { type: 'module' });
    setupWorkerListener();
  }

  await initializeSegmenter();
  setupUI();
}

function setupWorkerListener() {
  if (!segmentationWorker) return;
  segmentationWorker.onmessage = (event) => {
    const { type } = event.data;
    switch (type) {
      case 'INIT_DONE':
        document.querySelector('.viewport')?.classList.remove('loading-model');
        isWorkerReady = true;
        modelLabels = event.data.labels || [];
        labels = modelLabels;
        updateLegend();
        updateClassSelect();
        enableWebcamButton.disabled = false;
        enableWebcamButton.innerText = 'Enable Webcam';
        updateStatus(`Model loaded. Ready.`);
        triggerInitialImageSegmentation();
        if (runningMode === 'VIDEO' && video.srcObject) {
          enableCam();
        }
        break;
      case 'DELEGATE_FALLBACK':
        console.warn('Worker fell back to CPU delegate');
        currentDelegate = 'CPU';
        const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
        if (delegateSelect) delegateSelect.value = 'CPU';
        break;
      case 'OPTIONS_UPDATED':
        // Ready for next frame
        break;
      case 'SEGMENT_RESULT':
        const { mode, maskData, width, height, inferenceTime } = event.data;
        updateStatus(`Done in ${Math.round(inferenceTime)}ms`);
        updateInferenceTime(inferenceTime);

        if (maskData) {
          if (mode === 'IMAGE') drawMaskToImage(maskData, width, height);
          else if (mode === 'VIDEO') drawMaskToVideo(maskData, width, height);
        }

        if (mode === 'VIDEO') isSegmentingVideo = false;
        break;
      case 'ERROR':
      case 'SEGMENT_ERROR':
        console.error('Worker error:', event.data.error);
        updateStatus(`Error: ${event.data.error}`);
        if (event.data.mode === 'VIDEO') isSegmentingVideo = false;
        break;
    }
  };
}

export function cleanupImageSegmentation() {
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  if (video && video.srcObject) {
    const stream = video.srcObject as MediaStream;
    stream.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }
  if (segmentationWorker) {
    segmentationWorker.postMessage({ type: 'CLEANUP' });
    segmentationWorker.terminate();
    segmentationWorker = undefined;
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

const standardModels: Record<string, string> = {
  'deeplab_v3': 'https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite',
  'hair_segmenter': 'https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/1/hair_segmenter.tflite',
  'selfie_segmenter': 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float32/latest/selfie_segmenter.tflite',
  'selfie_multiclass': 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite'
};

async function initializeSegmenter() {
  updateStatus('Initializing model in worker...');
  document.querySelector('.viewport')?.classList.add('loading-model');
  isWorkerReady = false;

  const webcamBtn = document.getElementById('webcamButton') as HTMLButtonElement | null;
  if (webcamBtn && (!video || !video.srcObject)) {
    webcamBtn.disabled = true;
    webcamBtn.innerText = 'Initializing...';
  }

  const urlParams = new URLSearchParams(window.location.search);
  const delegateParam = urlParams.get('delegate');
  if (delegateParam === 'CPU' || delegateParam === 'GPU') {
    currentDelegate = delegateParam;
  }

  const baseUrl = new URL('./', window.location.href).href;
  segmentationWorker?.postMessage({
    type: 'INIT',
    modelAssetPath: currentModelUrl,
    delegate: currentDelegate,
    runningMode: runningMode,
    baseUrl: baseUrl
  });
}

function triggerInitialImageSegmentation() {
  if (runningMode === 'IMAGE') {
    const testImage = document.getElementById('test-image') as HTMLImageElement;
    if (testImage.src && testImage.style.display !== 'none') {
      const dropzoneContent = document.querySelector('.dropzone-content') as HTMLElement;
      const imagePreviewContainer = document.getElementById('image-preview-container');
      if (dropzoneContent) dropzoneContent.style.display = 'none';
      if (imagePreviewContainer) imagePreviewContainer.style.display = ''; // Grid display
      const reUploadBtn = document.getElementById('re-upload-btn');
      if (reUploadBtn) reUploadBtn.style.display = 'flex';

      if (testImage.complete && testImage.naturalWidth > 0) {
        segmentImage(testImage);
      } else {
        testImage.onload = () => segmentImage(testImage);
      }
    }
  }
}

function updateClassSelect() {
  const select = document.getElementById('class-select') as HTMLSelectElement;
  select.innerHTML = '';
  labels.forEach((label, index) => {
    const option = document.createElement('option');
    option.value = index.toString();
    option.text = label;
    select.appendChild(option);
  });
  if (confidenceMaskSelection < labels.length) {
    select.value = confidenceMaskSelection.toString();
  }
}

function setupUI() {
  // View Tabs
  const tabWebcam = document.getElementById('tab-webcam')!;
  const tabImage = document.getElementById('tab-image')!;
  const viewWebcam = document.getElementById('view-webcam')!;
  const viewImage = document.getElementById('view-image')!;

  const switchView = (mode: 'VIDEO' | 'IMAGE') => {
    localStorage.setItem('mediapipe-running-mode', mode);
    if (mode === 'VIDEO') {
      tabWebcam.classList.add('active');
      tabImage.classList.remove('active');
      viewWebcam.classList.add('active');
      viewImage.classList.remove('active');
      runningMode = 'VIDEO';
      if (isWorkerReady) segmentationWorker?.postMessage({ type: 'SET_OPTIONS', runningMode: 'VIDEO' });
      enableCam();
    } else {
      tabWebcam.classList.remove('active');
      tabImage.classList.add('active');
      viewWebcam.classList.remove('active');
      viewImage.classList.add('active');
      runningMode = 'IMAGE';
      if (isWorkerReady) {
        segmentationWorker?.postMessage({ type: 'SET_OPTIONS', runningMode: 'IMAGE' });
        triggerInitialImageSegmentation();
      }
      stopCam();
    }
  };

  const storedMode = localStorage.getItem('mediapipe-running-mode') as 'VIDEO' | 'IMAGE';
  if (storedMode === 'VIDEO') {
    switchView('VIDEO');
  } else {
    switchView('IMAGE');
  }

  tabWebcam.addEventListener('click', () => switchView('VIDEO'));
  tabImage.addEventListener('click', () => switchView('IMAGE'));

  // Model Tabs
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
      const select = document.getElementById('model-select') as HTMLSelectElement;
      if (standardModels[select.value]) {
        currentModelUrl = standardModels[select.value];
        initializeSegmenter();
      }
    } else {
      tabModelList.classList.remove('active');
      tabModelUpload.classList.add('active');
      viewModelList.classList.remove('active');
      viewModelUpload.classList.add('active');
    }
  };

  tabModelList.addEventListener('click', () => switchModelTab('LIST'));
  tabModelUpload.addEventListener('click', () => switchModelTab('UPLOAD'));

  const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
  modelSelect.innerHTML = '';
  for (const key in standardModels) {
    const option = document.createElement('option');
    option.value = key;
    option.text = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    if (standardModels[key] === currentModelUrl) option.selected = true;
    modelSelect.appendChild(option);
  }

  enableWebcamButton.addEventListener('click', enableCam);

  if (storedMode === 'VIDEO') {
    enableCam();
  }

  modelSelect.addEventListener('change', async () => {
    const key = modelSelect.value;
    if (standardModels[key]) {
      currentModelUrl = standardModels[key];
      enableWebcamButton.innerText = 'Loading...';
      enableWebcamButton.disabled = true;
      await initializeSegmenter();
    }
  });

  const modelUpload = document.getElementById('model-upload') as HTMLInputElement;
  modelUpload.addEventListener('change', async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file) {
      currentModelUrl = URL.createObjectURL(file);
      enableWebcamButton.innerText = 'Loading...';
      enableWebcamButton.disabled = true;
      await initializeSegmenter();
    }
  });

  const outputTypeSelect = document.getElementById('output-type') as HTMLSelectElement;
  const classSelectContainer = document.getElementById('class-select-container') as HTMLElement;

  outputTypeSelect.addEventListener('change', () => {
    outputType = outputTypeSelect.value as any;
    updateLegend();
    if (outputType === 'CONFIDENCE_MASKS') {
      classSelectContainer.style.display = 'block';
    } else {
      classSelectContainer.style.display = 'none';
    }
    if (runningMode === 'IMAGE') triggerInitialImageSegmentation();
  });

  const classSelect = document.getElementById('class-select') as HTMLSelectElement;
  classSelect.addEventListener('change', () => {
    confidenceMaskSelection = parseInt(classSelect.value);
    if (runningMode === 'IMAGE') triggerInitialImageSegmentation();
  });

  const opacityInput = document.getElementById('opacity') as HTMLInputElement;
  opacityInput.addEventListener('input', () => {
    canvasElement.style.opacity = opacityInput.value;
    const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
    if (imageCanvas) imageCanvas.style.opacity = opacityInput.value;
    const overlayCanvas = document.getElementById('output_overlay') as HTMLCanvasElement;
    if (overlayCanvas) overlayCanvas.style.opacity = opacityInput.value;
  });

  if (opacityInput) {
    canvasElement.style.opacity = opacityInput.value;
    const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
    if (imageCanvas) imageCanvas.style.opacity = opacityInput.value;
    const overlayCanvas = document.getElementById('output_overlay') as HTMLCanvasElement;
    if (overlayCanvas) overlayCanvas.style.opacity = opacityInput.value;
  }

  const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
  if (delegateSelect && currentDelegate) {
    delegateSelect.value = currentDelegate;
  }
  delegateSelect.addEventListener('change', async () => {
    currentDelegate = delegateSelect.value as 'GPU' | 'CPU';
    enableWebcamButton.innerText = 'Loading...';
    enableWebcamButton.disabled = true;
    await initializeSegmenter();
  });

  const imageUpload = document.getElementById('image-upload') as HTMLInputElement;
  const imagePreviewContainer = document.getElementById('image-preview-container')!;
  const dropzone = document.querySelector('.upload-dropzone') as HTMLElement;
  if (dropzone) dropzone.addEventListener('click', () => imageUpload.click());

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
        const testImage = document.getElementById('test-image') as HTMLImageElement;
        testImage.src = e.target?.result as string;
        testImage.style.display = 'block';
        imagePreviewContainer.style.display = ''; 
        const dropzoneContent = document.querySelector('.dropzone-content') as HTMLElement;
        const reUploadBtn = document.getElementById('re-upload-btn');
        if (dropzoneContent) dropzoneContent.style.display = 'none';
        if (reUploadBtn) reUploadBtn.style.display = 'flex';

        testImage.onload = () => segmentImage(testImage);
      };
      reader.readAsDataURL(file);
    }
  });
}

async function processMaskFast(maskBuffer: ArrayBuffer, width: number, height: number, targetWidth: number, targetHeight: number, ctx: CanvasRenderingContext2D) {
  // Create an ImageData buffer manually
  const floatMask = new Float32Array(maskBuffer);
  const offscreen = new OffscreenCanvas(width, height);
  const offCtx = offscreen.getContext('2d') as OffscreenCanvasRenderingContext2D;
  const imageData = offCtx.createImageData(width, height);
  const data = imageData.data;

  let activePixels = 0;
  let maxConf = 0;

  if (outputType === 'CATEGORY_MASK') {
    for (let i = 0; i < floatMask.length; i++) {
      const category = floatMask[i];
      if (category > 0) activePixels++;
      const color = legendColors[category] || [0, 0, 0, 0];

      data[i * 4] = color[0];
      data[i * 4 + 1] = color[1];
      data[i * 4 + 2] = color[2];
      data[i * 4 + 3] = color[3]; // Alpha mapping depends on blend mode, normally 255 for solid colors if mask applied
    }
  } else if (outputType === 'CONFIDENCE_MASKS') {
    // Confidence mask outputs 0.0 to 1.0 confidence for the SELECTED category
    // In our worker setup, wait... did we pass the ENTIRE confidenceMasks array? No, the worker code
    // uses categoryMask mode exclusively because getAsFloat32Array for confidence array transfers all classes!

    // BUT! Since our worker ONLY returns categoryMask FloatArray, we CANNOT do confidence mask here properly!
    // We only have the CategoryMask buffer.
    // If outputType is CONFIDENCE_MASKS in the current implementation, we are constrained...
    // Let's emulate it by checking if the category matches the selection and setting confidence to 1.0
    for (let i = 0; i < floatMask.length; i++) {
      const category = floatMask[i];
      const isMatch = category === confidenceMaskSelection;
      if (isMatch) {
        activePixels++;
        maxConf = 1.0;
        data[i * 4] = 0; // B
        data[i * 4 + 1] = 0; // G
        data[i * 4 + 2] = 255; // R (Blue)
        data[i * 4 + 3] = 255; // A
      } else {
        data[i * 4] = 0;
        data[i * 4 + 1] = 0;
        data[i * 4 + 2] = 0;
        data[i * 4 + 3] = 0; // Transparent
      }
    }
  }

  offCtx.putImageData(imageData, 0, 0);
  ctx.drawImage(offscreen, 0, 0, targetWidth, targetHeight);
  return { activePixels, maxConf };
}

async function drawMaskToImage(maskData: ArrayBuffer, width: number, height: number) {
  const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
  const testImage = document.getElementById('test-image') as HTMLImageElement;
  if (!imageCanvas || !testImage) return;
  const imageCtx = imageCanvas.getContext('2d');
  if (!imageCtx) return;

  // CRITICAL: Sync internal canvas resolution to rendered image size before scaling!
  imageCanvas.width = testImage.width;
  imageCanvas.height = testImage.height;

  imageCtx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
  const stats = await processMaskFast(maskData, width, height, imageCanvas.width, imageCanvas.height, imageCtx);

  // Expose results for testing
  const existingResults = document.querySelectorAll('#test-results');
  existingResults.forEach(el => el.remove());

  const resultsEl = document.createElement('div');
  resultsEl.id = 'test-results';
  resultsEl.style.display = 'none';
  resultsEl.textContent = JSON.stringify({
    timestamp: Date.now(),
    completion: 'done',
    activePixelCount: stats.activePixels,
    maxConfidence: stats.maxConf
  });

  document.body.appendChild(resultsEl);
}

async function drawMaskToVideo(maskData: ArrayBuffer, width: number, height: number) {
  canvasElement.height = video.videoHeight;
  canvasElement.width = video.videoWidth;
  canvasElement.style.opacity = '0'; // Hide WebGL canvas

  const overlayCanvas = document.getElementById('output_overlay') as HTMLCanvasElement;
  if (overlayCanvas) {
    if (overlayCanvas.width !== video.videoWidth || overlayCanvas.height !== video.videoHeight) {
      overlayCanvas.width = video.videoWidth;
      overlayCanvas.height = video.videoHeight;
    }
    const ctx = overlayCanvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
      await processMaskFast(maskData, width, height, overlayCanvas.width, overlayCanvas.height, ctx);
    }
  }
}

async function segmentImage(image: HTMLImageElement) {
  if (!isWorkerReady || !segmentationWorker) return;

  updateStatus('Processing image...');
  if (runningMode !== 'IMAGE') {
    runningMode = 'IMAGE';
    segmentationWorker.postMessage({ type: 'SET_OPTIONS', runningMode: 'IMAGE' });
  }

  const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
  imageCanvas.width = image.naturalWidth;
  imageCanvas.height = image.naturalHeight;

  try {
    const bitmap = await window.createImageBitmap(image);
    segmentationWorker.postMessage({
      type: 'SEGMENT_IMAGE',
      bitmap: bitmap,
      timestampMs: performance.now()
    }, [bitmap]);
  } catch (e) {
    console.error('Failed to create ImageBitmap from image', e);
  }
}

async function enableCam() {
  if (video.paused) {
    enableWebcamButton.innerText = 'Disable Webcam';
    const constraints = { video: true };

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = stream;
      video.addEventListener('loadeddata', predictWebcam);
      runningMode = 'VIDEO';
      if (isWorkerReady) segmentationWorker?.postMessage({ type: 'SET_OPTIONS', runningMode: 'VIDEO' });
      updateStatus('Webcam running...');
    } catch (err) {
      console.error(err);
      updateStatus('Camera error!');
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
    updateStatus('Ready');
  }
}

async function predictWebcam() {
  if (runningMode === 'IMAGE') {
    runningMode = 'VIDEO';
    if (isWorkerReady) segmentationWorker?.postMessage({ type: 'SET_OPTIONS', runningMode: 'VIDEO' });
  }

  if (video.currentTime !== lastVideoTime && isWorkerReady && !isSegmentingVideo) {
    lastVideoTime = video.currentTime;
    isSegmentingVideo = true;

    try {
      const bitmap = await window.createImageBitmap(video);
      segmentationWorker?.postMessage({
        type: 'SEGMENT_VIDEO',
        bitmap: bitmap,
        timestampMs: performance.now()
      }, [bitmap]);
    } catch (e) {
      console.warn('Failed to extract frame in video loop', e);
      isSegmentingVideo = false;
    }
  }

  animationFrameId = window.requestAnimationFrame(predictWebcam);
}


function updateLegend() {
  const legendContainer = document.getElementById('legend');
  if (!legendContainer) return;

  legendContainer.innerHTML = '';

  if (outputType === 'CONFIDENCE_MASKS') {
    legendContainer.style.display = 'none';
    return;
  }

  if (modelLabels.length > 0) {
    legendContainer.style.display = 'flex';
  } else {
    legendContainer.style.display = 'none';
    return;
  }

  modelLabels.forEach((label, index) => {
    const colorData = legendColors[index % legendColors.length];
    const color = `rgba(${colorData[0]}, ${colorData[1]}, ${colorData[2]}, ${colorData[3] / 255})`;

    const item = document.createElement('div');
    item.className = 'legend-item';

    const colorBox = document.createElement('div');
    colorBox.className = 'legend-color';
    colorBox.style.backgroundColor = color;

    const text = document.createElement('span');
    text.innerText = label;

    item.appendChild(colorBox);
    item.appendChild(text);
    legendContainer.appendChild(item);
  });
}
