import {
  ImageSegmenter,
  FilesetResolver,
  ImageSegmenterResult,
  DrawingUtils
} from '@mediapipe/tasks-vision';

let imageSegmenter: ImageSegmenter | undefined;
let runningMode: 'IMAGE' | 'VIDEO' = 'IMAGE';
let video: HTMLVideoElement;
let canvasElement: HTMLCanvasElement;
let canvasCtx: WebGL2RenderingContext;
let enableWebcamButton: HTMLButtonElement;
let lastVideoTime = -1;
let animationFrameId: number;
let drawingUtils: DrawingUtils | undefined;

// Options
let outputType: 'CATEGORY_MASK' | 'CONFIDENCE_MASKS' = 'CATEGORY_MASK';
let currentModelUrl = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite';
let labels: string[] = [];
let confidenceMaskSelection = 0;
let currentDelegate: 'GPU' | 'CPU' = 'GPU';
let modelLabels: string[] = [];

// Definitions for DrawingUtils
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

import template from '../templates/image-segmentation.html?raw';

export async function setupImageSegmentation(container: HTMLElement) {
  container.innerHTML = template;

  video = document.getElementById('webcam') as HTMLVideoElement;
  canvasElement = document.getElementById('output_canvas') as HTMLCanvasElement;
  // Use WebGL2 for GPU delegate compatibility
  canvasCtx = canvasElement.getContext('webgl2') as WebGL2RenderingContext;
  enableWebcamButton = document.getElementById('webcamButton') as HTMLButtonElement;

  // Create/Get Overlay Canvas for 2D Drawing (DrawingUtils)
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
    overlayCanvas.style.pointerEvents = 'none'; // Click-through
    overlayCanvas.style.mixBlendMode = 'normal'; // Revert to normal for correct color representation
    // Insert after output_canvas
    // Insert after output_canvas
    canvasElement.parentElement?.appendChild(overlayCanvas);
  }
  const overlayCtx = overlayCanvas.getContext('2d')!;

  // Initialize DrawingUtils with BOTH contexts to handle GPU<->CPU interop if needed
  drawingUtils = new DrawingUtils(overlayCtx, canvasCtx);

  await initializeSegmenter();
  setupUI();
}

export function cleanupImageSegmentation() {
  if (animationFrameId) cancelAnimationFrame(animationFrameId);
  if (video && video.srcObject) {
    const stream = video.srcObject as MediaStream;
    stream.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }
  imageSegmenter?.close();
  imageSegmenter = undefined;
  drawingUtils = undefined;
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
  updateStatus('Initializing model...');
  let vision: any;
  try {
    vision = await FilesetResolver.forVisionTasks('/mediapipe-samples-web/wasm');

    imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: currentModelUrl,
        delegate: currentDelegate,
      },
      canvas: canvasElement, // Provide main DOM canvas for WebGL
      outputCategoryMask: true,
      outputConfidenceMasks: true,
      runningMode: runningMode,
    });

    // Get labels if available
    modelLabels = imageSegmenter.getLabels();
    updateLegend();

    labels = imageSegmenter.getLabels();
    updateClassSelect();

    enableWebcamButton.disabled = false;
    enableWebcamButton.innerText = 'Enable Webcam';
    updateStatus('Model loaded. Ready.');

    if (runningMode === 'IMAGE') {
      const testImage = document.getElementById('test-image') as HTMLImageElement;
      if (testImage.src && testImage.style.display !== 'none') {
        const dropzoneContent = document.querySelector('.dropzone-content') as HTMLElement;
        const imagePreviewContainer = document.getElementById('image-preview-container');
        if (dropzoneContent) dropzoneContent.style.display = 'none';
        if (imagePreviewContainer) imagePreviewContainer.style.display = ''; // Let CSS grid handle it (was block)
        const reUploadBtn = document.getElementById('re-upload-btn');
        if (reUploadBtn) reUploadBtn.style.display = 'flex';

        if (testImage.complete && testImage.naturalWidth > 0) {
          segmentImage(testImage);
        } else {
          testImage.onload = () => {
            segmentImage(testImage);
          };
        }
      }
    }
  } catch (e) {
    console.warn('GPU initialization failed, falling back to CPU', e);
    if (currentDelegate === 'GPU') {
      currentDelegate = 'CPU';
      const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
      if (delegateSelect) delegateSelect.value = 'CPU';

      try {
        imageSegmenter = await ImageSegmenter.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: currentModelUrl,
            delegate: 'CPU',
          },
          outputCategoryMask: true,
          outputConfidenceMasks: true,
          runningMode: runningMode,
        });

        // Get labels if available
        modelLabels = imageSegmenter.getLabels();
        updateLegend();

        labels = imageSegmenter.getLabels();
        updateClassSelect();
        enableWebcamButton.disabled = false;
        enableWebcamButton.innerText = 'Enable Webcam';
        updateStatus('Model loaded (CPU). Ready.');

        // Retry image if needed
        if (runningMode === 'IMAGE') {
          const testImage = document.getElementById('test-image') as HTMLImageElement;
          if (testImage.src && testImage.style.display !== 'none') {
            if (testImage.complete && testImage.naturalWidth > 0) segmentImage(testImage);
            else testImage.onload = () => segmentImage(testImage);
          }
        }
        return;
      } catch (cpuError) {
        console.error('CPU initialization also failed:', cpuError);
        updateStatus(`Error loading model: ${cpuError}`);
        return;
      }
    }
    console.error(e);
    updateStatus(`Error loading model: ${e}`);
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
  // Main Tabs (Webcam/Image)
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
      if (imageSegmenter) imageSegmenter.setOptions({ runningMode: 'VIDEO' });
      enableCam();
    } else {
      tabWebcam.classList.remove('active');
      tabImage.classList.add('active');
      viewWebcam.classList.remove('active');
      viewImage.classList.add('active');
      runningMode = 'IMAGE';
      if (imageSegmenter) imageSegmenter.setOptions({ runningMode: 'IMAGE' });
      stopCam();
    }
  };

  // Persist State
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

  // Initialize Model Select
  const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
  modelSelect.innerHTML = '';
  for (const key in standardModels) {
    const option = document.createElement('option');
    option.value = key;
    option.text = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()); // Capitalize
    if (standardModels[key] === currentModelUrl) option.selected = true;
    modelSelect.appendChild(option);
  }

  enableWebcamButton.addEventListener('click', enableCam);

  // Auto-Start Webcam if persisted mode is VIDEO
  if (storedMode === 'VIDEO') {
    enableCam();
  }

  // Model Select Listener
  modelSelect.addEventListener('change', async () => {
    const key = modelSelect.value;
    if (standardModels[key]) {
      currentModelUrl = standardModels[key];
      enableWebcamButton.innerText = 'Loading...';
      enableWebcamButton.disabled = true;
      await initializeSegmenter();
    }
  });

  // Model Upload
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
    updateLegend(); // Toggle legend visibility
    if (outputType === 'CONFIDENCE_MASKS') {
      classSelectContainer.style.display = 'block';
    } else {
      classSelectContainer.style.display = 'none';
    }
    if (runningMode === 'IMAGE') {
      const testImage = document.getElementById('test-image') as HTMLImageElement;
      if (testImage.src) segmentImage(testImage);
    }
  });

  const classSelect = document.getElementById('class-select') as HTMLSelectElement;
  classSelect.addEventListener('change', () => {
    confidenceMaskSelection = parseInt(classSelect.value);
    if (runningMode === 'IMAGE') {
      const testImage = document.getElementById('test-image') as HTMLImageElement;
      if (testImage.src) segmentImage(testImage);
    }
  });

  const opacityInput = document.getElementById('opacity') as HTMLInputElement;
  opacityInput.addEventListener('input', () => {
    canvasElement.style.opacity = opacityInput.value;
    const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
    if (imageCanvas) imageCanvas.style.opacity = opacityInput.value;
    const overlayCanvas = document.getElementById('output_overlay') as HTMLCanvasElement;
    if (overlayCanvas) overlayCanvas.style.opacity = opacityInput.value;
  });

  // Apply initial opacity
  if (opacityInput) {
    canvasElement.style.opacity = opacityInput.value;
    const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
    if (imageCanvas) imageCanvas.style.opacity = opacityInput.value;
    const overlayCanvas = document.getElementById('output_overlay') as HTMLCanvasElement;
    if (overlayCanvas) overlayCanvas.style.opacity = opacityInput.value;
  }

  const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
  delegateSelect.addEventListener('change', async () => {
    currentDelegate = delegateSelect.value as 'GPU' | 'CPU';
    enableWebcamButton.innerText = 'Loading...';
    enableWebcamButton.disabled = true;
    await initializeSegmenter();
  });

  const imageUpload = document.getElementById('image-upload') as HTMLInputElement;
  const imagePreviewContainer = document.getElementById('image-preview-container')!;
  const testImage = document.getElementById('test-image') as HTMLImageElement;

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
        console.log('File reader loaded');
        const testImage = document.getElementById('test-image') as HTMLImageElement;
        testImage.src = e.target?.result as string;
        testImage.style.display = 'block';
        imagePreviewContainer.style.display = ''; // Let CSS grid handle it (was block)
        const dropzoneContent = document.querySelector('.dropzone-content') as HTMLElement;
        const reUploadBtn = document.getElementById('re-upload-btn');
        if (dropzoneContent) dropzoneContent.style.display = 'none';
        if (reUploadBtn) reUploadBtn.style.display = 'flex';

        testImage.onload = () => {
          segmentImage(testImage);
        };
      };
      reader.readAsDataURL(file);
    }
  });
}

async function segmentImage(image: HTMLImageElement) {
  if (!imageSegmenter) {
    console.log('ImageSegmenter not ready');
    return;
  }
  console.log('Starting segmentImage...');
  updateStatus('Processing image...');

  if (runningMode !== 'IMAGE') {
    runningMode = 'IMAGE';
    await imageSegmenter.setOptions({ runningMode: 'IMAGE' });
  }

  // Ensure image canvas is set up
  const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;

  // Match the display size of the image for proper overlay
  // The canvas internal resolution should match the natural size for sharp rendering
  // BUT DrawingUtils might need to be told where to draw if we scale it.
  // Actually, easiest is to correct the canvas CSS to match the image element EXACTLY.

  imageCanvas.width = image.naturalWidth;
  imageCanvas.height = image.naturalHeight;

  // We don't set style width/height here, we let CSS hande it (absolute + object-fit: contain)
  // this requires the canvas to have the same aspect ratio as the image?
  // No, if we use object-fit: contain on BOTH, they should align IF the container has the same aspect ratio?
  // Actually, standard overlay technique:
  // Container = relative. Image = static/block. Canvas = absolute, top 0, left 0, width 100%, height 100%.
  // AND Image must fill the container? Or Container shrinks to Image?

  const imageCtx = imageCanvas.getContext('2d');
  if (!imageCtx) return;

  imageCtx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);

  // Initialize DrawingUtils for Image View (2D destination, WebGL source)
  const imageDrawingUtils = new DrawingUtils(imageCtx, canvasCtx);

  const start = performance.now();
  const result = imageSegmenter.segment(image);
  const time = performance.now() - start;
  updateStatus(`Done in ${Math.round(time)}ms`);
  updateInferenceTime(time);

  // Draw directly to image-canvas
  if (outputType === 'CATEGORY_MASK' && result.categoryMask) {
    imageDrawingUtils.drawCategoryMask(result.categoryMask, legendColors as any);
  } else if (outputType === 'CONFIDENCE_MASKS' && result.confidenceMasks) {
    const mask = result.confidenceMasks[confidenceMaskSelection];
    if (mask) {
      imageDrawingUtils.drawConfidenceMask(mask, [0, 0, 0, 0], [0, 0, 255, 255]);
    }
  }

  // Cleanup local drawing utils if needed (though GC handles it, close() might free WebGL resources if any)
  imageDrawingUtils.close();

  // Expose results for testing
  let activePixelCount = 0;
  let maxConfidence = 0;

  if (outputType === 'CATEGORY_MASK' && result.categoryMask) {
    const mask = result.categoryMask.getAsUint8Array();
    for (let i = 0; i < mask.length; i++) {
      if (mask[i] > 0) activePixelCount++;
    }
  } else if (outputType === 'CONFIDENCE_MASKS' && result.confidenceMasks) {
    const mask = result.confidenceMasks[confidenceMaskSelection].getAsFloat32Array();
    for (let i = 0; i < mask.length; i++) {
      if (mask[i] > 0.1) activePixelCount++; // Count pixels with > 10% confidence
      if (mask[i] > maxConfidence) maxConfidence = mask[i];
    }
  }

  const resultsEl = document.createElement('div');
  resultsEl.id = 'test-results';
  resultsEl.style.display = 'none';
  resultsEl.textContent = JSON.stringify({
    timestamp: Date.now(),
    completion: 'done',
    activePixelCount,
    maxConfidence
  });

  document.body.appendChild(resultsEl);
}

async function enableCam() {
  if (!imageSegmenter) return;

  if (video.paused) {
    enableWebcamButton.innerText = 'Disable Webcam';
    const constraints = { video: true };

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = stream;
      video.addEventListener('loadeddata', predictWebcam);
      runningMode = 'VIDEO';
      await imageSegmenter.setOptions({ runningMode: 'VIDEO' });
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
    await imageSegmenter?.setOptions({ runningMode: 'VIDEO' });
  }

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const startTimeMs = performance.now();

    if (imageSegmenter) {
      imageSegmenter.segmentForVideo(video, startTimeMs, (result) => {
        const time = performance.now() - startTimeMs; // Approximation for callback time
        updateInferenceTime(time);
        displayVideoSegmentation(result);
      });
    }
  }

  animationFrameId = window.requestAnimationFrame(predictWebcam);
}

function displayVideoSegmentation(result: ImageSegmenterResult) {
  // Ensure video is visible by clearing canvases or managing opacity
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;

  // WebGL Canvas - clear it completely or hide it
  // We only use it for GPU context processing, we don't need to see it if we draw on overlay
  // canvasElement.style.opacity = '0'; // Or just clear it transparent
  canvasElement.style.opacity = '0'; // Hide WebGL canvas to show video
  canvasCtx.viewport(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.clearColor(0, 0, 0, 0); // Transparent
  canvasCtx.clear(canvasCtx.COLOR_BUFFER_BIT);

  // Update Overlay Canvas Size
  const overlayCanvas = document.getElementById('output_overlay') as HTMLCanvasElement;
  if (overlayCanvas) {
    if (overlayCanvas.width !== video.videoWidth || overlayCanvas.height !== video.videoHeight) {
      overlayCanvas.width = video.videoWidth;
      overlayCanvas.height = video.videoHeight;
    }
    const ctx = overlayCanvas.getContext('2d');
    // Clear overlay for new frame
    if (ctx) ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  }

  // Draw to Overlay
  if (outputType === 'CATEGORY_MASK' && result.categoryMask) {
    drawingUtils?.drawCategoryMask(result.categoryMask, legendColors as any);
  } else if (outputType === 'CONFIDENCE_MASKS' && result.confidenceMasks) {
    const mask = result.confidenceMasks[confidenceMaskSelection];
    if (mask) {
      // Draw confidence mask with Transparent background [0,0,0,0] to see video
      // Foreground color (high confidence) = Blue [0, 0, 255, 255]
      // Params: mask, defaultTexture (Background), overlayTexture (Foreground/Mask)
      drawingUtils?.drawConfidenceMask(mask, [0, 0, 0, 0], [0, 0, 255, 255]);
    }
  }
}

function updateLegend() {
  const legendContainer = document.getElementById('legend');
  if (!legendContainer) return;

  // Clear existing items
  legendContainer.innerHTML = '';

  // Hide legend if in CONFIDENCE_MASKS mode
  if (outputType === 'CONFIDENCE_MASKS') {
    legendContainer.style.display = 'none';
    return;
  }

  // Show legend if we have labels
  if (modelLabels.length > 0) {
    legendContainer.style.display = 'flex';
  } else {
    legendContainer.style.display = 'none';
    return;
  }

  modelLabels.forEach((label, index) => {
    // index 0 is typically background, but check if we want to show it.
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
