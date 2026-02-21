import template from '../templates/image-embedder.html?raw';

let worker: Worker | undefined;
let isWorkerReady = false;

// Options
const models: Record<string, string> = {
  'mobilenet_v3_small': 'https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite',
  'mobilenet_v3_large': 'https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_large/float32/1/mobilenet_v3_large.tflite'
};

let currentModel = 'mobilenet_v3_small';
let currentDelegate: 'CPU' | 'GPU' = 'CPU';

export async function setupImageEmbedder(container: HTMLElement) {
  container.innerHTML = template;

  // UI References

  const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
  const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;

  const image1 = document.getElementById('image-1') as HTMLImageElement;
  const image2 = document.getElementById('image-2') as HTMLImageElement;
  const display1 = document.getElementById('display-area-1')!;
  const display2 = document.getElementById('display-area-2')!;

  // Helper to set image and update UI
  const setImage = (img: HTMLImageElement, display: HTMLElement, src: string) => {
    img.src = src;
    img.style.display = 'block';
    display.classList.add('has-image');
    const placeholder = display.querySelector('.placeholder-text') as HTMLElement;
    if (placeholder) placeholder.style.display = 'none';

    // Enable button if both images are present
    checkEnableButton();
  };

  const checkEnableButton = () => {
    // Check if both images have valid sources
    if (image1.src && image2.src && image1.src.length > 0 && image2.src.length > 0) {
      computeSimilarity(image1, image2);
    }
  };

  const handleUpload = (input: HTMLInputElement, img: HTMLImageElement, display: HTMLElement) => {
    input.value = ''; // Reset input to allow re-selecting same file
    input.click();
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          setImage(img, display, e.target?.result as string);
        };
        reader.readAsDataURL(file);
      }
    };
  };

  // Attach listeners to samples
  document.querySelectorAll('.sample-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = (btn as HTMLElement).dataset.target;
      const src = (btn as HTMLElement).dataset.src;
      if (targetId === '1' && src) setImage(image1, display1, src);
      if (targetId === '2' && src) setImage(image2, display2, src);
    });
  });

  // Attach listeners to upload buttons
  document.querySelectorAll('.upload-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = (btn as HTMLElement).dataset.target;
      if (targetId === '1') handleUpload(document.getElementById('image-upload-1') as HTMLInputElement, image1, display1);
      if (targetId === '2') handleUpload(document.getElementById('image-upload-2') as HTMLInputElement, image2, display2);
    });
  });

  // Make display areas clickable for upload
  display1.addEventListener('click', () => {
    handleUpload(document.getElementById('image-upload-1') as HTMLInputElement, image1, display1);
  });

  display2.addEventListener('click', () => {
    handleUpload(document.getElementById('image-upload-2') as HTMLInputElement, image2, display2);
  });

  // Image onload handlers
  image1.onload = checkEnableButton;
  image2.onload = checkEnableButton;



  if (modelSelect) {
    modelSelect.addEventListener('change', (e) => {
      currentModel = (e.target as HTMLSelectElement).value;
      isWorkerReady = false;
      initEmbedder();
    });
  }

  if (delegateSelect) {
    delegateSelect.addEventListener('change', (e) => {
      currentDelegate = (e.target as HTMLSelectElement).value as 'CPU' | 'GPU';
      isWorkerReady = false;
      initEmbedder();
    });
  }

  // Init Worker
  if (!worker) {
    worker = new Worker(new URL('../workers/image-embedder.worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = handleWorkerMessage;
  }

  await initEmbedder();
}

async function initEmbedder() {
  const loadingOverlay = document.getElementById('loading-overlay');
  if (loadingOverlay) {
    loadingOverlay.style.display = 'flex';
    const loadingText = loadingOverlay.querySelector('.loading-text');
    if (loadingText) loadingText.textContent = 'Loading Model...';
  }

  const statusMessage = document.getElementById('status-message');
  if (statusMessage) statusMessage.innerText = 'Loading Model...';



  // @ts-ignore
  const baseUrl = import.meta.env.BASE_URL;

  try {
    worker?.postMessage({
      type: 'INIT',
      modelAssetPath: models[currentModel],
      delegate: currentDelegate,
      baseUrl: baseUrl
    });

    if ((window as any).imgEmbedLoadTimeout) clearTimeout((window as any).imgEmbedLoadTimeout);

    (window as any).imgEmbedLoadTimeout = setTimeout(() => {
      if (!isWorkerReady) {
        console.error("Model load timed out");
        handleWorkerError("Model loading timed out. Please try refreshing.");
      }
    }, 20000);

  } catch (e) {
    console.error(e);
    handleWorkerError(String(e));
  }
}

function handleWorkerError(errorMsg: string) {
  const loadingOverlay = document.getElementById('loading-overlay');
  const statusMessage = document.getElementById('status-message');

  if ((window as any).imgEmbedLoadTimeout) clearTimeout((window as any).imgEmbedLoadTimeout);

  if (loadingOverlay) {
    loadingOverlay.style.display = 'none';
  }

  if (statusMessage) statusMessage.innerText = `Error: ${errorMsg}`;

  // Only enable if images are present
  const image1 = document.getElementById('image-1') as HTMLImageElement;
  const image2 = document.getElementById('image-2') as HTMLImageElement;
  // Check using src property
  if (image1.src && image2.src && image1.src.length > 0 && image2.src.length > 0) {
    // Auto retry?
    computeSimilarity(image1, image2);
  }
}

function handleWorkerMessage(event: MessageEvent) {
  const { type } = event.data;
  const loadingOverlay = document.getElementById('loading-overlay');
  const statusMessage = document.getElementById('status-message');

  switch (type) {
    case 'INIT_DONE':
      isWorkerReady = true;
      if ((window as any).imgEmbedLoadTimeout) clearTimeout((window as any).imgEmbedLoadTimeout);

      if (loadingOverlay) loadingOverlay.style.display = 'none';
      if (statusMessage) statusMessage.innerText = 'Ready';

      const image1 = document.getElementById('image-1') as HTMLImageElement;
      const image2 = document.getElementById('image-2') as HTMLImageElement;

      if (image1 && image2 && image1.src && image2.src && image1.src.length > 0 && image2.src.length > 0) {
        // Auto-compute on init if images present
        computeSimilarity(image1, image2);
      }
      break;
    case 'EMBED_RESULT':
      const { similarity, timestampMs } = event.data;
      const duration = performance.now() - timestampMs;
      const inferenceTimeEl = document.getElementById('inference-time');
      if (inferenceTimeEl) {
        inferenceTimeEl.innerText = `Inference Time: ${duration.toFixed(1)} ms`;
      }
      displayResults(similarity);
      if (statusMessage) statusMessage.innerText = 'Done';
      break;
    case 'DELEGATE_FALLBACK':
      console.warn('Worker fell back to CPU delegate.');
      currentDelegate = 'CPU';
      const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
      if (delegateSelect) delegateSelect.value = 'CPU';
      break;
    case 'ERROR':
      console.error('Worker error:', event.data.error);
      handleWorkerError(event.data.error);
      break;
  }
}

async function computeSimilarity(img1: HTMLImageElement, img2: HTMLImageElement) {
  if (!worker || !isWorkerReady) return;

  const statusMessage = document.getElementById('status-message');

  if (statusMessage) statusMessage.innerText = 'Computing...';

  const bitmap1 = await createImageBitmap(img1);
  const bitmap2 = await createImageBitmap(img2);

  worker.postMessage({
    type: 'EMBED',
    image1: bitmap1,
    image2: bitmap2,
    timestampMs: performance.now()
  }, [bitmap1, bitmap2]);
}

function displayResults(similarity: number) {
  const container = document.getElementById('embedding-results');
  const valueEl = document.getElementById('similarity-value');

  if (container && valueEl) {
    valueEl.innerText = similarity.toFixed(4);
  }
}

export function cleanupImageEmbedder() {
  if (worker) {
    worker.postMessage({ type: 'CLEANUP' });
  }
}
