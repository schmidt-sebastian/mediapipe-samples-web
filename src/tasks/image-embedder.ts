import template from '../templates/image-embedder.html?raw';

let worker: Worker | undefined;
let isWorkerReady = false;

// Options
const models: Record<string, string> = {
  'mobilenet_v3_small': 'https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite',
  'mobilenet_v3_large': 'https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_large/float32/1/mobilenet_v3_large.tflite'
};

let currentModel = 'mobilenet_v3_small';
let currentDelegate: 'CPU' | 'GPU' = 'GPU';

export async function setupImageEmbedder(container: HTMLElement) {
  container.innerHTML = template;

  // UI References
  const embedBtn = document.getElementById('embed-btn') as HTMLButtonElement;
  const imageUpload1 = document.getElementById('image-upload-1') as HTMLInputElement;
  const imageUpload2 = document.getElementById('image-upload-2') as HTMLInputElement;
  const dropzone1 = document.getElementById('dropzone-1')!;
  const dropzone2 = document.getElementById('dropzone-2')!;
  const image1 = document.getElementById('image-1') as HTMLImageElement;
  const image2 = document.getElementById('image-2') as HTMLImageElement;
  const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
  const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;

  // Event Listeners
  embedBtn.addEventListener('click', () => {
    if (image1.src && image2.src) {
      computeSimilarity(image1, image2);
    }
  });

  const setupDropzone = (dropzone: HTMLElement, input: HTMLInputElement, img: HTMLImageElement) => {
    dropzone.addEventListener('click', () => input.click());

    input.addEventListener('change', (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          img.src = e.target?.result as string;
          dropzone.querySelector('.dropzone-content')!.setAttribute('style', 'display: none;');
          dropzone.querySelector('.preview-container')!.setAttribute('style', 'display: flex;');

          // Enable button if both images are present
          if (image1.src && image2.src && image1.naturalWidth > 0 && image2.naturalWidth > 0) {
            embedBtn.disabled = false;
          }
        };
        reader.readAsDataURL(file);
      }
    });

    // Handle drag and drop if desired, but click is essential
    // For now click is implemented
  };

  setupDropzone(dropzone1, imageUpload1, image1);
  setupDropzone(dropzone2, imageUpload2, image2);

  // Verify images loaded before enabling button completely
  image1.onload = () => {
    if (image2.src && image2.naturalWidth > 0) embedBtn.disabled = false;
  };
  image2.onload = () => {
    if (image1.src && image1.naturalWidth > 0) embedBtn.disabled = false;
  };

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

  const embedBtn = document.getElementById('embed-btn') as HTMLButtonElement;
  if (embedBtn) {
    embedBtn.disabled = true;
  }

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
  const embedBtn = document.getElementById('embed-btn') as HTMLButtonElement;

  if ((window as any).imgEmbedLoadTimeout) clearTimeout((window as any).imgEmbedLoadTimeout);

  if (loadingOverlay) {
    loadingOverlay.style.display = 'none';
  }

  if (statusMessage) statusMessage.innerText = `Error: ${errorMsg}`;
  if (embedBtn) {
    // Only enable if images are present
    const image1 = document.getElementById('image-1') as HTMLImageElement;
    const image2 = document.getElementById('image-2') as HTMLImageElement;
    if (image1.src && image2.src) {
      embedBtn.disabled = false;
      embedBtn.innerText = "Retry";
    }
  }
}

function handleWorkerMessage(event: MessageEvent) {
  const { type } = event.data;
  const loadingOverlay = document.getElementById('loading-overlay');
  const statusMessage = document.getElementById('status-message');
  const embedBtn = document.getElementById('embed-btn') as HTMLButtonElement;

  switch (type) {
    case 'INIT_DONE':
      isWorkerReady = true;
      if ((window as any).imgEmbedLoadTimeout) clearTimeout((window as any).imgEmbedLoadTimeout);

      if (loadingOverlay) loadingOverlay.style.display = 'none';
      if (statusMessage) statusMessage.innerText = 'Ready';

      const image1 = document.getElementById('image-1') as HTMLImageElement;
      const image2 = document.getElementById('image-2') as HTMLImageElement;
      if (embedBtn && image1.src && image2.src) {
        embedBtn.disabled = false;
        embedBtn.innerText = 'Compute Similarity';
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
      if (embedBtn) embedBtn.disabled = false;
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

  const embedBtn = document.getElementById('embed-btn') as HTMLButtonElement;
  const statusMessage = document.getElementById('status-message');

  embedBtn.disabled = true;
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
    container.style.display = 'block';
    valueEl.innerText = similarity.toFixed(4);
  }
}

export function cleanupImageEmbedder() {
  if (worker) {
    worker.postMessage({ type: 'CLEANUP' });
  }
}
