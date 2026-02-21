import template from '../templates/text-embedding.html?raw';

let worker: Worker | undefined;
let isWorkerReady = false;

// Options
const models: Record<string, string> = {
  'universal_sentence_encoder': 'https://storage.googleapis.com/mediapipe-models/text_embedder/universal_sentence_encoder/float32/1/universal_sentence_encoder.tflite'
};

let currentModel = 'universal_sentence_encoder';
let currentDelegate: 'CPU' | 'GPU' = 'CPU';

export async function setupTextEmbedding(container: HTMLElement) {
  container.innerHTML = template;

  // UI References
  const embedBtn = document.getElementById('embed-btn') as HTMLButtonElement;
  const textInput1 = document.getElementById('text-input-1') as HTMLTextAreaElement;
  const textInput2 = document.getElementById('text-input-2') as HTMLTextAreaElement;
  const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
  const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;

  // Event Listeners
  embedBtn.addEventListener('click', () => {
    if (textInput1.value.trim() && textInput2.value.trim()) {
      computeSimilarity(textInput1.value, textInput2.value);
    }
  });

  // Sample Buttons
  const sampleBtns = container.querySelectorAll('.sample-btn');
  sampleBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      const el = e.currentTarget as HTMLElement;
      const t1 = el.dataset.text1;
      const t2 = el.dataset.text2;
      if (t1 && t2) {
        textInput1.value = t1;
        textInput2.value = t2;
        computeSimilarity(t1, t2);
      }
    });
  });

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
    worker = new Worker(new URL('../workers/text-embedding.worker.ts', import.meta.url), { type: 'module' });
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

    // Clear any existing timeout
    if ((window as any).embedLoadTimeout) clearTimeout((window as any).embedLoadTimeout);

    (window as any).embedLoadTimeout = setTimeout(() => {
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

  if ((window as any).embedLoadTimeout) clearTimeout((window as any).embedLoadTimeout);

  if (loadingOverlay) {
    loadingOverlay.style.display = 'none';
  }

  if (statusMessage) statusMessage.innerText = `Error: ${errorMsg}`;
  if (embedBtn) {
    embedBtn.disabled = false;
    embedBtn.innerText = "Retry";
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
      if ((window as any).embedLoadTimeout) clearTimeout((window as any).embedLoadTimeout);

      if (loadingOverlay) loadingOverlay.style.display = 'none';
      if (statusMessage) statusMessage.innerText = 'Ready';
      if (embedBtn) {
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
    case 'ERROR':
      console.error('Worker error:', event.data.error);
      handleWorkerError(event.data.error);
      break;
  }
}

function computeSimilarity(text1: string, text2: string) {
  if (!worker || !isWorkerReady) return;

  const embedBtn = document.getElementById('embed-btn') as HTMLButtonElement;
  const statusMessage = document.getElementById('status-message');

  embedBtn.disabled = true;
  if (statusMessage) statusMessage.innerText = 'Computing...';

  worker.postMessage({
    type: 'EMBED',
    text1: text1,
    text2: text2,
    timestampMs: performance.now()
  });
}

function displayResults(similarity: number) {
  const container = document.getElementById('embedding-results');
  const valueEl = document.getElementById('similarity-value');

  if (container && valueEl) {
    container.style.display = 'block';
    valueEl.innerText = similarity.toFixed(4);

    // Maybe color code based on high/low?
    // valueEl.style.color = similarity > 0.8 ? 'green' : similarity > 0.5 ? 'orange' : 'red';
  }
}

export function cleanupTextEmbedding() {
  if (worker) {
    worker.postMessage({ type: 'CLEANUP' });
    worker.terminate();
    worker = undefined;
  }
}
