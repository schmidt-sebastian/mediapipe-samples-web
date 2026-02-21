import template from '../templates/language-detector.html?raw';

let worker: Worker | undefined;
let isWorkerReady = false;

// Options
const models: Record<string, string> = {
  'language_detector': 'https://storage.googleapis.com/mediapipe-models/language_detector/language_detector/float32/1/language_detector.tflite'
};

let currentModel = 'language_detector';
let maxResults = 3;
let scoreThreshold = 0.0;
let currentDelegate: 'CPU' | 'GPU' = 'CPU';

export async function setupLanguageDetector(container: HTMLElement) {
  container.innerHTML = template;

  // UI References
  const detectBtn = document.getElementById('detect-btn') as HTMLButtonElement;
  const textInput = document.getElementById('text-input') as HTMLTextAreaElement;
  const modelSelect = document.getElementById('model-select') as HTMLSelectElement;
  const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
  const maxResultsInput = document.getElementById('max-results') as HTMLInputElement;
  const maxResultsValue = document.getElementById('max-results-value');
  const scoreThresholdInput = document.getElementById('score-threshold') as HTMLInputElement;
  const scoreThresholdValue = document.getElementById('score-threshold-value');

  // Event Listeners
  detectBtn.addEventListener('click', () => {
    if (textInput.value.trim()) {
      detectLanguage(textInput.value);
    }
  });

  // Sample Buttons
  const sampleBtns = container.querySelectorAll('.sample-btn');
  sampleBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      const text = (e.currentTarget as HTMLElement).dataset.text;
      if (text) {
        textInput.value = text;
        detectLanguage(text);
      }
    });
  });

  if (modelSelect) {
    modelSelect.addEventListener('change', (e) => {
      currentModel = (e.target as HTMLSelectElement).value;
      isWorkerReady = false;
      initDetector();
    });
  }

  if (delegateSelect) {
    delegateSelect.addEventListener('change', (e) => {
      currentDelegate = (e.target as HTMLSelectElement).value as 'CPU' | 'GPU';
      isWorkerReady = false;
      initDetector();
    });
  }

  if (maxResultsInput) {
    maxResultsInput.addEventListener('input', (e) => {
      maxResults = parseInt((e.target as HTMLInputElement).value);
      if (maxResultsValue) maxResultsValue.innerText = maxResults.toString();
      isWorkerReady = false;
      initDetector();
    });
  }

  if (scoreThresholdInput) {
    scoreThresholdInput.addEventListener('input', (e) => {
      scoreThreshold = parseFloat((e.target as HTMLInputElement).value);
      if (scoreThresholdValue) scoreThresholdValue.innerText = scoreThreshold.toString();
      isWorkerReady = false;
      initDetector();
    });
  }

  // Init Worker
  if (!worker) {
    worker = new Worker(new URL('../workers/language-detector.worker.ts', import.meta.url), { type: 'classic' });
    worker.onmessage = handleWorkerMessage;
  }

  // Initial Load
  await initDetector();
}

async function initDetector() {
  const loadingOverlay = document.getElementById('loading-overlay');
  if (loadingOverlay) {
    loadingOverlay.style.display = 'flex';
    const loadingText = loadingOverlay.querySelector('.loading-text');
    if (loadingText) loadingText.textContent = 'Loading Model...';
  }

  const statusMessage = document.getElementById('status-message');
  if (statusMessage) statusMessage.innerText = 'Loading Model...';

  const detectBtn = document.getElementById('detect-btn') as HTMLButtonElement;
  if (detectBtn) {
    detectBtn.disabled = true;
  }

  // @ts-ignore
  const baseUrl = import.meta.env.BASE_URL;

  try {
    worker?.postMessage({
      type: 'INIT',
      modelAssetPath: models[currentModel],
      delegate: currentDelegate,
      maxResults: maxResults,
      scoreThreshold: scoreThreshold,
      baseUrl: baseUrl
    });

    if ((window as any).modelLoadTimeout) clearTimeout((window as any).modelLoadTimeout);

    (window as any).modelLoadTimeout = setTimeout(() => {
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
  const detectBtn = document.getElementById('detect-btn') as HTMLButtonElement;

  if ((window as any).modelLoadTimeout) clearTimeout((window as any).modelLoadTimeout);

  if (loadingOverlay) {
    loadingOverlay.style.display = 'none';
  }

  if (statusMessage) statusMessage.innerText = `Error: ${errorMsg}`;
  if (detectBtn) {
    detectBtn.disabled = false;
    detectBtn.innerText = "Retry";
  }
}

function handleWorkerMessage(event: MessageEvent) {
  const { type } = event.data;
  const loadingOverlay = document.getElementById('loading-overlay');
  const statusMessage = document.getElementById('status-message');
  const detectBtn = document.getElementById('detect-btn') as HTMLButtonElement;

  switch (type) {
    case 'INIT_DONE':
      isWorkerReady = true;
      if ((window as any).modelLoadTimeout) clearTimeout((window as any).modelLoadTimeout);

      if (loadingOverlay) loadingOverlay.style.display = 'none';
      if (statusMessage) statusMessage.innerText = 'Ready';
      if (detectBtn) {
        detectBtn.disabled = false;
        detectBtn.innerText = 'Detect Language';
      }
      break;
    case 'DETECT_RESULT':
      const { result, timestampMs } = event.data;
      const duration = performance.now() - timestampMs;
      const inferenceTimeEl = document.getElementById('inference-time');
      if (inferenceTimeEl) {
        inferenceTimeEl.innerText = `Inference Time: ${duration.toFixed(1)} ms`;
      }
      displayResults(result);
      if (detectBtn) detectBtn.disabled = false;
      if (statusMessage) statusMessage.innerText = 'Done';
      break;
    case 'ERROR':
      console.error('Worker error:', event.data.error);
      handleWorkerError(event.data.error);
      break;
  }
}

function detectLanguage(text: string) {
  if (!worker || !isWorkerReady) return;

  const detectBtn = document.getElementById('detect-btn') as HTMLButtonElement;
  const statusMessage = document.getElementById('status-message');

  detectBtn.disabled = true;
  if (statusMessage) statusMessage.innerText = 'Detecting...';

  worker.postMessage({
    type: 'DETECT',
    text: text,
    timestampMs: performance.now()
  });
}

function displayResults(result: any) {
  const container = document.getElementById('detection-results');
  if (!container || !result.languages || result.languages.length === 0) return;

  container.innerHTML = '';
  const languages = result.languages;

  // Sort by probability desc if not already
  languages.sort((a: any, b: any) => b.probability - a.probability);

  languages.forEach((lang: any) => {
    const item = document.createElement('div');
    item.className = 'classification-item';

    const scorePct = Math.round(lang.probability * 100);

    item.innerHTML = `
      <div class="class-name">${lang.languageCode}</div>
      <div class="class-bar-container">
        <div class="class-bar" style="width: ${scorePct}%"></div>
      </div>
      <div class="class-score">${scorePct}%</div>
    `;
    container.appendChild(item);
  });
}

export function cleanupLanguageDetector() {
  if (worker) {
    worker.postMessage({ type: 'CLEANUP' });
  }
}
