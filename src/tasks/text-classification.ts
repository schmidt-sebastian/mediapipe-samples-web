/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// @ts-ignore
import template from '../templates/text-classification.html?raw';

import { ModelSelector } from '../components/model-selector';
import { ClassificationResult, ClassificationItem } from '../components/classification-result';

let classificationResultUI: ClassificationResult;
let worker: Worker | undefined;
let isWorkerReady = false;

// Options
const models: Record<string, string> = {
  'bert_classifier': 'https://storage.googleapis.com/mediapipe-models/text_classifier/bert_classifier/float32/1/bert_classifier.tflite',
  'average_word_classifier': 'https://storage.googleapis.com/mediapipe-models/text_classifier/average_word_classifier/float32/1/average_word_classifier.tflite'
};

let currentModel = 'bert_classifier';
let modelSelector: ModelSelector;
let maxResults = 3;
let currentDelegate: 'CPU' | 'GPU' = 'CPU';

export async function setupTextClassification(container: HTMLElement) {
  container.innerHTML = template;

  // UI References
  const classifyBtn = document.getElementById('classify-btn') as HTMLButtonElement;
  const textInput = document.getElementById('text-input') as HTMLTextAreaElement;

  const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
  const maxResultsInput = document.getElementById('max-results') as HTMLInputElement;
  const maxResultsValue = document.getElementById('max-results-value');

  classificationResultUI = new ClassificationResult('classification-results');

  // Event Listeners
  classifyBtn.addEventListener('click', () => {
    if (textInput.value.trim()) {
      classifyText(textInput.value);
    }
  });

  // Sample Buttons
  const sampleBtns = container.querySelectorAll('.sample-btn');
  sampleBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      const text = (e.currentTarget as HTMLElement).dataset.text;
      if (text) {
        textInput.value = text;
        classifyText(text);
      }
    });
  });

  modelSelector = new ModelSelector(
    'model-selector-container',
    [
      { label: 'BERT Classifier', value: 'bert_classifier', isDefault: true },
      { label: 'Average Word Classifier', value: 'average_word_classifier' }
    ],
    async (selection) => {
      if (selection.type === 'standard') {
        currentModel = selection.value;
      } else if (selection.type === 'custom') {
        models['custom'] = URL.createObjectURL(selection.file);
        currentModel = 'custom';
      }
      isWorkerReady = false;
      await initClassifier();
    }
  );

  if (delegateSelect) {
    delegateSelect.addEventListener('change', (e) => {
      currentDelegate = (e.target as HTMLSelectElement).value as 'CPU' | 'GPU';
      isWorkerReady = false;
      initClassifier();
    });
  }

  if (maxResultsInput) {
    maxResultsInput.addEventListener('input', (e) => {
      maxResults = parseInt((e.target as HTMLInputElement).value);
      if (maxResultsValue) maxResultsValue.innerText = maxResults.toString();
      isWorkerReady = false;
      initClassifier();
    });
  }

  // Init Worker
  if (!worker) {
    worker = new Worker(new URL('../workers/text-classification.worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = handleWorkerMessage;
  }

  // Initial Load
  await initClassifier();
}

async function initClassifier() {
  const loadingOverlay = document.getElementById('loading-overlay');
  if (loadingOverlay) {
    loadingOverlay.style.display = 'flex';
    // Reset text if retrying
    const loadingText = loadingOverlay.querySelector('.loading-text');
    if (loadingText) loadingText.textContent = 'Loading Model...';
  }

  const statusMessage = document.getElementById('status-message');
  if (statusMessage) statusMessage.innerText = 'Loading Model...';

  const classifyBtn = document.getElementById('classify-btn') as HTMLButtonElement;
  if (classifyBtn) {
    classifyBtn.disabled = true;
  }

  // @ts-ignore
  const baseUrl = import.meta.env.BASE_URL;

  try {
    // Send INIT
    worker?.postMessage({
      type: 'INIT',
      modelAssetPath: currentModel === 'custom' && models['custom'] ? models['custom'] : models[currentModel],
      delegate: currentDelegate,
      maxResults: maxResults,
      baseUrl: baseUrl
    });

    // We can't easily await the specific worker response here without refactoring the message handler 
    // to be request/response based or having a shared state.
    // So we will set a safety timeout in the UI state.

    // Clear any existing timeout
    if ((window as any).modelLoadTimeout) clearTimeout((window as any).modelLoadTimeout);

    (window as any).modelLoadTimeout = setTimeout(() => {
      if (!isWorkerReady) {
        console.error("Model load timed out");
        handleWorkerError("Model loading timed out. Please try refreshing or switching models.");
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
  const classifyBtn = document.getElementById('classify-btn') as HTMLButtonElement;

  if ((window as any).modelLoadTimeout) clearTimeout((window as any).modelLoadTimeout);

  if (loadingOverlay) {
    // Show error state in overlay or hide it?
    // Let's hide it and show error in status, or show retry button in overlay.
    // For now, hide overlay so user can see UI.
    loadingOverlay.style.display = 'none';
  }

  if (statusMessage) statusMessage.innerText = `Error: ${errorMsg}`;
  if (classifyBtn) {
    classifyBtn.disabled = false;
    classifyBtn.innerText = "Retry";
  }
}

function handleWorkerMessage(event: MessageEvent) {
  const { type } = event.data;
  const loadingOverlay = document.getElementById('loading-overlay');
  const statusMessage = document.getElementById('status-message');
  const classifyBtn = document.getElementById('classify-btn') as HTMLButtonElement;

  switch (type) {
    case 'LOAD_PROGRESS':
      const { loaded, total } = event.data;
      modelSelector?.showProgress(loaded, total);
      if (loadingOverlay && total > 0) {
        const percent = Math.round((loaded / total) * 100);
        const loadingText = loadingOverlay.querySelector('.loading-text');
        if (loadingText) loadingText.textContent = `Loading Model... ${percent}%`;
      }
      if (statusMessage && total > 0) {
        const percent = Math.round((loaded / total) * 100);
        statusMessage.innerText = `Loading Model... ${percent}%`;
      }
      if (loaded >= total) setTimeout(() => modelSelector?.hideProgress(), 500);
      break;

    case 'INIT_DONE':
      modelSelector?.hideProgress();
      isWorkerReady = true;
      if ((window as any).modelLoadTimeout) clearTimeout((window as any).modelLoadTimeout);

      if (loadingOverlay) loadingOverlay.style.display = 'none';
      if (statusMessage) statusMessage.innerText = 'Ready';
      if (classifyBtn) {
        classifyBtn.disabled = false;
        classifyBtn.innerText = 'Classify';
      }
      break;
    case 'CLASSIFY_RESULT':
      const { result, timestampMs } = event.data;
      const duration = performance.now() - timestampMs;
      const inferenceTimeEl = document.getElementById('inference-time');
      if (inferenceTimeEl) {
        inferenceTimeEl.innerText = `Inference Time: ${duration.toFixed(1)} ms`;
      }
      displayResults(result);
      if (classifyBtn) classifyBtn.disabled = false;
      if (statusMessage) statusMessage.innerText = 'Done';
      break;
    case 'ERROR':
      console.error('Worker error:', event.data.error);
      handleWorkerError(event.data.error);
      break;
  }
}

function classifyText(text: string) {
  if (!worker || !isWorkerReady) return;

  const classifyBtn = document.getElementById('classify-btn') as HTMLButtonElement;
  const statusMessage = document.getElementById('status-message');

  classifyBtn.disabled = true;
  if (statusMessage) statusMessage.innerText = 'Classifying...';

  worker.postMessage({
    type: 'CLASSIFY',
    text: text,
    timestampMs: performance.now()
  });
}

function displayResults(result: any) {
  if (!result || !result.classifications || result.classifications.length === 0 || !classificationResultUI) return;

  const categories = result.classifications[0].categories;

  // Sort by score desc
  categories.sort((a: any, b: any) => b.score - a.score);

  const items: ClassificationItem[] = categories.map((c: any) => ({
    label: c.categoryName,
    score: c.score
  }));

  classificationResultUI.updateResults(items);
}

export function cleanupTextClassification() {
  if (worker) {
    worker.terminate();
    worker = undefined;
  }
}
