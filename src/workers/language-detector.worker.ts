import {
  LanguageDetector,
  FilesetResolver
} from '@mediapipe/tasks-text';

const basePath = '/mediapipe-samples-web/';

// @ts-ignore
if (typeof self.import === 'undefined') {
  // @ts-ignore
  self.import = (url) => import(/* @vite-ignore */ url);
}

let languageDetector: LanguageDetector | undefined;
let isInitializing = false;

// Define message types
type WorkerMessage =
  | { type: 'INIT'; modelAssetPath: string; delegate?: 'CPU' | 'GPU'; maxResults?: number; scoreThreshold?: number; baseUrl?: string }
  | { type: 'DETECT'; text: string; timestampMs: number }
  | { type: 'CLEANUP' };

async function loadModel(path: string) {
  const response = await fetch(path);
  const reader = response.body?.getReader();
  const contentLength = +response.headers.get('Content-Length')!;

  if (!reader) {
    return await response.arrayBuffer();
  }

  let receivedLength = 0;
  const chunks = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    receivedLength += value.length;
    self.postMessage({
      type: 'LOAD_PROGRESS',
      loaded: receivedLength,
      total: contentLength
    });
  }

  const chunksAll = new Uint8Array(receivedLength);
  let position = 0;
  for (let chunk of chunks) {
    chunksAll.set(chunk, position);
    position += chunk.length;
  }

  return chunksAll.buffer;
}

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const { type } = event.data;

  switch (type) {
    case 'INIT':
      await initDetector(event.data);
      break;
    case 'DETECT':
      if (languageDetector && 'text' in event.data) {
        try {
          const result = languageDetector.detect(event.data.text);
          self.postMessage({
            type: 'DETECT_RESULT',
            result,
            timestampMs: event.data.timestampMs
          });
        } catch (error) {
          self.postMessage({ type: 'ERROR', error: error instanceof Error ? error.message : String(error) });
        }
      }
      break;
    case 'CLEANUP':
      if (languageDetector) {
        languageDetector.close();
        languageDetector = undefined;
      }
      break;
  }
};

async function initDetector(data: any) {
  if (isInitializing) return;
  isInitializing = true;

  try {
    if (languageDetector) {
      languageDetector.close();
      languageDetector = undefined;
    }

    const wasmPath = new URL(`${data.baseUrl || basePath}wasm`, self.location.origin).href;

    const vision = await FilesetResolver.forTextTasks(wasmPath);
    const modelBuffer = await loadModel(data.modelAssetPath);

    languageDetector = await LanguageDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: data.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
      maxResults: data.maxResults || 3,
      scoreThreshold: data.scoreThreshold || 0,
    });

    self.postMessage({ type: 'INIT_DONE' });

  } catch (error: any) {
    console.error("Language Detector Worker Init Error:", error);
    self.postMessage({ type: 'ERROR', error: error.message });
  } finally {
    isInitializing = false;
  }
}
