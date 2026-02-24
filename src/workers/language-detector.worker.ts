import { loadModel } from '../utils/model-loader';
import {
  LanguageDetector,
  FilesetResolver
} from '@mediapipe/tasks-text';

const basePath = '/mediapipe-samples-web/';

import { loadWasmModule } from '../utils/wasm-loader';

let languageDetector: LanguageDetector | undefined;
let isInitializing = false;

// Define message types
type WorkerMessage =
  | { type: 'INIT'; modelAssetPath: string; delegate?: 'CPU' | 'GPU'; maxResults?: number; scoreThreshold?: number; baseUrl?: string }
  | { type: 'DETECT'; text: string; timestampMs: number }
  | { type: 'CLEANUP' };


// ... (rest of imports)

// (Deleted local loadModel function)

async function initDetector(data: any) {
  if (isInitializing) return;
  isInitializing = true;

  try {
    if (languageDetector) {
      languageDetector.close();
      languageDetector = undefined;
    }

    const wasmPath = new URL(`${data.baseUrl || basePath}wasm`, self.location.origin).href;
    const wasmLoaderUrl = `${wasmPath}/text_wasm_internal.js`;

    // Inject the loader using our shared utility
    const factory = await loadWasmModule(wasmLoaderUrl, ';ModuleFactory;');
    // @ts-ignore
    self.createMediapipeTasksTextModule = factory;

    const text = await FilesetResolver.forTextTasks(wasmPath);

    const modelBuffer = await loadModel(data.modelAssetPath, (loaded, total) => {
      self.postMessage({
        type: 'LOAD_PROGRESS',
        loaded,
        total
      });
    });

    languageDetector = await LanguageDetector.createFromOptions(text, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: data.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
      maxResults: data.maxResults,
      scoreThreshold: data.scoreThreshold
    });

    self.postMessage({ type: 'INIT_DONE' });

  } catch (error: any) {
    console.error("Language Detector Init Error:", error);
    self.postMessage({ type: 'ERROR', error: error.message });
  } finally {
    isInitializing = false;
  }
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


