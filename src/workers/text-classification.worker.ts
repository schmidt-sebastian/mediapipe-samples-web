import {
  TextClassifier,
  FilesetResolver
} from '@mediapipe/tasks-text';
import { loadModel } from '../utils/model-loader';

const basePath = '/mediapipe-samples-web/';

import { loadWasmModule } from '../utils/wasm-loader';

let textClassifier: TextClassifier | undefined;
let isInitializing = false;

// Define message types
type WorkerMessage =
  | { type: 'INIT'; modelAssetPath: string; delegate?: 'CPU' | 'GPU'; runningMode?: 'TEXT'; maxResults?: number; scoreThreshold?: number; baseUrl?: string }
  | { type: 'CLASSIFY'; text: string; timestampMs: number }
  | { type: 'CLEANUP' };

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const { type } = event.data;

  switch (type) {
    case 'INIT':
      await initClassifier(event.data);
      break;
    case 'CLASSIFY':
      if (textClassifier && 'text' in event.data) {
        try {
          const result = textClassifier.classify(event.data.text);
          self.postMessage({
            type: 'CLASSIFY_RESULT',
            result,
            timestampMs: event.data.timestampMs
          });
        } catch (error) {
          self.postMessage({ type: 'ERROR', error: error instanceof Error ? error.message : String(error) });
        }
      }
      break;
    case 'CLEANUP':
      if (textClassifier) {
        textClassifier.close();
        textClassifier = undefined;
      }
      break;
  }
};

async function initClassifier(data: any) {
  if (isInitializing) return;
  isInitializing = true;

  try {
    if (textClassifier) {
      textClassifier.close();
      textClassifier = undefined;
    }

    const wasmPath = new URL(`${data.baseUrl || basePath}wasm`, self.location.origin).href;

    // Workaround for Vite + MediaPipe WASM loading in workers
    // Workaround for Vite + MediaPipe WASM loading in workers
    const wasmLoaderUrl = `${wasmPath}/text_wasm_internal.js`;

    // Inject the loader using our shared utility
    const factory = await loadWasmModule(wasmLoaderUrl, ';ModuleFactory;');
    // @ts-ignore
    self.createMediapipeTasksTextModule = factory;



    // ...

    const vision = await FilesetResolver.forTextTasks(wasmPath);

    // Fetch model manually for progress reporting
    const modelBuffer = await loadModel(data.modelAssetPath, (loaded, total) => {
      self.postMessage({
        type: 'LOAD_PROGRESS',
        loaded,
        total
      });
    });

    textClassifier = await TextClassifier.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: data.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
      maxResults: data.maxResults || 3,
      scoreThreshold: data.scoreThreshold || 0,
    });


    self.postMessage({ type: 'INIT_DONE' });

  } catch (error: any) {
    console.error("Text Classifier Worker Init Error:", error);
    self.postMessage({ type: 'ERROR', error: error.message });
  } finally {
    isInitializing = false;
  }
}
