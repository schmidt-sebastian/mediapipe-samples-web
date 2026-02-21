import {
  TextClassifier,
  FilesetResolver
} from '@mediapipe/tasks-text';

const basePath = '/mediapipe-samples-web/';

// @ts-ignore
if (typeof self.import === 'undefined') {
  // @ts-ignore
  self.import = (url) => import(/* @vite-ignore */ url);
}

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
    const wasmLoaderUrl = `${wasmPath}/text_wasm_internal.js`;
    try {
      const response = await fetch(wasmLoaderUrl);
      if (response.ok) {
        const loaderCode = await response.text();
        const factory = (0, eval)(loaderCode + ';ModuleFactory;');
        // @ts-ignore
        self.createMediapipeTasksTextModule = factory;
      }
    } catch (e) {
      console.error('Failed to manually load FWASM loader:', e);
    }

    const vision = await FilesetResolver.forTextTasks(wasmPath);

    // We can stick to standard loading since text models are usually small and less prone to the flatbuffer issue specific to some vision models?
    // But let's fail fast if model is missing.
    try {
      const response = await fetch(data.modelAssetPath);
      if (!response.ok) {
        throw new Error(`Failed to load model from ${data.modelAssetPath}: ${response.status} ${response.statusText}`);
      }
    } catch (e) {
      console.error("Model check failed", e);
      throw e;
    }

    textClassifier = await TextClassifier.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: data.modelAssetPath,
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
