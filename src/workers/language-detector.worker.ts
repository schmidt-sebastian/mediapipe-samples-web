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
      console.error('Failed to manually load WASM loader:', e);
    }

    const vision = await FilesetResolver.forTextTasks(wasmPath);

    try {
      const response = await fetch(data.modelAssetPath);
      if (!response.ok) {
        throw new Error(`Failed to load model from ${data.modelAssetPath}: ${response.status} ${response.statusText}`);
      }
    } catch (e) {
      console.error("Model check failed", e);
      throw e;
    }

    languageDetector = await LanguageDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: data.modelAssetPath,
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
