import {
  TextEmbedder,
  FilesetResolver,
  TextEmbedderResult
} from '@mediapipe/tasks-text';

const basePath = '/mediapipe-samples-web/';

import { loadWasmModule } from '../utils/wasm-loader';

let textEmbedder: TextEmbedder | undefined;
let isInitializing = false;

// Define message types
type WorkerMessage =
  | { type: 'INIT'; modelAssetPath: string; delegate?: 'CPU' | 'GPU'; baseUrl?: string }
  | { type: 'EMBED'; text1: string; text2?: string; timestampMs: number }
  | { type: 'CLEANUP' };

import { loadModel } from '../utils/model-loader';

// ... (rest of imports)



async function initEmbedder(data: any) {
  if (isInitializing) return;
  isInitializing = true;

  try {
    if (textEmbedder) {
      textEmbedder.close();
      textEmbedder = undefined;
    }

    const wasmPath = new URL(`${data.baseUrl || basePath}wasm`, self.location.origin).href;
    const wasmLoaderUrl = `${wasmPath}/text_wasm_internal.js`;

    // Inject the loader using our shared utility
    await loadWasmModule(wasmLoaderUrl, ';ModuleFactory;'); // The 'true' indicates appendScript (removed as it was invalid)
    // The local polyfill `self.createMediapipeTasksTextModule = factory;` is removed as loadWasmModule now handles appending.

    const text = await FilesetResolver.forTextTasks(wasmPath);

    const modelBuffer = await loadModel(data.modelAssetPath, (loaded, total) => {
      self.postMessage({
        type: 'LOAD_PROGRESS',
        loaded,
        total
      });
    });

    textEmbedder = await TextEmbedder.createFromOptions(text, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: data.delegate === 'GPU' ? 'GPU' : 'CPU',
      }
    });

    self.postMessage({ type: 'INIT_DONE' });

  } catch (error: any) {
    console.error("Text Embedder Init Error:", error);
    self.postMessage({ type: 'ERROR', error: error.message });
  } finally {
    isInitializing = false;
  }
}

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const { type } = event.data;

  switch (type) {
    case 'INIT':
      await initEmbedder(event.data);
      break;
    case 'EMBED':
      if (textEmbedder && 'text1' in event.data) {
        try {
          const result1 = textEmbedder.embed(event.data.text1);
          let result2: TextEmbedderResult | undefined;
          let similarity: number | undefined;

          if (event.data.text2) {
            result2 = textEmbedder.embed(event.data.text2);
            // Compute similarity
            const embedding1 = result1.embeddings[0];
            const embedding2 = result2.embeddings[0];
            similarity = TextEmbedder.cosineSimilarity(embedding1, embedding2);
          }

          self.postMessage({
            type: 'EMBED_RESULT',
            result1,
            result2,
            similarity,
            timestampMs: event.data.timestampMs
          });
        } catch (error) {
          self.postMessage({ type: 'ERROR', error: error instanceof Error ? error.message : String(error) });
        }
      }
      break;
    case 'CLEANUP':
      if (textEmbedder) {
        textEmbedder.close();
        textEmbedder = undefined;
      }
      break;
  }
};


