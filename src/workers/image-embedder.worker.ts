import { loadModel } from '../utils/model-loader';
import {
  ImageEmbedder,
  FilesetResolver,
  ImageEmbedderResult
} from '@mediapipe/tasks-vision';

const basePath = '/mediapipe-samples-web/';

import { loadWasmModule } from '../utils/wasm-loader';

// MediaPipe Emscripten fallback
// @ts-ignore
self.createMediapipeTasksVisionModule = self.createMediapipeTasksVisionModule || undefined;

let imageEmbedder: ImageEmbedder | undefined;
let isInitializing = false;

// Define message types
type WorkerMessage =
  | { type: 'INIT'; modelAssetPath: string; delegate?: 'CPU' | 'GPU'; baseUrl?: string }
  | { type: 'EMBED'; image1: ImageBitmap; image2?: ImageBitmap; timestampMs: number }
  | { type: 'CLEANUP' };

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const { type } = event.data;

  switch (type) {
    case 'INIT':
      await initEmbedder(event.data);
      break;
    case 'EMBED':
      if (imageEmbedder && 'image1' in event.data) {
        try {
          const result1 = imageEmbedder.embed(event.data.image1);
          event.data.image1.close();
          
          let result2: ImageEmbedderResult | undefined;
          let similarity: number | undefined;

          if (event.data.image2) {
            result2 = imageEmbedder.embed(event.data.image2);
            event.data.image2.close();
            
            // Compute similarity
            const embedding1 = result1.embeddings[0];
            const embedding2 = result2.embeddings[0];
            similarity = ImageEmbedder.cosineSimilarity(embedding1, embedding2);
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
      if (imageEmbedder) {
        imageEmbedder.close();
        imageEmbedder = undefined;
      }
      break;
  }
};



// ... (imports)

// (Deleted local loadModel)

async function initEmbedder(data: any) {
  if (isInitializing) return;
  isInitializing = true;

  try {
    if (imageEmbedder) {
      imageEmbedder.close();
      imageEmbedder = undefined;
    }

    const wasmPath = new URL(`${data.baseUrl || basePath}wasm`, self.location.origin).href;
    const wasmLoaderUrl = `${wasmPath}/vision_wasm_internal.js`;

    // Inject the loader using our shared utility
    await loadWasmModule(wasmLoaderUrl);

    const vision = await FilesetResolver.forVisionTasks(wasmPath);
    const modelBuffer = await loadModel(data.modelAssetPath, (loaded, total) => {
      self.postMessage({
        type: 'LOAD_PROGRESS',
        loaded,
        total
      });
    });

    if (data.delegate === 'GPU') {
      console.warn('[Worker] GPU Delegate requested.');
    }

    try {
        imageEmbedder = await ImageEmbedder.createFromOptions(vision, {
        baseOptions: {
            modelAssetBuffer: new Uint8Array(modelBuffer),
            delegate: data.delegate === 'GPU' ? 'GPU' : 'CPU',
        }
        });
    } catch (finalError) {
        console.error('ImageEmbedder initialization failed:', finalError);
        if (data.delegate === 'GPU') {
            console.warn('GPU init failed, falling back to CPU', finalError);
            self.postMessage({ type: 'DELEGATE_FALLBACK', newDelegate: 'CPU' });
            imageEmbedder = await ImageEmbedder.createFromOptions(vision, {
              baseOptions: {
                modelAssetBuffer: new Uint8Array(modelBuffer),
                delegate: 'CPU',
              }
            });
        } else {
            throw finalError;
        }
    }

    self.postMessage({ type: 'INIT_DONE' });

  } catch (error: any) {
    console.error("Image Embedder Worker Init Error:", error);
    self.postMessage({ type: 'ERROR', error: error.message });
  } finally {
    isInitializing = false;
  }
}


