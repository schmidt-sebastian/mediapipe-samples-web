import {
  ImageEmbedder,
  FilesetResolver,
  ImageEmbedderResult
} from '@mediapipe/tasks-vision';

const basePath = '/mediapipe-samples-web/';

// @ts-ignore
if (typeof self.import === 'undefined') {
  // @ts-ignore
  self.import = (url) => import(/* @vite-ignore */ url);
}

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

    try {
      const response = await fetch(wasmLoaderUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch WASM loader: ${response.status} ${response.statusText}`);
      }
      const loaderCode = await response.text();
      (0, eval)(loaderCode);
    } catch (e) {
      console.error('Failed to manually load WASM loader:', e);
    }

    const vision = await FilesetResolver.forVisionTasks(wasmPath);
    const modelBuffer = await loadModel(data.modelAssetPath);

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
