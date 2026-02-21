import {
  TextEmbedder,
  FilesetResolver,
  TextEmbedderResult
} from '@mediapipe/tasks-text';

const basePath = '/mediapipe-samples-web/';

// @ts-ignore
if (typeof self.import === 'undefined') {
  // @ts-ignore
  self.import = (url) => import(/* @vite-ignore */ url);
}

let textEmbedder: TextEmbedder | undefined;
let isInitializing = false;

// Define message types
type WorkerMessage =
  | { type: 'INIT'; modelAssetPath: string; delegate?: 'CPU' | 'GPU'; baseUrl?: string }
  | { type: 'EMBED'; text1: string; text2?: string; timestampMs: number }
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

async function initEmbedder(data: any) {
  if (isInitializing) return;
  isInitializing = true;

  try {
    if (textEmbedder) {
      textEmbedder.close();
      textEmbedder = undefined;
    }

    const wasmPath = new URL(`${data.baseUrl || basePath}wasm`, self.location.origin).href;

    const vision = await FilesetResolver.forTextTasks(wasmPath);
    const modelBuffer = await loadModel(data.modelAssetPath);

    textEmbedder = await TextEmbedder.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: data.delegate === 'GPU' ? 'GPU' : 'CPU',
      }
    });

    self.postMessage({ type: 'INIT_DONE' });

  } catch (error: any) {
    console.error("Text Embedder Worker Init Error:", error);
    self.postMessage({ type: 'ERROR', error: error.message });
  } finally {
    isInitializing = false;
  }
}
