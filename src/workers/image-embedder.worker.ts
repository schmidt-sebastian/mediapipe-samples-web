import {
  ImageEmbedder,
  FilesetResolver,
  ImageEmbedderResult
} from '@mediapipe/tasks-vision';
import { BaseWorker } from './base-worker';

// MediaPipe Emscripten fallback
// @ts-ignore
self.createMediapipeTasksVisionModule = self.createMediapipeTasksVisionModule || undefined;

class ImageEmbedderWorker extends BaseWorker<ImageEmbedder> {
  protected async initializeTask(): Promise<void> {
    const wasmLoaderUrl = `${this.getWasmPath()}/vision_wasm_internal.js`;
    await this.loadWasmModule(wasmLoaderUrl);

    const vision = await FilesetResolver.forVisionTasks(this.getWasmPath());
    const modelBuffer = await this.loadModelAsset();

    this.taskInstance = await ImageEmbedder.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: this.currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU',
      }
    });
  }

  protected async updateOptions(): Promise<void> {
    // ImageEmbedder does not have runtime updateable options like runningMode
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    if (data.type === 'EMBED') {
      if (!this.taskInstance || !data.image1) {
        data.image1?.close();
        data.image2?.close();
        self.postMessage({ type: 'ERROR', error: 'Not initialized or missing primary image' });
        return;
      }

      try {
        const result1 = this.taskInstance.embed(data.image1);
        data.image1.close();

        let result2: ImageEmbedderResult | undefined;
        let similarity: number | undefined;

        if (data.image2) {
          result2 = this.taskInstance.embed(data.image2);
          data.image2.close();

          const embedding1 = result1.embeddings[0];
          const embedding2 = result2.embeddings[0];
          similarity = ImageEmbedder.cosineSimilarity(embedding1, embedding2);
        }

        (self as any).postMessage({
          type: 'EMBED_RESULT',
          result1,
          result2,
          similarity,
          timestampMs: data.timestampMs
        });
      } catch (error: any) {
        console.error("Worker embed error:", error);
        self.postMessage({ type: 'ERROR', error: error.message || 'Embed failed' });
      }
    }
  }
}

new ImageEmbedderWorker();
