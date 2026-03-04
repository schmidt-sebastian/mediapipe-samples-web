import {
  TextEmbedder,
  FilesetResolver,
  TextEmbedderResult
} from '@mediapipe/tasks-text';
import { BaseWorker } from './base-worker';

class TextEmbedderWorker extends BaseWorker<TextEmbedder> {
  protected async initializeTask(): Promise<void> {
    const wasmLoaderUrl = `${this.getWasmPath()}/text_wasm_internal.js`;
    await this.loadWasmModule(wasmLoaderUrl, ';ModuleFactory;');

    const text = await FilesetResolver.forTextTasks(this.getWasmPath());
    const modelBuffer = await this.loadModelAsset();

    this.taskInstance = await TextEmbedder.createFromOptions(text, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: this.currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU',
      }
    });
  }

  protected async updateOptions(): Promise<void> {
  // TextEmbedder does not have runtime updateable options
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    if (data.type === 'EMBED') {
      if (!this.taskInstance || !data.text1) {
        self.postMessage({ type: 'ERROR', error: 'Not initialized or missing text' });
        return;
      }

      try {
        const result1 = this.taskInstance.embed(data.text1);
        let result2: TextEmbedderResult | undefined;
        let similarity: number | undefined;

        if (data.text2) {
          result2 = this.taskInstance.embed(data.text2);
          const embedding1 = result1.embeddings[0];
          const embedding2 = result2.embeddings[0];
          similarity = TextEmbedder.cosineSimilarity(embedding1, embedding2);
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

new TextEmbedderWorker();
