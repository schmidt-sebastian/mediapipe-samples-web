import {
  TextClassifier,
  FilesetResolver
} from '@mediapipe/tasks-text';
import { BaseWorker } from './base-worker';

class TextClassifierWorker extends BaseWorker<TextClassifier> {
  protected async initializeTask(): Promise<void> {
    const wasmLoaderUrl = `${this.getWasmPath()}/text_wasm_internal.js`;
    // Workaround for Vite + MediaPipe WASM loading in workers
    const factory = await this.loadWasmModule(wasmLoaderUrl, ';ModuleFactory;');
    // @ts-ignore
    self.createMediapipeTasksTextModule = factory;

    const vision = await FilesetResolver.forTextTasks(this.getWasmPath());
    const modelBuffer = await this.loadModelAsset();

    this.taskInstance = await TextClassifier.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: this.currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
      maxResults: this.currentOptions.maxResults || 3,
      scoreThreshold: this.currentOptions.scoreThreshold || 0,
    });
  }

  protected async updateOptions(): Promise<void> {
    // TextClassifier might not have dynamically updateable options in the same way,
    // but if it does, we implement it here. Otherwise, leave empty.
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    if (data.type === 'CLASSIFY') {
      if (!this.taskInstance || !data.text) {
        self.postMessage({ type: 'ERROR', error: 'Not initialized or missing text' });
        return;
      }

      try {
        const result = this.taskInstance.classify(data.text);
        (self as any).postMessage({
          type: 'CLASSIFY_RESULT',
          result,
          timestampMs: data.timestampMs
        });
      } catch (error: any) {
        console.error("Worker classify error:", error);
        self.postMessage({ type: 'ERROR', error: error.message || 'Classification failed' });
      }
    }
  }
}

new TextClassifierWorker();
