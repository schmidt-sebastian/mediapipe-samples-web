import { LanguageDetector, FilesetResolver } from '@mediapipe/tasks-text';
import { BaseWorker } from './base-worker';

class LanguageDetectorWorker extends BaseWorker<LanguageDetector> {
  protected async initializeTask(data: any): Promise<void> {
    const wasmPath = new URL(`${data.baseUrl || import.meta.env.BASE_URL}wasm`, self.location.origin).href;
    const wasmLoaderUrl = `${wasmPath}/text_wasm_internal.js`;

    const factory = await this.loadWasmModule(wasmLoaderUrl, ';ModuleFactory;');
    // @ts-ignore
    self.createMediapipeTasksTextModule = factory;

    const text = await FilesetResolver.forTextTasks(wasmPath);

    const modelBuffer = await this.loadModelAsset();

    this.taskInstance = await LanguageDetector.createFromOptions(text, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: this.currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
      maxResults: this.currentOptions.maxResults,
      scoreThreshold: this.currentOptions.scoreThreshold
    });
  }

  protected async updateOptions(/* data: any */): Promise<void> {
  // Language detector doesn't seem to have a dynamic setOptions
  }

  protected async handleCustomMessage(event: any): Promise<void> {
    const { type, ...data } = event;

    if (type === 'DETECT' && this.taskInstance && 'text' in data) {
      try {
        const result = this.taskInstance.detect(data.text);
        self.postMessage({
          type: 'DETECT_RESULT',
          result,
          timestampMs: data.timestampMs
        });
      } catch (error: any) {
        self.postMessage({ type: 'ERROR', error: error.message || String(error) });
      }
    }
  }
}

new LanguageDetectorWorker();
