import {
  ImageClassifier,
  FilesetResolver,
  ImageClassifierResult
} from '@mediapipe/tasks-vision';
import { BaseWorker } from './base-worker';

class ImageClassifierWorker extends BaseWorker<ImageClassifier> {
  protected async initializeTask(): Promise<void> {
    const wasmLoaderUrl = `${this.getWasmPath()}/vision_wasm_internal.js`;
    await this.loadWasmModule(wasmLoaderUrl, ';ModuleFactory;');

    const vision = await FilesetResolver.forVisionTasks(this.getWasmPath());
    const modelBuffer = await this.loadModelAsset();

    this.taskInstance = await ImageClassifier.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: this.currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
      runningMode: this.currentOptions.runningMode || 'IMAGE',
      maxResults: this.currentOptions.maxResults || 5,
      scoreThreshold: this.currentOptions.scoreThreshold || 0.0,
    });
  }

  protected async updateOptions(): Promise<void> {
    if (this.taskInstance) {
      await this.taskInstance.setOptions({
        runningMode: this.currentOptions.runningMode,
        maxResults: this.currentOptions.maxResults,
        scoreThreshold: this.currentOptions.scoreThreshold,
      });
    }
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    if (data.type === 'DETECT_IMAGE' || data.type === 'DETECT_VIDEO') {
      const { bitmap, timestampMs } = data;
      const requiredMode = data.type === 'DETECT_IMAGE' ? 'IMAGE' : 'VIDEO';

      if (!this.taskInstance) {
        bitmap?.close();
        self.postMessage({ type: 'ERROR', error: 'Not initialized' });
        return;
      }

      if (this.currentOptions.runningMode !== requiredMode) {
        this.currentOptions.runningMode = requiredMode;
        await this.updateOptions();
      }

      const startTimeMs = performance.now();
      let result: ImageClassifierResult;

      try {
        if (requiredMode === 'VIDEO') {
          result = this.taskInstance.classifyForVideo(bitmap, timestampMs);
        } else {
          result = this.taskInstance.classify(bitmap);
        }
      } catch (e: any) {
        console.error("Worker classification error:", e);
        bitmap?.close();
        self.postMessage({ type: 'ERROR', error: e.message || 'Classification failed' });
        return;
      }

      const inferenceTime = performance.now() - startTimeMs;
      bitmap?.close();

      (self as any).postMessage({
        type: 'DETECT_RESULT',
        mode: requiredMode,
        result: result,
        inferenceTime: inferenceTime
      });
    }
  }
}

new ImageClassifierWorker();
