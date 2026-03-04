import {
  ObjectDetector,
  FilesetResolver,
  Detection,
} from '@mediapipe/tasks-vision';
import { BaseWorker } from './base-worker';

// MediaPipe Emscripten fallback
// @ts-ignore
self.createMediapipeTasksVisionModule = self.createMediapipeTasksVisionModule || undefined;

class ObjectDetectionWorker extends BaseWorker<ObjectDetector> {
  protected async initializeTask(): Promise<void> {
    const wasmLoaderUrl = `${this.getWasmPath()}/vision_wasm_internal.js`;
    await this.loadWasmModule(wasmLoaderUrl);

    const vision = await FilesetResolver.forVisionTasks(this.getWasmPath());
    const modelBuffer = await this.loadModelAsset();

    this.taskInstance = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: this.currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
      scoreThreshold: this.currentOptions.scoreThreshold,
      maxResults: this.currentOptions.maxResults,
      runningMode: this.currentOptions.runningMode,
    });
  }

  protected async updateOptions(): Promise<void> {
    if (this.taskInstance) {
      await this.taskInstance.setOptions({
        scoreThreshold: this.currentOptions.scoreThreshold,
        maxResults: this.currentOptions.maxResults,
        runningMode: this.currentOptions.runningMode
      });
    }
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    if (data.type === 'DETECT_IMAGE' || data.type === 'DETECT_VIDEO') {
      const { bitmap, timestampMs } = data;
      const requiredMode = data.type === 'DETECT_IMAGE' ? 'IMAGE' : 'VIDEO';

      if (!this.taskInstance) {
        console.warn('ObjectDetector not initialized yet.');
        bitmap.close();
        self.postMessage({ type: 'DETECT_ERROR', error: 'Not initialized' });
        return;
      }

      if (this.currentOptions.runningMode !== requiredMode) {
        this.currentOptions.runningMode = requiredMode;
        await this.updateOptions();
      }

      const startTimeMs = performance.now();
      let detections: { detections: Detection[] };

      try {
        if (requiredMode === 'VIDEO') {
          detections = this.taskInstance.detectForVideo(bitmap, timestampMs);
        } else {
          detections = this.taskInstance.detect(bitmap);
        }
      } catch (e: any) {
        console.error("Worker detection error:", e);
        bitmap.close();
        self.postMessage({ type: 'DETECT_ERROR', error: e.message || 'Detection failed' });
        return;
      }

      const inferenceTime = performance.now() - startTimeMs;
      bitmap.close();

      (self as any).postMessage({
        type: 'DETECT_RESULT',
        mode: requiredMode,
        result: detections,
        inferenceTime: inferenceTime
      });
    }
  }
}

new ObjectDetectionWorker();
