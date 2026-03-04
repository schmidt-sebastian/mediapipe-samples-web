import {
  GestureRecognizer,
  FilesetResolver,
  GestureRecognizerResult
} from '@mediapipe/tasks-vision';

import { BaseWorker } from './base-worker';

// MediaPipe Emscripten fallback
// @ts-ignore
self.createMediapipeTasksVisionModule = self.createMediapipeTasksVisionModule || undefined;

class GestureRecognizerWorker extends BaseWorker<GestureRecognizer> {
  protected async initializeTask(): Promise<void> {
    const wasmLoaderUrl = `${this.getWasmPath()}/vision_wasm_internal.js`;
    await this.loadWasmModule(wasmLoaderUrl, ';ModuleFactory;');

    const vision = await FilesetResolver.forVisionTasks(this.getWasmPath());
    const modelBuffer = await this.loadModelAsset();

    this.taskInstance = await GestureRecognizer.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: this.currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
      minHandDetectionConfidence: this.currentOptions.minHandDetectionConfidence,
      minHandPresenceConfidence: this.currentOptions.minHandPresenceConfidence,
      minTrackingConfidence: this.currentOptions.minTrackingConfidence,
      numHands: this.currentOptions.numHands,
      runningMode: this.currentOptions.runningMode,
    });
  }

  protected async updateOptions(): Promise<void> {
    if (this.taskInstance) {
      await this.taskInstance.setOptions({
        minHandDetectionConfidence: this.currentOptions.minHandDetectionConfidence,
        minHandPresenceConfidence: this.currentOptions.minHandPresenceConfidence,
        minTrackingConfidence: this.currentOptions.minTrackingConfidence,
        numHands: this.currentOptions.numHands,
        runningMode: this.currentOptions.runningMode
      });
    }
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    if (data.type === 'DETECT_IMAGE' || data.type === 'DETECT_VIDEO') {
      const { bitmap, timestampMs } = data;
      const requiredMode = data.type === 'DETECT_IMAGE' ? 'IMAGE' : 'VIDEO';

      if (!this.taskInstance) {
        bitmap.close();
        self.postMessage({ type: 'DETECT_ERROR', error: 'Not initialized' });
        return;
      }

      if (this.currentOptions.runningMode !== requiredMode) {
        this.currentOptions.runningMode = requiredMode;
        await this.updateOptions();
      }

      const startTimeMs = performance.now();
      let result: GestureRecognizerResult;

      try {
        if (requiredMode === 'VIDEO') {
          result = this.taskInstance.recognizeForVideo(bitmap, timestampMs);
        } else {
          result = this.taskInstance.recognize(bitmap);
        }
      } catch (e: any) {
        console.error("Worker recognition error:", e);
        bitmap.close();
        self.postMessage({ type: 'DETECT_ERROR', error: e.message || 'Recognition failed' });
        return;
      }

      const inferenceTime = performance.now() - startTimeMs;
      bitmap.close();

      (self as any).postMessage({
        type: 'DETECT_RESULT',
        mode: requiredMode,
        result: result,
        inferenceTime: inferenceTime
      });
    }
  }
}

new GestureRecognizerWorker();
