import {
  HandLandmarker,
  FilesetResolver,
  HandLandmarkerResult
} from '@mediapipe/tasks-vision';

import { BaseWorker } from './base-worker';

// MediaPipe Emscripten fallback
// @ts-ignore
self.createMediapipeTasksVisionModule = self.createMediapipeTasksVisionModule || undefined;

class HandLandmarkerWorker extends BaseWorker<HandLandmarker> {
  protected async initializeTask(): Promise<void> {
    const wasmLoaderUrl = `${this.getWasmPath()}/vision_wasm_internal.js`;
    await this.loadWasmModule(wasmLoaderUrl);

    const vision = await FilesetResolver.forVisionTasks(this.getWasmPath());
    const modelBuffer = await this.loadModelAsset();

    this.taskInstance = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: this.currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
      numHands: this.currentOptions.numHands || 2,
      minHandDetectionConfidence: this.currentOptions.minHandDetectionConfidence || 0.5,
      minHandPresenceConfidence: this.currentOptions.minHandPresenceConfidence || 0.5,
      minTrackingConfidence: this.currentOptions.minTrackingConfidence || 0.5,
      runningMode: this.currentOptions.runningMode,
    });
  }

  protected async updateOptions(): Promise<void> {
    if (this.taskInstance) {
      await this.taskInstance.setOptions({
        numHands: this.currentOptions.numHands,
        minHandDetectionConfidence: this.currentOptions.minHandDetectionConfidence,
        minHandPresenceConfidence: this.currentOptions.minHandPresenceConfidence,
        minTrackingConfidence: this.currentOptions.minTrackingConfidence,
        runningMode: this.currentOptions.runningMode
      });
    }
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    if (data.type === 'DETECT_IMAGE' || data.type === 'DETECT_VIDEO') {
      const { bitmap, timestampMs } = data;
      const requiredMode = data.type === 'DETECT_IMAGE' ? 'IMAGE' : 'VIDEO';

      if (!this.taskInstance) {
        console.warn('HandLandmarker not initialized yet.');
        bitmap.close();
        self.postMessage({ type: 'DETECT_ERROR', error: 'Not initialized' });
        return;
      }

      if (this.currentOptions.runningMode !== requiredMode) {
        this.currentOptions.runningMode = requiredMode;
        await this.updateOptions();
      }

      const startTimeMs = performance.now();
      let result: HandLandmarkerResult;

      try {
        if (requiredMode === 'VIDEO') {
          result = this.taskInstance.detectForVideo(bitmap, timestampMs);
        } else {
          result = this.taskInstance.detect(bitmap);
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
        result: result,
        inferenceTime: inferenceTime
      });
    }
  }
}

new HandLandmarkerWorker();
