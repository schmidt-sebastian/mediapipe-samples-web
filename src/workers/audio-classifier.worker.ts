import {
  AudioClassifier,
  AudioClassifierResult
} from '@mediapipe/tasks-audio';
import { BaseWorker } from './base-worker';

// @ts-ignore: Pre-declare the factory MediaPipe will try to attach
self.createMediapipeTasksAudioModule = self.createMediapipeTasksAudioModule || undefined;

class AudioClassifierWorker extends BaseWorker<AudioClassifier> {
  protected async initializeTask(): Promise<void> {
    const wasmLoaderUrl = `${this.getWasmPath()}/audio_wasm_internal.js`;
    await this.loadWasmModule(wasmLoaderUrl, ';ModuleFactory;');

    const audioFileset = {
      wasmLoaderPath: `${this.getWasmPath()}/audio_wasm_internal.js`,
      wasmBinaryPath: `${this.getWasmPath()}/audio_wasm_internal.wasm`
    };

    const modelBuffer = await this.loadModelAsset();

    this.taskInstance = await AudioClassifier.createFromOptions(audioFileset, {
      baseOptions: {
        modelAssetBuffer: new Uint8Array(modelBuffer),
        delegate: this.currentOptions.delegate === 'GPU' ? 'GPU' : 'CPU',
      },
      maxResults: this.currentOptions.maxResults,
      scoreThreshold: this.currentOptions.scoreThreshold
    });
  }

  protected async updateOptions(): Promise<void> {
    if (this.taskInstance) {
      await this.taskInstance.setOptions({
        maxResults: this.currentOptions.maxResults,
        scoreThreshold: this.currentOptions.scoreThreshold
      });
    }
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    if (data.type === 'CLASSIFY') {
      const { audioData, sampleRate } = data;
      if (!this.taskInstance) {
        self.postMessage({ type: 'CLASSIFY_ERROR', error: 'Not initialized' });
        return;
      }

      const startTimeMs = performance.now();
      let results: AudioClassifierResult[] = [];

      try {
        results = this.taskInstance.classify(audioData, sampleRate);
      } catch (e: any) {
        console.error("Worker classification error:", e);
        self.postMessage({ type: 'CLASSIFY_ERROR', error: e.message || 'Classification failed' });
        return;
      }

      (self as any).postMessage({
        type: 'CLASSIFY_RESULT',
        results: results,
        inferenceTime: performance.now() - startTimeMs
      });
    }
  }
}

new AudioClassifierWorker();
