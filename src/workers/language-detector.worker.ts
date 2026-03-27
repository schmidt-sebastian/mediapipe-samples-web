/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
