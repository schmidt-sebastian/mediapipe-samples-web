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

/// <reference types="vite/client" />
import { InteractiveSegmenter, FilesetResolver } from '@mediapipe/tasks-vision';
import { BaseWorker } from './base-worker';

class InteractiveSegmenterWorker extends BaseWorker<InteractiveSegmenter> {
  protected async initializeTask(/* data: any */): Promise<void> {
    const wasmPath = new URL(`${import.meta.env.BASE_URL}wasm`, self.location.origin).href;
    const wasmLoaderUrl = `${wasmPath}/vision_wasm_internal.js`;

    await this.loadWasmModule(wasmLoaderUrl, ';ModuleFactory;');

    const vision = await FilesetResolver.forVisionTasks(wasmPath);

    this.taskInstance = await InteractiveSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: this.currentOptions.modelAssetPath,
        delegate: this.currentOptions.delegate || 'GPU',
      },
      outputCategoryMask: true,
      outputConfidenceMasks: false
    });
  }

  protected async updateOptions(data: any): Promise<void> {
    if (this.taskInstance) {
      await this.taskInstance.setOptions(data);
    }
  }

  protected async handleCustomMessage(data: any): Promise<void> {
    const { type, ...rest } = data;

    if (type === 'SEGMENT' && this.taskInstance) {
      try {
        const { bitmap, pt } = rest;
        const timestampMs = performance.now();

        const result = this.taskInstance.segment(bitmap, {
          keypoint: { x: pt.x, y: pt.y }
        });

        const categoryMask = result.categoryMask;
        let maskData: Uint8Array | Float32Array | null = null;
        let width = 0;
        let height = 0;

        if (categoryMask) {
          width = categoryMask.width;
          height = categoryMask.height;
          maskData = categoryMask.getAsUint8Array();
        }

        const transfer = maskData ? [maskData.buffer] : [];
        (self as any).postMessage({
          type: 'SEGMENT_RESULT',
          maskData,
          width,
          height,
          inferenceTime: performance.now() - timestampMs
        }, transfer);

      } catch (error: any) {
        console.error("Segmentation Error:", error);
        self.postMessage({ type: 'ERROR', error: error.message });
      }
    }
  }
}

new InteractiveSegmenterWorker();
