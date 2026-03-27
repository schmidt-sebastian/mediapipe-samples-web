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

import { ObjectDetectorResult } from '@mediapipe/tasks-vision';
import { BaseTask, BaseTaskOptions } from '../components/base-task';

// @ts-ignore
import template from '../templates/object-detection.html?raw';
// @ts-ignore
import ObjectDetectionWorker from '../workers/object-detection.worker.ts?worker';

class ObjectDetectionTask extends BaseTask {
  private scoreThreshold = 0.5;
  private maxResults = 3;

  constructor(options: BaseTaskOptions) {
    super({ ...options, defaultDelegate: 'CPU' });
  }

  protected override onInitializeUI() {
    // Check for delegate override in URL
    const urlParams = new URLSearchParams(window.location.search);
    const delegateParam = urlParams.get('delegate');
    if (delegateParam === 'GPU') {
      console.warn('GPU requested via URL, but Object Detection Worker only supports CPU. Forcing CPU.');
      this.currentDelegate = 'CPU';

      const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
      if (delegateSelect) delegateSelect.value = 'CPU';
    }

    const setupSlider = (id: string, onChange: (val: number) => void) => {
      const input = document.getElementById(id) as HTMLInputElement;
      const valueDisplay = document.getElementById(`${id}-value`)!;
      if (input && valueDisplay) {
        input.addEventListener('input', () => {
          const val = id === 'max-results' ? parseInt(input.value) : parseFloat(input.value);
          valueDisplay.innerText = val.toString();
          onChange(val);
        });
      }
    };

    setupSlider('max-results', (val) => {
      this.maxResults = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', maxResults: this.maxResults });
      this.triggerRedetection();
    });

    setupSlider('score-threshold', (val) => {
      this.scoreThreshold = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', scoreThreshold: this.scoreThreshold });
      this.triggerRedetection();
    });

    // Custom model options for Object Detection
    this.models = {
      'efficientdet_lite0': 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite',
      'efficientdet_lite2': 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float32/1/efficientdet_lite2.tflite',
      'ssd_mobilenet_v2': 'https://storage.googleapis.com/mediapipe-models/object_detector/ssd_mobilenet_v2/float32/1/ssd_mobilenet_v2.tflite'
    };

    if (this.modelSelector) {
      this.modelSelector.updateOptions([
        { label: 'EfficientDet-Lite0', value: 'efficientdet_lite0', isDefault: true },
        { label: 'EfficientDet-Lite2', value: 'efficientdet_lite2' },
        { label: 'SSD MobileNet V2', value: 'ssd_mobilenet_v2' }
      ]);
    }
  }

  private triggerRedetection() {
    if (this.runningMode === 'IMAGE') {
      const testImage = document.getElementById('test-image') as HTMLImageElement;
      if (testImage && testImage.src) {
        this.detectImage(testImage);
      }
    }
  }

  protected override getWorkerInitParams(): Record<string, any> {
    return {
      scoreThreshold: this.scoreThreshold,
      maxResults: this.maxResults,
    };
  }

  protected override displayImageResult(result: ObjectDetectorResult) {
    const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
    const testImage = document.getElementById('test-image') as HTMLImageElement;
    const ctx = imageCanvas.getContext('2d')!;

    imageCanvas.width = testImage.naturalWidth;
    imageCanvas.height = testImage.naturalHeight;

    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);

    if (result.detections) {
      for (let detection of result.detections) {
        this.drawDetection(ctx, detection, false);
      }

      // Keep existing logic to expose results for testing
      const resultsEl = document.createElement('div');
      resultsEl.id = 'test-results';
      resultsEl.style.display = 'none';
      resultsEl.textContent = JSON.stringify(result.detections);
      const oldResults = document.getElementById('test-results');
      if (oldResults) oldResults.remove();
      document.body.appendChild(resultsEl);
    }
  }

  protected override displayVideoResult(result: ObjectDetectorResult) {
    this.canvasElement.width = this.video.videoWidth;
    this.canvasElement.height = this.video.videoHeight;
    this.canvasCtx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);

    if (result.detections) {
      for (let detection of result.detections) {
        this.drawDetection(this.canvasCtx, detection, true);
      }
    }
  }

  private drawDetection(ctx: CanvasRenderingContext2D, detection: any, mirror: boolean) {
    ctx.beginPath();
    ctx.lineWidth = 4;
    ctx.strokeStyle = '#007f8b';

    const { originX, originY, width, height } = detection.boundingBox!;
    let x = originX;

    ctx.strokeRect(x, originY, width, height);
    ctx.fillStyle = '#007f8b';
    ctx.font = '16px sans-serif';

    const category = detection.categories[0];
    const score = category.score ? Math.round(category.score * 100) : 0;
    const labelText = `${category.categoryName} - ${score}%`;
    const textWidth = ctx.measureText(labelText).width;

    if (mirror) {
      ctx.save();
      const centerX = x + (textWidth + 10) / 2;
      const centerY = originY + 12.5; 

      ctx.translate(centerX, centerY);
      ctx.scale(-1, 1);
      ctx.translate(-centerX, -centerY);

      ctx.fillRect(x, originY, textWidth + 10, 25);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, x + 5, originY + 18);
      ctx.restore();
    } else {
      ctx.fillRect(x, originY, textWidth + 10, 25);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, x + 5, originY + 18);
    }
  }
}

// Singleton instance
let activeTask: ObjectDetectionTask | null = null;

export async function setupObjectDetection(container: HTMLElement) {
  activeTask = new ObjectDetectionTask({
    container,
    template,
    defaultModelName: 'efficientdet_lite0',
    defaultModelUrl: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite',
    workerConstructor: ObjectDetectionWorker
  });

  await activeTask.initialize();
}

export function cleanupObjectDetection() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}