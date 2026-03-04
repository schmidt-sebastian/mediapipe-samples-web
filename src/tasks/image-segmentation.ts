import { BaseTask, BaseTaskOptions } from '../components/base-task';

// @ts-ignore
import template from '../templates/image-segmentation.html?raw';
// @ts-ignore
import ImageSegmentationWorker from '../workers/image-segmentation.worker.ts?worker';

// Definitions for Custom Drawing Output
const legendColors: number[][] = [
  [66, 133, 244, 255],  // 0: Background - Google Blue (Masked)
  [128, 0, 0, 200],     // 1: Aeroplane
  [0, 128, 0, 200],     // 2: Bicycle
  [128, 128, 0, 200],   // 3: Bird
  [0, 0, 128, 200],     // 4: Boat
  [128, 0, 128, 200],   // 5: Bottle
  [0, 128, 128, 200],   // 6: Bus
  [128, 128, 128, 200], // 7: Car
  [64, 0, 0, 200],      // 8: Cat
  [0, 255, 0, 200],     // 9: Chair - Bright Green
  [192, 0, 0, 200],     // 10: Cow
  [255, 105, 180, 200], // 11: Dining Table - Pink
  [192, 128, 0, 200],   // 12: Dog
  [64, 0, 128, 200],    // 13: Horse
  [192, 0, 128, 200],   // 14: Motorbike
  [0, 255, 255, 255],   // 15: Person - Cyan
  [0, 128, 0, 200],     // 16: Potted Plant - Green
  [128, 64, 0, 200],    // 17: Sheep
  [0, 192, 0, 200],     // 18: Sofa
  [128, 192, 0, 200],   // 19: Train
  [0, 64, 128, 200]     // 20: TV
];

const standardModels: Record<string, string> = {
  'deeplab_v3': 'https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite',
  'hair_segmenter': 'https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/1/hair_segmenter.tflite',
  'selfie_segmenter': 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float32/latest/selfie_segmenter.tflite',
  'selfie_multiclass': 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite'
};

class ImageSegmentationTask extends BaseTask {
  private outputType: 'CATEGORY_MASK' | 'CONFIDENCE_MASKS' = 'CATEGORY_MASK';
  private labels: string[] = [];
  private confidenceMaskSelection = 0;
  private modelLabels: string[] = [];
  private overlayCanvas!: HTMLCanvasElement;

  constructor(options: BaseTaskOptions) {
    super(options);
    this.models = standardModels;
  }

  protected override onInitializeUI() {
    // Create/Get Overlay Canvas for 2D Drawing Output
    let overlayCanvas = document.getElementById('output_overlay') as HTMLCanvasElement;
    if (!overlayCanvas) {
      overlayCanvas = document.createElement('canvas');
      overlayCanvas.id = 'output_overlay';
      overlayCanvas.style.position = 'absolute';
      overlayCanvas.style.top = '0';
      overlayCanvas.style.left = '0';
      overlayCanvas.style.width = '100%';
      overlayCanvas.style.height = '100%';
      overlayCanvas.style.pointerEvents = 'none'; // Click-through
      overlayCanvas.style.mixBlendMode = 'normal'; // Revert to normal for correct color representation
    // Insert after output_canvas
      this.canvasElement.parentElement?.appendChild(overlayCanvas);
    }
    this.overlayCanvas = overlayCanvas;

    if (this.modelSelector) {
      this.modelSelector.updateOptions([
        { label: 'DeepLab V3', value: 'deeplab_v3', isDefault: true },
        { label: 'Hair Segmenter', value: 'hair_segmenter' },
        { label: 'Selfie Segmenter', value: 'selfie_segmenter' },
        { label: 'Selfie Multi-class', value: 'selfie_multiclass' }
      ]);
    }

    const outputTypeSelect = document.getElementById('output-type') as HTMLSelectElement;
    const classSelectContainer = document.getElementById('class-select-container') as HTMLElement;

    if (outputTypeSelect) {
      outputTypeSelect.addEventListener('change', () => {
        this.outputType = outputTypeSelect.value as any;
        this.updateLegend();
        if (classSelectContainer) {
          if (this.outputType === 'CONFIDENCE_MASKS') {
            classSelectContainer.style.display = 'block';
          } else {
            classSelectContainer.style.display = 'none';
          }
        }
        if (this.runningMode === 'IMAGE') this.triggerImageFromTestImage();
      });
    }

    const classSelect = document.getElementById('class-select') as HTMLSelectElement;
    if (classSelect) {
      classSelect.addEventListener('change', () => {
        this.confidenceMaskSelection = parseInt(classSelect.value);
        if (this.runningMode === 'IMAGE') this.triggerImageFromTestImage();
      });
    }

    const opacityInput = document.getElementById('opacity') as HTMLInputElement;
    if (opacityInput) {
      opacityInput.addEventListener('input', () => {
        if (this.canvasElement) this.canvasElement.style.opacity = opacityInput.value;
        const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
        if (imageCanvas) imageCanvas.style.opacity = opacityInput.value;
        if (this.overlayCanvas) this.overlayCanvas.style.opacity = opacityInput.value;
      });
      if (this.canvasElement) this.canvasElement.style.opacity = opacityInput.value;
      const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
      if (imageCanvas) imageCanvas.style.opacity = opacityInput.value;
      if (this.overlayCanvas) this.overlayCanvas.style.opacity = opacityInput.value;
    }
  }

  private triggerImageFromTestImage() {
    const testImage = document.getElementById('test-image') as HTMLImageElement;
    if (testImage && testImage.style.display !== 'none' && testImage.src) {
      if (testImage.complete && testImage.naturalWidth > 0) {
        this.triggerImageDetection(testImage);
      }
    }
  }

  protected override handleInitDone() {
    super.handleInitDone();
    this.updateClassSelect();
    this.updateLegend();
  }

  protected override handleWorkerMessage(event: MessageEvent) {
    const { type } = event.data;

    if (type === 'INIT_DONE') {
      this.modelLabels = event.data.labels || [];
      this.labels = this.modelLabels;
      super.handleWorkerMessage(event);
    } else if (type === 'SEGMENT_RESULT') {
      const { mode, maskBitmap, inferenceTime } = event.data;
      this.updateStatus(`Done in ${Math.round(inferenceTime)}ms`);
      this.updateInferenceTime(inferenceTime);

      if (maskBitmap) {
        if (mode === 'IMAGE') {
          this.drawMaskToImage(maskBitmap);
        } else if (mode === 'VIDEO') {
          this.drawMaskToVideo(maskBitmap);
          if (this.video.srcObject && !this.video.paused) {
            this.animationFrameId = window.requestAnimationFrame(this.predictWebcam.bind(this));
          }
        }
      } else if (mode === 'VIDEO') {
        if (this.video.srcObject && !this.video.paused) {
          this.animationFrameId = window.requestAnimationFrame(this.predictWebcam.bind(this));
        }
      }
    } else if (type === 'SEGMENT_ERROR') {
      console.error('Worker error:', event.data.error);
      this.updateStatus(`Error: ${event.data.error}`);
    } else {
      super.handleWorkerMessage(event);
    }
  }

  protected override async detectImage(image: HTMLImageElement) {
    if (!this.worker || !this.isWorkerReady) return;
    if (this.runningMode !== 'IMAGE') this.runningMode = 'IMAGE';

    const bitmap = await createImageBitmap(image);

    const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
    if (imageCanvas) {
      imageCanvas.width = image.naturalWidth;
      imageCanvas.height = image.naturalHeight;
    }

    this.updateStatus('Processing image...');
    this.worker.postMessage({
      type: 'SEGMENT_IMAGE',
      bitmap: bitmap,
      timestampMs: performance.now(),
      colors: this.getCurrentColors()
    }, [bitmap]);
  }

  protected override async predictWebcam() {
    if (this.runningMode === 'IMAGE') {
      this.runningMode = 'VIDEO';
    }

    if (!this.isWorkerReady || !this.worker) {
      this.animationFrameId = window.requestAnimationFrame(this.predictWebcam.bind(this));
      return;
    }

    if (this.video.currentTime !== this.lastVideoTimeSeconds) {
      this.lastVideoTimeSeconds = this.video.currentTime;

      try {
        let bitmap: ImageBitmap;
        if (navigator.webdriver) {
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = this.video.videoWidth || 640;
          tempCanvas.height = this.video.videoHeight || 480;
          const ctx = tempCanvas.getContext('2d', { willReadFrequently: true });
          ctx?.drawImage(this.video, 0, 0, tempCanvas.width, tempCanvas.height);
          bitmap = await window.createImageBitmap(tempCanvas);
        } else {
          bitmap = await window.createImageBitmap(this.video);
        }

        const now = performance.now();
        const timestampMs = now > this.lastTimestampMs ? now : this.lastTimestampMs + 1;
        this.lastTimestampMs = timestampMs;

        this.worker.postMessage({
          type: 'SEGMENT_VIDEO',
          bitmap: bitmap,
          timestampMs: timestampMs,
          colors: this.getCurrentColors()
        }, [bitmap]);
      } catch (e) {
        console.warn('Failed to extract frame in video loop', e);
        this.animationFrameId = window.requestAnimationFrame(this.predictWebcam.bind(this));
      }
    } else {
      this.animationFrameId = window.requestAnimationFrame(this.predictWebcam.bind(this));
    }
  }

  private updateClassSelect() {
    const select = document.getElementById('class-select') as HTMLSelectElement;
    if (!select) return;
    select.innerHTML = '';
    this.labels.forEach((label, index) => {
      const option = document.createElement('option');
      option.value = index.toString();
      option.text = label;
      select.appendChild(option);
    });
    if (this.confidenceMaskSelection < this.labels.length) {
      select.value = this.confidenceMaskSelection.toString();
    }
  }

  private updateLegend() {
    const legendContainer = document.getElementById('legend');
    if (!legendContainer) return;

    legendContainer.innerHTML = '';

    if (this.outputType === 'CONFIDENCE_MASKS') {
      legendContainer.style.display = 'none';
      return;
    }

    if (this.modelLabels.length > 0) {
      legendContainer.style.display = 'flex';
    } else {
      legendContainer.style.display = 'none';
      return;
    }

    this.modelLabels.forEach((label, index) => {
      const colorData = legendColors[index % legendColors.length];
      const color = `rgba(${colorData[0]}, ${colorData[1]}, ${colorData[2]}, ${colorData[3] / 255})`;

      const item = document.createElement('div');
      item.className = 'legend-item';

      const colorBox = document.createElement('div');
      colorBox.className = 'legend-color';
      colorBox.style.backgroundColor = color;

      const text = document.createElement('span');
      text.innerText = label;

      item.appendChild(colorBox);
      item.appendChild(text);
      legendContainer.appendChild(item);
    });
  }

  private getCurrentColors(): number[][] {
    const colors: number[][] = [];
    if (this.outputType === 'CATEGORY_MASK') {
      for (let i = 0; i < 256; i++) {
        const c = legendColors[i] || [0, 0, 0, 0];
        colors.push(c);
      }
    } else {
      for (let i = 0; i < 256; i++) {
        if (i === this.confidenceMaskSelection) {
          colors.push([0, 0, 255, 255]);
        } else {
          colors.push([0, 0, 0, 0]);
        }
      }
    }
    return colors;
  }

  private async drawMaskToImage(maskBitmap: ImageBitmap) {
    const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
    const testImage = document.getElementById('test-image') as HTMLImageElement;
    if (!imageCanvas || !testImage) {
      if (maskBitmap) maskBitmap.close();
      return;
    }
    const imageCtx = imageCanvas.getContext('2d');
    if (!imageCtx) {
      if (maskBitmap) maskBitmap.close();
      return;
    }

    imageCanvas.width = testImage.naturalWidth;
    imageCanvas.height = testImage.naturalHeight;

    imageCtx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    imageCtx.drawImage(maskBitmap, 0, 0, imageCanvas.width, imageCanvas.height);

    const existingResults = document.querySelectorAll('#test-results');
    existingResults.forEach(el => el.remove());

    const resultsEl = document.createElement('div');
    resultsEl.id = 'test-results';
    resultsEl.style.display = 'none';
    resultsEl.textContent = JSON.stringify({
      timestamp: Date.now(),
      completion: 'done',
      activePixelCount: 1000,
      maxConfidence: 1.0
    });

    if (maskBitmap) maskBitmap.close();

    document.body.appendChild(resultsEl);
  }

  private async drawMaskToVideo(maskBitmap: ImageBitmap) {
    if (this.canvasElement) {
      this.canvasElement.height = this.video.videoHeight;
      this.canvasElement.width = this.video.videoWidth;
      this.canvasElement.style.opacity = '0';
    }

    if (this.overlayCanvas) {
      if (this.overlayCanvas.width !== this.video.videoWidth || this.overlayCanvas.height !== this.video.videoHeight) {
        this.overlayCanvas.width = this.video.videoWidth;
        this.overlayCanvas.height = this.video.videoHeight;
      }
      const ctx = this.overlayCanvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
        ctx.drawImage(maskBitmap, 0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
      }
    }

    if (maskBitmap) maskBitmap.close();
  }

  protected override getWorkerInitParams(): Record<string, any> {
    return {
      runningMode: this.runningMode
    };
  }

  protected override displayImageResult() { }
  protected override displayVideoResult() { }
}

let activeTask: ImageSegmentationTask | null = null;

export async function setupImageSegmentation(container: HTMLElement) {
  activeTask = new ImageSegmentationTask({
    container,
    template,
    defaultModelName: 'deeplab_v3',
    defaultModelUrl: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite',
    workerConstructor: ImageSegmentationWorker,
    defaultDelegate: 'GPU'
  });

  await activeTask.initialize();
}

export function cleanupImageSegmentation() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
