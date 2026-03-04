import { HandLandmarkerResult, DrawingUtils, HandLandmarker } from '@mediapipe/tasks-vision';
import { BaseTask, BaseTaskOptions } from '../components/base-task';

// @ts-ignore
import template from '../templates/hand-landmarker.html?raw';
// @ts-ignore
import HandLandmarkerWorker from '../workers/hand-landmarker.worker.ts?worker';

class HandLandmarkerTask extends BaseTask {
  private drawingUtils: DrawingUtils | undefined;

  private numHands = 2;
  private minHandDetectionConfidence = 0.5;
  private minHandPresenceConfidence = 0.5;
  private minTrackingConfidence = 0.5;

  constructor(options: BaseTaskOptions) {
    super(options);
  }

  protected override onInitializeUI() {
    // Confidence Sliders
    const setupSlider = (id: string, onChange: (val: number) => void) => {
      const input = document.getElementById(id) as HTMLInputElement;
      const valueDisplay = document.getElementById(`${id}-value`)!;
      if (input && valueDisplay) {
        input.addEventListener('input', () => {
          const val = parseFloat(input.value);
          valueDisplay.innerText = val.toString();
          onChange(val);
        });
      }
    };

    setupSlider('min-hand-detection-confidence', (val) => {
      this.minHandDetectionConfidence = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', minHandDetectionConfidence: this.minHandDetectionConfidence });
      this.triggerRedetection();
    });

    setupSlider('min-hand-presence-confidence', (val) => {
      this.minHandPresenceConfidence = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', minHandPresenceConfidence: this.minHandPresenceConfidence });
      this.triggerRedetection();
    });

    setupSlider('min-tracking-confidence', (val) => {
      this.minTrackingConfidence = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', minTrackingConfidence: this.minTrackingConfidence });
      this.triggerRedetection();
    });

    setupSlider('num-hands', (val) => {
      this.numHands = val;
      this.worker?.postMessage({ type: 'SET_OPTIONS', numHands: this.numHands });
      this.triggerRedetection();
    });

    // Custom model options for Hand Landmarker
    this.models = {
      'hand_landmarker': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    };

    if (this.modelSelector) {
      this.modelSelector.updateOptions([
        { label: 'Hand Landmarker', value: 'hand_landmarker', isDefault: true }
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
      numHands: this.numHands,
      minHandDetectionConfidence: this.minHandDetectionConfidence,
      minHandPresenceConfidence: this.minHandPresenceConfidence,
      minTrackingConfidence: this.minTrackingConfidence,
    };
  }

  protected override displayImageResult(result: HandLandmarkerResult) {
    const imageCanvas = document.getElementById('image-canvas') as HTMLCanvasElement;
    const testImage = document.getElementById('test-image') as HTMLImageElement;
    const ctx = imageCanvas.getContext('2d')!;

    imageCanvas.width = testImage.naturalWidth;
    imageCanvas.height = testImage.naturalHeight;

    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    ctx.beginPath();
    ctx.rect(0, 0, imageCanvas.width, imageCanvas.height);
    ctx.clip();

    if (result.landmarks) {
      if (!this.drawingUtils) this.drawingUtils = new DrawingUtils(ctx);
      else this.drawingUtils = new DrawingUtils(ctx);

      for (const landmarks of result.landmarks) {
        this.drawLandmarks(this.drawingUtils, landmarks);
      }
    }
  }

  protected override displayVideoResult(result: HandLandmarkerResult) {
    this.canvasElement.width = this.video.videoWidth;
    this.canvasElement.height = this.video.videoHeight;
    this.canvasCtx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);

    this.canvasCtx.beginPath();
    this.canvasCtx.rect(0, 0, this.canvasElement.width, this.canvasElement.height);
    this.canvasCtx.clip();

    if (result.landmarks) {
      if (!this.drawingUtils) this.drawingUtils = new DrawingUtils(this.canvasCtx);
      else this.drawingUtils = new DrawingUtils(this.canvasCtx);

      for (const landmarks of result.landmarks) {
        this.drawLandmarks(this.drawingUtils, landmarks);
      }
    }
  }

  private drawLandmarks(drawingUtils: DrawingUtils, landmarks: any[]) {
    drawingUtils.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
      color: "#00FF00",
      lineWidth: 5
    });
    drawingUtils.drawLandmarks(landmarks, { color: "#FF0000", lineWidth: 2 });
  }
}

// Singleton instance to support modular cleanup
let activeTask: HandLandmarkerTask | null = null;

export async function setupHandLandmarker(container: HTMLElement) {
  activeTask = new HandLandmarkerTask({
    container,
    template,
    defaultModelName: 'hand_landmarker',
    defaultModelUrl: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
    workerConstructor: HandLandmarkerWorker
  });

  await activeTask.initialize();
}

export function cleanupHandLandmarker() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
