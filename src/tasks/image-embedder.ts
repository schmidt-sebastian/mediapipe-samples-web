import { BaseTask, BaseTaskOptions } from '../components/base-task';

// @ts-ignore
import template from '../templates/image-embedder.html?raw';
// @ts-ignore
import ImageEmbedderWorker from '../workers/image-embedder.worker.ts?worker';

class ImageEmbedderTask extends BaseTask {
  private image1: HTMLImageElement | null = null;
  private image2: HTMLImageElement | null = null;

  constructor(options: BaseTaskOptions) {
    super(options);
    this.models = {
      'mobilenet_v3_small': 'https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite',
      'mobilenet_v3_large': 'https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_large/float32/1/mobilenet_v3_large.tflite'
    };
  }

  protected override onInitializeUI() {
    this.image1 = document.getElementById('image-1') as HTMLImageElement;
    this.image2 = document.getElementById('image-2') as HTMLImageElement;
    const display1 = document.getElementById('display-area-1')!;
    const display2 = document.getElementById('display-area-2')!;

    const setImage = (img: HTMLImageElement, display: HTMLElement, src: string) => {
      img.src = src;
      img.style.display = 'block';
      display.classList.add('has-image');
      const placeholder = display.querySelector('.placeholder-text') as HTMLElement;
      if (placeholder) placeholder.style.display = 'none';
      this.checkEnableButton();
    };

    const handleUpload = (input: HTMLInputElement, img: HTMLImageElement, display: HTMLElement) => {
      input.value = '';
      input.click();
      input.onchange = (e) => {
        const file = (e.target as HTMLInputElement).files?.[0];
        if (file) {
          const reader = new FileReader();
          reader.onload = (e) => {
            setImage(img, display, e.target?.result as string);
          };
          reader.readAsDataURL(file);
        }
      };
    };

    // Attach listeners to samples
    document.querySelectorAll('.sample-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const targetId = (btn as HTMLElement).dataset.target;
        const src = (btn as HTMLElement).dataset.src;
        if (targetId === '1' && src && this.image1) setImage(this.image1, display1, src);
        if (targetId === '2' && src && this.image2) setImage(this.image2, display2, src);
      });
    });

    // Attach listeners to upload buttons
    document.querySelectorAll('.upload-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const targetId = (btn as HTMLElement).dataset.target;
        if (targetId === '1' && this.image1) handleUpload(document.getElementById('image-upload-1') as HTMLInputElement, this.image1, display1);
        if (targetId === '2' && this.image2) handleUpload(document.getElementById('image-upload-2') as HTMLInputElement, this.image2, display2);
      });
    });

    // Make display areas clickable for upload
    if (display1 && this.image1) {
      display1.addEventListener('click', () => handleUpload(document.getElementById('image-upload-1') as HTMLInputElement, this.image1!, display1));
    }
    if (display2 && this.image2) {
      display2.addEventListener('click', () => handleUpload(document.getElementById('image-upload-2') as HTMLInputElement, this.image2!, display2));
    }

    if (this.image1) this.image1.onload = () => this.checkEnableButton();
    if (this.image2) this.image2.onload = () => this.checkEnableButton();

    this.models = {
      'mobilenet_v3_small': 'https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite',
      'mobilenet_v3_large': 'https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_large/float32/1/mobilenet_v3_large.tflite'
    };

    if (this.modelSelector) {
      this.modelSelector.updateOptions([
        { label: 'MobileNet V3 Small', value: 'mobilenet_v3_small', isDefault: true },
        { label: 'MobileNet V3 Large', value: 'mobilenet_v3_large' }
      ]);
    }
  }

  private checkEnableButton() {
    if (this.image1 && this.image2 && this.image1.src && this.image2.src && this.isWorkerReady) {
      this.computeSimilarity(this.image1, this.image2);
    }
  }

  protected override handleInitDone() {
    super.handleInitDone();

    const loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay) loadingOverlay.style.display = 'none';

    this.checkEnableButton();
  }

  protected override async initializeDetector() {
    const loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay) {
      loadingOverlay.style.display = 'flex';
      const loadingText = loadingOverlay.querySelector('.loading-text');
      if (loadingText) loadingText.textContent = 'Loading Model...';
    }

    await super.initializeDetector();
  }

  protected override handleWorkerMessage(event: MessageEvent) {
    const { type } = event.data;

    if (type === 'EMBED_RESULT') {
      const { similarity, timestampMs } = event.data;
      const duration = performance.now() - timestampMs;
      this.updateInferenceTime(duration);
      this.displayResults(similarity);
      this.updateStatus('Done');
    } else {
      super.handleWorkerMessage(event);
    }
  }

  private async computeSimilarity(img1: HTMLImageElement, img2: HTMLImageElement) {
    if (!this.worker || !this.isWorkerReady) return;

    this.updateStatus('Computing...');

    const bitmap1 = await createImageBitmap(img1);
    const bitmap2 = await createImageBitmap(img2);

    this.worker.postMessage({
      type: 'EMBED',
      image1: bitmap1,
      image2: bitmap2,
      timestampMs: performance.now()
    }, [bitmap1, bitmap2]);
  }

  private displayResults(similarity: number) {
    const valueEl = document.getElementById('similarity-value');
    if (valueEl && similarity !== undefined) {
      valueEl.innerText = similarity.toFixed(4);
    } else if (valueEl) {
      valueEl.innerText = '--';
    }
  }

  protected override getWorkerInitParams(): Record<string, any> {
    return {};
  }

  protected override displayImageResult() { }
  protected override displayVideoResult() { }
}

let activeTask: ImageEmbedderTask | null = null;

export async function setupImageEmbedder(container: HTMLElement) {
  activeTask = new ImageEmbedderTask({
    container,
    template,
    defaultModelName: 'mobilenet_v3_small',
    defaultModelUrl: 'https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite',
    workerConstructor: ImageEmbedderWorker,
    defaultDelegate: 'CPU'
  });

  await activeTask.initialize();
}

export function cleanupImageEmbedder() {
  if (activeTask) {
    activeTask.cleanup();
    activeTask = null;
  }
}
