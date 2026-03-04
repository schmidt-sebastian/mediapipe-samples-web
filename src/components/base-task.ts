import { ModelSelector } from './model-selector';
import { ViewToggle } from './view-toggle';

export interface BaseTaskOptions {
  container: HTMLElement;
  template: string;
  defaultModelName: string;
  defaultModelUrl: string;
  workerConstructor: new () => Worker;
  defaultDelegate?: 'CPU' | 'GPU';
}

export abstract class BaseTask {
  protected container: HTMLElement;
  protected worker: Worker | undefined;

  protected currentModel: string;
  protected models: Record<string, string> = {};
  protected modelSelector!: ModelSelector;
  protected currentDelegate: 'CPU' | 'GPU' = 'GPU';

  protected runningMode: 'IMAGE' | 'VIDEO' = 'IMAGE';
  protected video!: HTMLVideoElement;
  protected canvasElement!: HTMLCanvasElement;
  protected canvasCtx!: CanvasRenderingContext2D;
  protected enableWebcamButton!: HTMLButtonElement;

  protected lastVideoTimeSeconds = -1;
  protected lastTimestampMs = -1;
  protected animationFrameId: number | undefined;
  protected isWorkerReady = false;

  constructor(protected options: BaseTaskOptions) {
    this.container = options.container;
    this.currentModel = options.defaultModelName;
    this.models[options.defaultModelName] = options.defaultModelUrl;

    const urlParams = new URLSearchParams(window.location.search);
    const delegateParam = urlParams.get('delegate');
    if (delegateParam === 'CPU' || delegateParam === 'GPU') {
      this.currentDelegate = delegateParam;
    } else if (options.defaultDelegate) {
      this.currentDelegate = options.defaultDelegate;
    }
  }

  public async initialize() {
    this.container.innerHTML = this.options.template;

    this.video = document.getElementById('webcam') as HTMLVideoElement;
    this.canvasElement = document.getElementById('output_canvas') as HTMLCanvasElement;
    if (this.canvasElement) {
      this.canvasCtx = this.canvasElement.getContext('2d')!;
    }
    this.enableWebcamButton = document.getElementById('webcamButton') as HTMLButtonElement;

    this.initWorker();
    this.setupUI();
    this.setupViewToggle();
    this.setupImageUpload();

    // Child class hook
    this.onInitializeUI();

    this.setupDelegateSelect();

    await this.initializeDetector();
  }

  protected initWorker() {
    if (!this.worker) {
      this.worker = new this.options.workerConstructor();
    }
    if (this.worker) {
      this.worker.onmessage = this.handleWorkerMessage.bind(this);
    }
  }

  protected handleWorkerMessage(event: MessageEvent) {
    const { type } = event.data;

    switch (type) {
      case 'LOAD_PROGRESS':
        this.handleLoadProgress(event.data);
        break;

      case 'INIT_DONE':
        this.handleInitDone();
        break;

      case 'DELEGATE_FALLBACK':
        console.warn('Worker fell back to CPU delegate.');
        this.currentDelegate = 'CPU';
        const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
        if (delegateSelect) delegateSelect.value = 'CPU';
        break;

      case 'DETECT_RESULT':
        const { mode, result, inferenceTime } = event.data;
        this.updateStatus(`Done in ${Math.round(inferenceTime)}ms`);
        this.updateInferenceTime(inferenceTime);

        if (mode === 'IMAGE') {
          this.displayImageResult(result);
        } else if (mode === 'VIDEO') {
          this.displayVideoResult(result);
          if (this.video.srcObject && !this.video.paused) {
            this.animationFrameId = window.requestAnimationFrame(this.predictWebcam.bind(this));
          }
        }
        break;

      case 'ERROR':
      case 'DETECT_ERROR':
        console.error('Worker error:', event.data.error);
        this.updateStatus(`Error: ${event.data.error}`);
        break;
    }
  }

  protected handleLoadProgress(data: any) {
    const { progress, loaded, total } = data;
    if (progress !== undefined) {
      this.modelSelector?.showProgress(progress * 100, 100);
      if (progress >= 1) setTimeout(() => this.modelSelector?.hideProgress(), 500);
    } else if (loaded !== undefined && total !== undefined) {
      this.modelSelector?.showProgress(loaded, total);
      if (loaded >= total) setTimeout(() => this.modelSelector?.hideProgress(), 500);
    }
  }

  protected handleInitDone() {
    this.modelSelector?.hideProgress();
    document.querySelector('.viewport')?.classList.remove('loading-model');
    this.isWorkerReady = true;

    if (this.video && this.video.srcObject && this.enableWebcamButton) {
      this.enableWebcamButton.innerText = 'Disable Webcam';
      this.enableWebcamButton.disabled = false;
    } else if (this.enableWebcamButton && this.enableWebcamButton.innerText !== 'Starting...') {
      this.enableWebcamButton.innerText = 'Enable Webcam';
      this.enableWebcamButton.disabled = false;
    }

    this.updateStatus('Ready');

    if (this.runningMode === 'VIDEO') {
      if (this.video.srcObject) {
        this.enableCam();
      }
    } else if (this.runningMode === 'IMAGE') {
      const testImage = document.getElementById('test-image') as HTMLImageElement;
      if (testImage && testImage.style.display !== 'none' && testImage.src) {
        this.triggerImageDetection(testImage);
      }
    }
  }

  protected setupViewToggle() {
    const viewWebcam = document.getElementById('view-webcam');
    const viewImage = document.getElementById('view-image');

    if (!viewWebcam || !viewImage) return;

    const switchView = (mode: 'VIDEO' | 'IMAGE') => {
      localStorage.setItem('mediapipe-running-mode', mode);
      if (mode === 'VIDEO') {
        viewWebcam.classList.add('active');
        viewImage.classList.remove('active');
        this.runningMode = 'VIDEO';
        this.worker?.postMessage({ type: 'SET_OPTIONS', runningMode: 'VIDEO' });

        const isWebcamActive = localStorage.getItem('mediapipe-webcam-active') === 'true';
        if (isWebcamActive) {
          this.enableCam();
        }
      } else {
        viewWebcam.classList.remove('active');
        viewImage.classList.add('active');
        this.runningMode = 'IMAGE';
        this.worker?.postMessage({ type: 'SET_OPTIONS', runningMode: 'IMAGE' });
        this.stopCam(true);

        if (this.isWorkerReady) {
          const testImage = document.getElementById('test-image') as HTMLImageElement;
          if (testImage && testImage.src) this.triggerImageDetection(testImage);
        }
      }
    };

    const storedMode = localStorage.getItem('mediapipe-running-mode') as 'VIDEO' | 'IMAGE';
    const initialMode = storedMode || 'IMAGE';

    new ViewToggle(
      'view-mode-toggle',
      [
        { label: 'Webcam', value: 'video' },
        { label: 'Image', value: 'image' }
      ],
      initialMode.toLowerCase(),
      (value) => {
        switchView(value === 'video' ? 'VIDEO' : 'IMAGE');
      }
    );

    switchView(initialMode);
    if (this.enableWebcamButton) {
      this.enableWebcamButton.addEventListener('click', this.toggleCam.bind(this));
    }
  }

  protected setupImageUpload() {
    const imageUpload = document.getElementById('image-upload') as HTMLInputElement;
    const imagePreviewContainer = document.getElementById('image-preview-container')!;
    const testImage = document.getElementById('test-image') as HTMLImageElement;
    const dropzone = document.querySelector('.upload-dropzone') as HTMLElement;
    const dropzoneContent = document.querySelector('.dropzone-content') as HTMLElement;

    if (testImage && testImage.src && dropzoneContent) {
      dropzoneContent.style.display = 'none';
    }

    if (dropzone) dropzone.addEventListener('click', () => imageUpload?.click());

    imageUpload?.addEventListener('change', (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          if (testImage) testImage.src = e.target?.result as string;
          if (imagePreviewContainer) imagePreviewContainer.style.display = '';
          const dc = document.querySelector('.dropzone-content') as HTMLElement;
          if (dc) dc.style.display = 'none';

          if (testImage) this.triggerImageDetection(testImage);
        };
        reader.readAsDataURL(file);
      }
    });
  }

  protected setupDelegateSelect() {
    const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
    if (delegateSelect) {
      delegateSelect.addEventListener('change', async () => {
        this.currentDelegate = delegateSelect.value as 'GPU' | 'CPU';
        await this.initializeDetector();
      });
      delegateSelect.value = this.currentDelegate;
    }
  }

  protected setupUI() {
    this.modelSelector = new ModelSelector(
      'model-selector-container',
      [
        { label: `${this.options.defaultModelName} (Default)`, value: this.options.defaultModelName, isDefault: true }
      ],
      async (selection) => {
        if (selection.type === 'standard') {
          this.currentModel = selection.value;
        } else if (selection.type === 'custom') {
          this.models['custom'] = URL.createObjectURL(selection.file);
          this.currentModel = 'custom';
        }
        if (this.enableWebcamButton) {
          this.enableWebcamButton.innerText = 'Loading...';
          this.enableWebcamButton.disabled = true;
        }
        await this.initializeDetector();
      }
    );
  }

  protected async initializeDetector() {
    if (this.enableWebcamButton) {
      this.enableWebcamButton.disabled = true;
      if (!this.video || !this.video.srcObject) {
        this.enableWebcamButton.innerText = 'Initializing...';
      }
    }
    document.querySelector('.viewport')?.classList.add('loading-model');
    this.isWorkerReady = false;
    this.updateStatus('Loading Model...');

    // @ts-ignore
    const baseUrl = import.meta.env.BASE_URL;
    let modelPath = this.models[this.currentModel];

    if (this.currentModel === 'custom' && this.models['custom']) {
      modelPath = this.models['custom'];
    } else if (!modelPath.startsWith('http')) {
      modelPath = new URL(modelPath, new URL(baseUrl, window.location.origin)).href;
    }

    const initParams = this.getWorkerInitParams();

    this.worker?.postMessage({
      type: 'INIT',
      modelAssetPath: modelPath,
      delegate: this.currentDelegate,
      runningMode: this.runningMode,
      baseUrl,
      ...initParams
    });
  }

  protected triggerImageDetection(image: HTMLImageElement) {
    if (image.complete && image.naturalWidth > 0) {
      this.detectImage(image);
    } else {
      image.onload = () => {
        if (image.naturalWidth > 0) {
          this.detectImage(image);
        }
      };
    }
  }

  protected async detectImage(image: HTMLImageElement) {
    if (!this.worker || !this.isWorkerReady) return;
    if (this.runningMode !== 'IMAGE') this.runningMode = 'IMAGE';

    const bitmap = await createImageBitmap(image);
    this.updateStatus(`Processing image...`);
    this.worker.postMessage({
      type: 'DETECT_IMAGE',
      bitmap: bitmap,
      timestampMs: performance.now()
    }, [bitmap]);
  }

  protected async enableCam() {
    if (!this.worker || !this.video) return;
    if (this.video.srcObject) return;

    if (this.enableWebcamButton) {
      this.enableWebcamButton.innerText = 'Starting...';
      this.enableWebcamButton.disabled = true;
    }
    const constraints = { video: true };

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.video.srcObject = stream;
      document.getElementById('webcam-placeholder')?.classList.add('hidden');

      const playAndPredict = () => {
        if (!this.video) return;
        this.video.play().catch(console.error);
        this.predictWebcam();
      };

      if (this.video.readyState >= 2) {
        playAndPredict();
      } else {
        this.video.addEventListener('loadeddata', playAndPredict, { once: true });
      }

      this.runningMode = 'VIDEO';
      localStorage.setItem('mediapipe-webcam-active', 'true');
      this.worker.postMessage({ type: 'SET_OPTIONS', runningMode: 'VIDEO' });
      this.updateStatus('Webcam running...');
      if (this.enableWebcamButton) {
        this.enableWebcamButton.innerText = 'Disable Webcam';
        this.enableWebcamButton.disabled = false;
      }
    } catch (err) {
      console.error(err);
      this.updateStatus('Camera error!');
      if (this.enableWebcamButton) {
        this.enableWebcamButton.innerText = 'Enable Webcam';
        this.enableWebcamButton.disabled = false;
      }
    }
  }

  protected toggleCam() {
    if (this.video && this.video.srcObject) {
      this.stopCam(true);
    } else {
      this.enableCam();
    }
  }

  protected stopCam(persistState = true) {
    if (this.video.srcObject) {
      const stream = this.video.srcObject as MediaStream;
      const tracks = stream.getTracks();
      tracks.forEach((track) => track.stop());
      this.video.srcObject = null;
      document.getElementById('webcam-placeholder')?.classList.remove('hidden');
      if (this.enableWebcamButton) this.enableWebcamButton.innerText = 'Enable Webcam';
      if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);

      if (persistState) {
        localStorage.setItem('mediapipe-webcam-active', 'false');
      }
    }
  }

  protected async predictWebcam() {
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

        this.worker?.postMessage({
          type: 'DETECT_VIDEO',
          bitmap: bitmap,
          timestampMs: timestampMs
        }, [bitmap]);
      } catch (e) {
        console.error("Failed to create ImageBitmap from video", e);
        this.animationFrameId = window.requestAnimationFrame(this.predictWebcam.bind(this));
      }
    } else {
      this.animationFrameId = window.requestAnimationFrame(this.predictWebcam.bind(this));
    }
  }

  protected updateStatus(msg: string) {
    const el = document.getElementById('status-message');
    if (el) el.innerText = msg;
  }

  protected updateInferenceTime(time: number) {
    const el = document.getElementById('inference-time');
    if (el) el.innerText = `Inference Time: ${time.toFixed(2)} ms`;
  }

  public cleanup() {
    if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);
    this.stopCam(false);

    if (this.worker) {
      this.worker.postMessage({ type: 'CLEANUP' });
      this.worker.terminate();
      this.worker = undefined;
    }

    this.isWorkerReady = false;

    if (this.canvasCtx && this.canvasElement) {
      this.canvasCtx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
    }
  }

  // Custom hooks for child tasks
  protected onInitializeUI(): void { }
  protected abstract getWorkerInitParams(): Record<string, any>;
  protected abstract displayImageResult(result: any): void;
  protected abstract displayVideoResult(result: any): void;
}
