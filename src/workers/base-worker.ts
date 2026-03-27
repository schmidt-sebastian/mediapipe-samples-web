
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

export abstract class BaseWorker<T> {
  protected taskInstance: T | undefined;
  protected isInitializing = false;
  protected currentOptions: any = {};
  protected basePath = '/';
  protected isProcessing = false;

  constructor() {
    self.onmessage = this.handleMessage.bind(this);
  }

  protected async handleMessage(event: MessageEvent) {
    const { type } = event.data;

    while (this.isProcessing) {
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    this.isProcessing = true;

    try {
      if (type === 'INIT') {
        const { modelAssetPath, delegate, baseUrl, ...rest } = event.data;
        this.basePath = baseUrl || '/';
        this.currentOptions = { modelAssetPath, delegate, ...rest };

        await this.initializeBase(event.data);

        const payload = this.getInitPayload();
        self.postMessage({ type: 'INIT_DONE', ...payload });

      } else if (type === 'SET_OPTIONS') {
        const { type: _type, ...optionsToUpdate } = event.data;
        Object.assign(this.currentOptions, optionsToUpdate);
        await this.updateOptions();
        self.postMessage({ type: 'OPTIONS_UPDATED' });

      } else if (type === 'CLEANUP') {
        if (this.taskInstance) {
          (this.taskInstance as any).close?.();
          this.taskInstance = undefined;
        }
        self.postMessage({ type: 'CLEANUP_DONE' });
      } else {
        await this.handleCustomMessage(event.data);
      }
    } catch (error: any) {
      console.error("Worker Error:", error);
      self.postMessage({ type: 'ERROR', error: error?.message || String(error) });
    } finally {
      this.isProcessing = false;
    }
  }

  private async initializeBase(data: any) {
    if (this.isInitializing) return;
    this.isInitializing = true;

    try {
      if (this.taskInstance) {
        (this.taskInstance as any).close?.();
        this.taskInstance = undefined;
      }
      await this.initializeTask(data);
    } finally {
      this.isInitializing = false;
    }
  }

  protected async loadModelAsset(): Promise<ArrayBuffer> {
    const response = await fetch(this.currentOptions.modelAssetPath);
    if (!response.ok) {
      throw new Error(`Failed to load model: ${response.statusText}`);
    }

    const contentLength = response.headers.get('content-length');
    const total = contentLength ? parseInt(contentLength, 10) : 0;

    const reader = response.body?.getReader();
    if (!reader) {
      return response.arrayBuffer();
    }

    let receivedLength = 0;
    const chunks: Uint8Array[] = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      chunks.push(value);
      receivedLength += value.length;

      if (total > 0) {
        self.postMessage({ type: 'LOAD_PROGRESS', loaded: receivedLength, total });
      }
    }

    const chunksAll = new Uint8Array(receivedLength);
    let position = 0;
    for (const chunk of chunks) {
      chunksAll.set(chunk, position);
      position += chunk.length;
    }

    return chunksAll.buffer;
  }

  /**
   * Return the constructed wasm path based on the current basePath
   */
  protected getWasmPath(): string {
    const formattedBasePath = this.basePath.endsWith('/') ? this.basePath : `${this.basePath}/`;
    return new URL(`${formattedBasePath}wasm`, self.location.origin).href.replace(/\/$/, '');
  }

  protected async loadWasmModule(wasmLoaderUrl: string, appendScript?: string): Promise<any> {
    // @ts-ignore
    if (typeof self.import === 'undefined') {
      // @ts-ignore
      self.import = (url) => import(/* @vite-ignore */ url);
    }
    try {
      const response = await fetch(wasmLoaderUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch WASM loader: ${response.status} ${response.statusText}`);
      }
      const loaderCode = await response.text();
      const codeToEval = appendScript ? loaderCode + appendScript : loaderCode;
      const result = (0, eval)(codeToEval);
      console.log(`MediaPipe WASM loader injected successfully from ${wasmLoaderUrl}`);
      return result;
    } catch (e) {
      console.error('Failed to inject MediaPipe WASM loader:', e);
      throw e;
    }
  }

  protected abstract initializeTask(data?: any): Promise<void>;
  protected abstract updateOptions(data?: any): Promise<void>;
  protected abstract handleCustomMessage(data: any): Promise<void>;

  protected getInitPayload(): any {
    return {};
  }
}
