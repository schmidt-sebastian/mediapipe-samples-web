/**
 * Polyfills and loaders for MediaPipe WASM compatibility in Vite workers.
 */

// 1. Polyfill self.import for Emscripten
// MediaPipe's Emscripten-generated WASM loader explicitly calls self.import() 
// in worker environments. We must polyfill it for Vite.
// @ts-ignore
if (typeof self.import === 'undefined') {
  // @ts-ignore
  self.import = (url) => import(/* @vite-ignore */ url);
}

/**
 * Loads the MediaPipe WASM loader script (legacy Emscripten) into the global scope.
 * This workaround is necessary because Vite dev workers are ES modules,
 * but MediaPipe's loader expects to be in a classic worker environment with importScripts support.
 * 
 * @param wasmLoaderUrl URL to the _wasm_internal.js file
 * @param appendScript Optional script modification to append before eval (e.g. ";ModuleFactory;")
 * @returns The result of the evaluated script
 */
export async function loadWasmModule(wasmLoaderUrl: string, appendScript?: string): Promise<any> {
  try {
    const response = await fetch(wasmLoaderUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch WASM loader: ${response.status} ${response.statusText}`);
    }
    const loaderCode = await response.text();
    // Execute in global scope so it attaches its ModuleFactory to self
    const codeToEval = appendScript ? loaderCode + appendScript : loaderCode;
    const result = (0, eval)(codeToEval);
    console.log(`MediaPipe WASM loader injected successfully from ${wasmLoaderUrl}`);
    return result;
  } catch (e) {
    console.error('Failed to inject MediaPipe WASM loader:', e);
    throw e;
  }
}
