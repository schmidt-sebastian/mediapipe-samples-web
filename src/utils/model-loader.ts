/**
 * Loads a model from a URL with progress reporting.
 * @param path The URL to the model file.
 * @param onProgress Callback function for progress updates (loaded, total).
 * @returns A Promise resolving to the model's ArrayBuffer.
 */
export async function loadModel(
  path: string, 
  onProgress?: (loaded: number, total: number) => void
): Promise<ArrayBuffer> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load model from ${path}: ${response.status} ${response.statusText}`);
  }

  const contentLength = response.headers.get('Content-Length');
  const total = contentLength ? parseInt(contentLength, 10) : 0;
  
  const reader = response.body?.getReader();
  if (!reader) {
    return await response.arrayBuffer();
  }

  let receivedLength = 0;
  const chunks: Uint8Array[] = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    chunks.push(value);
    receivedLength += value.length;

    if (onProgress && total > 0) {
      onProgress(receivedLength, total);
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
