import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
  // 1. MATCH REPO NAME: Use your actual repository name here
  base: '/mediapipe-samples-web/',

  plugins: [
    viteStaticCopy({
      targets: [
        {
          // 2. TARGET FOLDER: Copy the files INTO a 'wasm' folder
          src: 'node_modules/@mediapipe/tasks-vision/wasm/*',
          dest: 'wasm'
        },
        {
          src: 'node_modules/@mediapipe/tasks-audio/wasm/*',
          dest: 'wasm'
        },
        {
          src: 'node_modules/@mediapipe/tasks-text/wasm/*',
          dest: 'wasm'
        }
      ]
    })
  ],
  worker: {
    format: 'iife'
  },
  server: {
    port: 5174,
    strictPort: true,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    }
  },
  preview: {
    port: 5174,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp'
    }
  }
});
