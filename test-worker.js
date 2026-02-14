const { Worker } = require('worker_threads');
const fs = require('fs');

console.log("Starting Worker Test");
// Running the bundled worker from `dist` to see if it throws immediately.
const worker = new Worker('./dist/js/object-detection.worker.js');
worker.on('message', (msg) => console.log(msg));
worker.on('error', (err) => console.log("WORKER ERROR:", err));
worker.postMessage({
  type: 'INIT',
  modelAssetPath: '',
  delegate: 'CPU',
  baseUrl: ''
});
