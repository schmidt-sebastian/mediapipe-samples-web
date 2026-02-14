const { FilesetResolver, ObjectDetector } = require('@mediapipe/tasks-vision');

async function run() {
  try {
    const vision = await FilesetResolver.forVisionTasks('./node_modules/@mediapipe/tasks-vision/wasm');
    console.log("WASM resolved");
    const detector = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite',
        delegate: 'CPU'
      },
      scoreThreshold: 0.5,
      runningMode: 'IMAGE'
    });
    console.log("Detector created");
  } catch (err) {
    console.error("TEST FAILED:", err);
  }
}
run();
