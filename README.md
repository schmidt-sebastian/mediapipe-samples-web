# MediaPipe Web Task Demo

Experience the power of on-device Machine Learning with MediaPipe Tasks Vision. This demo showcases real-time Object Detection and Image Segmentation running entirely in your browser using WebGL2 acceleration.

![Screenshot](file:///Users/mrschmidt/.gemini/jetski/brain/6dc85e9d-ef5a-4915-be6a-69b171d8c4bc/image_segmentation_webcam_working_1770841181049.png)

## Features

### 🚀 High Performance
- **GPU Acceleration**: Utilizes WebGL2 for smooth, real-time performance.
- **Privacy First**: All processing happens locally on your device. No images are sent to the cloud.

### 👁️ Object Detection
- **State-of-the-Art Models**: Choose from EfficientDet-Lite0, Lite2, and SSD MobileNetV2.
- **Custom Models**: Upload your own TFLite models for specialized detection tasks.
- **Real-time Feedback**: View confidence scores and bounding boxes instantly.
- **Instant Demo**: Pre-loaded with a sample image (`dog_fluffy.jpg`) so you can see it in action immediately.

### 🎨 Image Segmentation
- **Precise Masks**: Generate pixel-perfect category masks for objects.
- **Confidence Visualization**: Inspect confidence heatmaps to understand model certainty.
- **Flexible Inputs**: Works seamlessly with both Webcam feed and static images.
- **Instant Demo**: Pre-loaded with a sample image (`dog_fluffy.jpg`) for immediate testing.

### 📱 Responsive Design
- **Mobile Friendly**: Fully responsive interface that adapts to your device.
- **Deep Linking**: Share specific tasks easily with hash-based routing (e.g., `#/vision/object_detector`).

## Quick Start

1.  **Clone the repository**
2.  **Install dependencies**: `npm install`
3.  **Run the app**: `npm run dev`
4.  **Open browser**: Navigate to the local URL (typically `http://localhost:5173`)

## Testing

We ensure robustness with a comprehensive End-to-End (E2E) test suite using Playwright.

```bash
npx playwright test
```

*Tests verify navigation, inference accuracy using standard test images, and UI responsiveness.*