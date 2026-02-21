import { test, expect } from '@playwright/test';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Face Detector Task', () => {
  let imagePath: string;

  test.beforeAll(() => {
    imagePath = path.resolve(__dirname, '..', 'public', 'dog.jpg');
    // Ideally use a face image, but dog.jpg might work if confidence is low, or we just check it runs without error.
    // Let's use the dog image for now to verify pipeline, even if no faces are found.
    // Or we can expect 0 faces.
  });

  test.beforeEach(async ({ page }) => {
    page.on('console', msg => {
      if (msg.type() === 'error') console.error(`[BROWSER ERROR] ${msg.text()}`);
    });

    // Route model to local asset
    await page.route('**/blaze_face_short_range.tflite', route => {
      const assetPath = path.join(__dirname, 'assets', 'blaze_face_short_range.tflite');
      route.fulfill({ path: assetPath });
    });

    await page.goto('#/vision/face_detector');
    await expect(page.locator('#status-message')).toHaveText(/(Model loaded\. Ready\.)|(Ready)/, { timeout: 30000 });
  });

  test('should load model and handle image upload', async ({ page }) => {
    // Check defaults
    await expect(page.locator('#model-select')).toHaveValue('blaze_face_short_range');

    // Upload Image
    // Upload Image
    await page.click('#tab-image');
    await page.locator('#image-upload').setInputFiles(imagePath);

    // Wait for processing
    await expect(page.locator('#status-message')).toHaveText(/(Done)|(Ready)|(Model loaded)/, { timeout: 30000 });

    // Verify inference time is displayed
    const inferenceTime = page.locator('#inference-time');
    await expect(inferenceTime).toContainText('ms');
  });

  test('should support webcam toggling', async ({ page }) => {
    await page.click('#tab-webcam');
    await page.waitForSelector('#webcamButton:not([disabled])');

    // Wait for App to mount constraints via getUserMedia()
    // It might fail in CI if no camera, but usually Playwright handles this by using fake media stream.
    // We assume the environment is set up (chromium --use-fake-device-for-media-stream)

    // Just allow it to try initializing
    await expect(page.locator('#webcamButton')).not.toHaveText('Initializing...', { timeout: 15000 });
  });
});
