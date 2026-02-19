import { test, expect } from '@playwright/test';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Object Detection Task', () => {
  let imagePath: string;

  test.beforeAll(() => {
    imagePath = path.resolve(__dirname, '..', 'public', 'dog.jpg');
  });

  test.beforeEach(async ({ page }) => {
    page.on('console', msg => {
      // Filter out overly noisy WebGL verbose logs from Chromium if desired, but keep for now
      console.log(`[BROWSER ${msg.type()}] ${msg.text()}`);
    });
    page.on('pageerror', exc => console.log(`[BROWSER UNCAUGHT ERROR] ${exc}`));

    // Clear local storage to ensure fresh start
    await page.addInitScript(() => window.localStorage.clear());
    // Force CPU by default to ensure reliability in standard tests
    await page.goto('#/vision/object_detector');
    await page.waitForSelector('h2:has-text("Object Detection")');
    // Wait for model to load
    await expect(page.locator('#status-message')).toHaveText(/(Model loaded\. Ready\.)|(Running detection\.\.\.)|(Done)|(Ready)/, { timeout: 60000 });
  });

  test('should load with default settings', async ({ page }) => {
    await expect(page.locator('#model-select')).toHaveValue('efficientdet_lite0');
    // Expect CPU because we forced it in beforeEach
    await expect(page.locator('#delegate-select')).toHaveValue('CPU');
  });

  test('should detect objects on CPU', async ({ page }) => {
    // CPU is default via URL, check it
    await expect(page.locator('#delegate-select')).toHaveValue('CPU');

    // Upload Image
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('#tab-image'); // Switch to Image tab
    await page.click('.upload-dropzone'); // Click dropzone to trigger
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(imagePath);

    // Wait for processing
    await expect(page.locator('#status-message')).toHaveText(/(Done)|(Ready)|(Model loaded)/, { timeout: 15000 });

    // Check results
    await expect(page.locator('#inference-time')).toContainText('Inference Time:');
    await expect(page.locator('#inference-time')).not.toContainText('- ms');

    // Verify detection logic
    const resultsText = await page.locator('#test-results').textContent();
    const detections = JSON.parse(resultsText || '[]');
    expect(detections.length).toBeGreaterThan(0);
    const firstCat = detections[0].categories[0];
    expect(firstCat.categoryName.toLowerCase()).toContain('dog');
  });

  test('should handle model switching', async ({ page }) => {
    await page.selectOption('#model-select', 'efficientdet_lite2');
    await expect(page.locator('#status-message')).toHaveText(/(Model loaded\. Ready\.)|(Ready)/, { timeout: 60000 });

    // Wait for worker re-instantiation to fully commit DOM layout reflow
    await page.waitForTimeout(1000);

    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('#tab-image');
    await page.click('.upload-dropzone', { force: true });
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(imagePath);

    await expect(page.locator('#status-message')).toHaveText(/(Done)|(Ready)|(Model loaded)/, { timeout: 15000 });
    const resultsText = await page.locator('#test-results').textContent();
    expect(JSON.parse(resultsText || '[]').length).toBeGreaterThan(0);
  });

  test('should handle max results & threshold changes', async ({ page }) => {
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('#tab-image');
    await page.click('.upload-dropzone');
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(imagePath);

    await expect(page.locator('#status-message')).toHaveText(/(Done)|(Ready)|(Model loaded)/, { timeout: 15000 });

    // Lower threshold and increase max results
    await page.fill('#score-threshold', '0.1');
    await page.fill('#max-results', '5');
    // Dispatch input to trigger event listener
    await page.locator('#score-threshold').dispatchEvent('input');
    await page.locator('#max-results').dispatchEvent('input');

    await expect(page.locator('#score-threshold-value')).toHaveText('0.1');
    await expect(page.locator('#max-results-value')).toHaveText('5');

    await expect(page.locator('#status-message')).toHaveText(/(Done)|(Ready)|(Model loaded)/, { timeout: 15000 });
  });

  test('should handle delegate switching to GPU', async ({ page }) => {
    // Note: CPU is forced in beforeAll, so changing here triggers initialization.
    await page.selectOption('#delegate-select', 'GPU');

    // Note: If running on a system without a GPU, it might fallback to CPU and warn, but it should still become Ready.
    await expect(page.locator('#status-message')).toHaveText(/(Model loaded\. Ready\.)|(Ready)|(Done)/, { timeout: 60000 });
  });

  test('should support webcam toggling', async ({ page }) => {
    await page.click('#tab-webcam');
    // Ensure button is clickable before interacting
    await page.waitForSelector('#webcamButton:not([disabled])');

    // Wait for App to mount constraints via getUserMedia() - button turns to 'Disable Webcam'
    await expect(page.locator('#webcamButton')).not.toHaveText('Initializing...', { timeout: 15000 });
    await expect(page.locator('#status-message')).toHaveText(/(Webcam running\.\.\.)|(Done)|(Ready)/, { timeout: 15000 });

    // Disable
    await page.click('#webcamButton', { force: true });
    await expect(page.locator('#webcamButton')).toHaveText('Enable Webcam', { timeout: 10000 });
  });

  test('should handle custom model uploads', async ({ page }) => {
    const modelPath = path.resolve(__dirname, 'assets', 'efficientdet_lite0.tflite');

    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('#tab-model-upload');
    await page.click('.file-upload-btn');
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(modelPath);

    await expect(page.locator('#upload-status')).toHaveText('efficientdet_lite0.tflite');
    await expect(page.locator('#status-message')).toHaveText(/(Model loaded\. Ready\.)|(Ready)|(Done)/, { timeout: 60000 });

    // Verify it still detects
    const imgChooserPromise = page.waitForEvent('filechooser');
    await page.click('#tab-image');
    await page.click('.upload-dropzone');
    const imgChooser = await imgChooserPromise;
    await imgChooser.setFiles(imagePath);

    await expect(page.locator('#status-message')).toContainText('Done', { timeout: 15000 });
    const resultsText = await page.locator('#test-results').textContent();
    expect(JSON.parse(resultsText || '[]').length).toBeGreaterThan(0);
  });
});
