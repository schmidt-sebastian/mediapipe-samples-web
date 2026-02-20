import { test, expect } from '@playwright/test';

import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Audio Classifier Task', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', msg => console.log(`[Browser Console]: ${msg.text()}`));
    page.on('pageerror', err => console.log(`[Browser Error]: ${err}`));
    page.on('requestfailed', req => console.log(`[Request Failed]: ${req.url()} - ${req.failure()?.errorText}`));

    await page.route('**/yamnet.tflite', route => {
      console.log('Route matched for yamnet.tflite');
      route.fulfill({ path: path.join(__dirname, 'assets', 'yamnet.tflite') });
    });
    await page.goto('/#/audio/audio_classifier');
    // Wait for connection
    await page.waitForLoadState('domcontentloaded');
  });


  test('should load audio classifier page', async ({ page }) => {
    await expect(page.locator('.task-container')).toBeVisible();
    await expect(page.locator('h2')).toHaveText('Audio Classifier');
  });

  test('should show model selection', async ({ page }) => {
    const modelSelect = page.locator('#model-select');
    await expect(modelSelect).toBeVisible();
    await expect(modelSelect).toHaveValue('yamnet');
  });

  test('should switch between microphone and file tabs', async ({ page }) => {
    const tabFile = page.locator('#tab-file');
    const viewFile = page.locator('#view-file');
    const viewMic = page.locator('#view-microphone');

    await expect(viewMic).toBeVisible();
    await expect(viewFile).not.toBeVisible();

    await tabFile.click();
    await expect(viewFile).toBeVisible();
    await expect(viewMic).not.toBeVisible();
  });

  test('should handle max results change', async ({ page }) => {
    const maxResultsInput = page.locator('#max-results');
    const maxResultsValue = page.locator('#max-results-value');

    await maxResultsInput.fill('3');
    // Trigger input event
    await maxResultsInput.evaluate(e => e.dispatchEvent(new Event('input')));

    await expect(maxResultsValue).toHaveText('3');
  });

  // We can't easily test microphone in CI without fake media stream, 
  // but Playwright usually launches with --use-fake-device-for-media-stream if configured.
  // Let's assume we can at least check if buttons are enabled after init.
  // But init takes time (downloading model).

  test('should initialize and enable record button', async ({ page }) => {
    // Setup timeout for model download and init
    test.setTimeout(60000);
    const statusMessage = page.locator('#status-message');

    // Wait for "Ready" or "Model loaded"
    await expect(statusMessage).toHaveText('Ready', { timeout: 30000 });

    const recordButton = page.locator('#recordButton');
    await expect(recordButton).toBeEnabled();
    await expect(recordButton).toHaveText(/Start Recording/);
  });
});
