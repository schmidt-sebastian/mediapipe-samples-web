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

  test('should hide classification results label by default', async ({ page }) => {
    // The results container should be hidden initially
    const resultsContainer = page.locator('.results-container');
    await expect(resultsContainer).toBeHidden();

    // Check via computed style just to be sure
    const display = await resultsContainer.evaluate(el => window.getComputedStyle(el).display);
    expect(display).toBe('none');
  });

  test('should layout file upload view correctly', async ({ page }) => {
    await page.click('#tab-file');

    // Simulate file upload state (since we can't easily upload in this env without a real file)
    // We'll just check the structure that SHOULD be there
    const previewContainer = page.locator('#audio-preview-container');

    // Force show it for testing layout computations
    await page.evaluate(() => {
      const pc = document.getElementById('audio-preview-container');
      if (pc) {
        pc.style.display = 'flex';
        pc.style.flexDirection = 'column';
      }
    });

    await expect(previewContainer).toHaveCSS('flex-direction', 'column');
    await expect(previewContainer).toHaveCSS('align-items', 'center');

    const classifyBtn = page.locator('#run-file-classification');
    const player = page.locator('#audio-player');

    // Check order: Player first, then button
    const playerBox = await player.boundingBox();
    const btnBox = await classifyBtn.boundingBox();

    if (playerBox && btnBox) {
      expect(btnBox.y).toBeGreaterThan(playerBox.y); // Button below player
    }
  });

  test('should adapt layout for mobile', async ({ page }) => {
    // Resize to mobile
    await page.setViewportSize({ width: 375, height: 667 });
    await page.reload(); // Reload to force layout recalc if needed
    await page.waitForLoadState('domcontentloaded');

    // Check controls panel width
    const controlsPanel = page.locator('.controls-panel');
    await expect(controlsPanel).toBeVisible();

    // Allow small margin of error for scrollbars etc
    const box = await controlsPanel.boundingBox();
    if (box) {
      expect(box.width).toBeGreaterThan(340);
      expect(box.width).toBeLessThan(376);
    }

    // Check header text (Mobile logic)
    const mobileHeader = page.locator('.mobile-header');
    await expect(mobileHeader).toBeVisible();

    // Check if the select has correct value
    const select = page.locator('#mobile-task-select');
    await expect(select).toHaveValue('#/audio/audio_classifier');
  });
});
