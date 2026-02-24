import { test, expect } from '@playwright/test';

test.describe('Gesture Recognizer Task', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    page.on('console', msg => console.log(`[Browser Console] ${msg.text()}`));
    page.on('pageerror', err => console.log(`[Browser Error] ${err.message}`));
    await page.click('a[href="#/vision/gesture_recognizer"]');
    await page.waitForSelector('.viewport.loading-model', { state: 'detached', timeout: 30000 });
  });

  test('should load model and handle image upload', async ({ page }) => {
    // Check initial state
    await expect(page.locator('#status-message')).toHaveText(/(Done)|(Ready)|(Model loaded)/, { timeout: 30000 });

    // Upload image
    const fileInput = page.locator('#image-upload');
    await fileInput.setInputFiles('public/hand_model.png'); // Use hand image for better detection

    // Wait for result
    await expect(page.locator('#status-message')).toHaveText(/Done in/, { timeout: 15000 });
    await expect(page.locator('#gesture-output')).not.toBeEmpty();

    // Check inference time display
    await expect(page.locator('#inference-time')).toContainText('Inference Time:');
  });

  test('should support webcam toggling', async ({ page }) => {
    await page.click('#tab-webcam');
    await page.waitForSelector('#webcamButton:not([disabled])');

    const webcamBtn = page.locator('#webcamButton');
    await expect(webcamBtn).toHaveText('Enable Webcam');

    await webcamBtn.click();
    await expect(webcamBtn).toHaveText('Disable Webcam');
    await expect(page.locator('#status-message')).toHaveText(/(Webcam running...)|(Done)|(Ready)/, { timeout: 15000 });

    await webcamBtn.click();
    await expect(webcamBtn).toHaveText('Enable Webcam');
  });

  test('should handle delegate switching', async ({ page }) => {
    await page.selectOption('#delegate-select', 'CPU');
    // Wait for re-initialization
    await expect(page.locator('#status-message')).toHaveText(/(Model loaded. Ready.)|(Ready)|(Done)/, { timeout: 60000 });

    // Switch back to GPU
    await page.selectOption('#delegate-select', 'GPU');
    await expect(page.locator('#status-message')).toHaveText(/(Model loaded. Ready.)|(Ready)|(Done)/, { timeout: 60000 });
  });
});
