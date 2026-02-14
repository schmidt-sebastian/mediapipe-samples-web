import { test, expect } from '@playwright/test';
import { downloadTestImage } from './utils';

test.describe('Object Detection Task', () => {
  let imagePath: string;

  test.beforeAll(async () => {
    // Download a test image (dog)
    imagePath = await downloadTestImage('dog.jpg', 'https://storage.googleapis.com/mediapipe-assets/dog_fluffy.jpg');
  });

  test.beforeEach(async ({ page }) => {
    page.on('console', msg => {
      console.log(`[BROWSER ${msg.type()}] ${msg.text()}`);
    });
    page.on('pageerror', exc => console.log(`[BROWSER UNCAUGHT ERROR] ${exc}`));
    page.on('worker', worker => {
      worker.on('console', msg => console.log(`[WORKER ${msg.type()}] ${msg.text()}`));
    });

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
    await expect(page.locator('#status-message')).toContainText('Done', { timeout: 10000 });

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
});
