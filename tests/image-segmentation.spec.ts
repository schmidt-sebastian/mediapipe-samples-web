import { test, expect } from '@playwright/test';
import { downloadTestImage } from './utils';

test.describe('Image Segmentation Task', () => {
  let imagePath: string;

  test.beforeAll(async () => {
    imagePath = await downloadTestImage('dog.jpg', 'https://storage.googleapis.com/mediapipe-assets/dog_fluffy.jpg');
  });

  test.beforeEach(async ({ page }) => {
    await page.goto('/#/vision/image_segmenter');
    page.on('console', msg => console.log(`BROWSER LOG: ${msg.text()}`));
    await page.waitForSelector('h2:has-text("Image Segmentation")');
    await page.waitForSelector('h2:has-text("Image Segmentation")');
    // Wait for the UI to be ready
    await page.waitForSelector('#tab-image');
    // Wait for model to load
    await expect(page.locator('#status-message')).toContainText('Model loaded', { timeout: 60000 });
  });

  test('should load with default settings', async ({ page }) => {
    await expect(page.locator('#model-select')).toHaveValue('deeplab_v3');
    await expect(page.locator('#delegate-select')).toHaveValue('GPU');
    await expect(page.locator('#output-type')).toHaveValue('CATEGORY_MASK');
  });

  test('should segment image on CPU', async ({ page }) => {
    await page.selectOption('#delegate-select', 'CPU');

    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('#tab-image');
    await page.click('.upload-dropzone');
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(imagePath);

    await expect(page.locator('#status-message')).toContainText('Done', { timeout: 15000 });
    await expect(page.locator('#inference-time')).not.toContainText('- ms');

    // Verify mask result via visual regression
    await expect(page.locator('#image-canvas')).toHaveScreenshot('golden-mask-cpu.png', { maxDiffPixelRatio: 0.05 });
  });

  test('should segment image on CPU explicitly', async ({ page }) => {
    await page.selectOption('#delegate-select', 'CPU');
    await expect(page.locator('#delegate-select')).toHaveValue('CPU');

    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('#tab-image');
    await page.click('.upload-dropzone');
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(imagePath);

    await expect(page.locator('#status-message')).toContainText('Done', { timeout: 15000 });
    await expect(page.locator('#inference-time')).not.toContainText('- ms');

    // Verify mask result
    // Verify mask result via visual regression (CPU)
    await expect(page.locator('#image-canvas')).toHaveScreenshot('golden-mask-cpu-explicit.png', { maxDiffPixelRatio: 0.05 });

    // Check availability of results element
    await expect(page.locator('#test-results')).toBeAttached();

    // Visual Comparison
    // This will generate a golden on first run
    await expect(page).toHaveScreenshot('segmentation-cpu-explicit.png', { maxDiffPixelRatio: 0.2, timeout: 10000 });
  });

  test('should handle component changes (Confidence Mask)', async ({ page }) => {
    await page.selectOption('#output-type', 'CONFIDENCE_MASKS');

    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('#tab-image');
    await page.click('.upload-dropzone');
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(imagePath);

    await expect(page.locator('#status-message')).toContainText('Done', { timeout: 15000 });

    // Verify mask result
    // Verify mask result via visual regression (Confidence Mask)
    await expect(page.locator('#image-canvas')).toHaveScreenshot('golden-mask-confidence.png', { maxDiffPixelRatio: 0.05 });
  });
});
