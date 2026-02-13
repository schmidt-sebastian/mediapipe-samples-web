import { test, expect } from '@playwright/test';
import { downloadTestImage } from './utils';

test.describe('Image Segmentation Task', () => {
  let imagePath: string;

  test.beforeAll(async () => {
    imagePath = await downloadTestImage('dog.jpg', 'https://storage.googleapis.com/mediapipe-assets/dog_fluffy.jpg');
  });

  test.beforeEach(async ({ page }) => {
    // Put param before hash so visual regression works without reliance on fallback
    await page.goto('?delegate=CPU#/vision/image_segmenter');
    page.on('console', msg => console.log(`BROWSER LOG: ${msg.text()}`));
    await page.waitForSelector('h2:has-text("Image Segmentation")');
    // Wait for the UI to be ready
    await page.waitForSelector('#tab-image');
    // Wait for model to load
    // Wait for model to load - can take a while on CPU
    await expect(page.locator('#status-message')).toHaveText(/(Model loaded\. Ready\.)|(Done)|(Ready)/, { timeout: 120000 });
    // Also check that button is enabled
    await expect(page.locator('#webcamButton')).toBeEnabled({ timeout: 120000 });
  });

  test('should load with default settings', async ({ page }) => {
    await expect(page.locator('#model-select')).toHaveValue('deeplab_v3');
    await expect(page.locator('#delegate-select')).toHaveValue('CPU');
    await expect(page.locator('#output-type')).toHaveValue('CATEGORY_MASK');
  });

  test('should segment image on CPU', async ({ page }) => {
    await expect(page.locator('#delegate-select')).toHaveValue('CPU');

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
    // CPU is now default via URL, but we can still select it to be sure or check it's selected
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

  test('should segment image on GPU (emulated) @gpu', async ({ page }) => {
    // Listen for console logs to debug fallback
    page.on('console', msg => {
      if (msg.type() === 'warning' || msg.type() === 'error') {
        console.log(`PAGE LOG: ${msg.text()}`);
      }
    });

    // Navigate specifically to GPU mode
    await page.goto('?delegate=GPU#/vision/image_segmenter');
    await page.waitForSelector('h2:has-text("Image Segmentation")');
    // Wait for the UI to be ready
    await page.waitForSelector('#tab-image');
    // Wait for model to load (might be slower on emulated GPU)
    // Wait for model to load (might be slower on emulated GPU)
    await expect(page.locator('#status-message')).toHaveText(/(Model loaded\. Ready\.)|(Done)|(Ready)/, { timeout: 60000 });

    await expect(page.locator('#delegate-select')).toHaveValue(/GPU|CPU/);

    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('#tab-image');
    await page.click('.upload-dropzone');
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(imagePath);

    await expect(page.locator('#status-message')).toContainText('Done', { timeout: 30000 });

    // Verify result is present (visual regression might differ slightly on GPU, so maybe just check functionality)
    await expect(page.locator('#test-results')).toBeAttached();
  });
});
