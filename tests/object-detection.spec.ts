import { test, expect } from '@playwright/test';
import { downloadTestImage } from './utils';

test.describe('Object Detection Task', () => {
  let imagePath: string;

  test.beforeAll(async () => {
    // Download a test image (dog)
    imagePath = await downloadTestImage('dog.jpg', 'https://storage.googleapis.com/mediapipe-assets/dog_fluffy.jpg');
  });

  test.beforeEach(async ({ page }) => {
    await page.goto('/#/vision/object_detector');
    await page.waitForSelector('h2:has-text("Object Detection")');
    await expect(page.locator('#status-message')).toHaveText(/(Model loaded\. Ready\.)|(Running detection\.\.\.)|(Done)/, { timeout: 30000 });
  });

  test('should load with default settings', async ({ page }) => {
    await expect(page.locator('#model-select')).toHaveValue('efficientdet_lite0');
    await expect(page.locator('#delegate-select')).toHaveValue('GPU');
  });

  test('should detect objects on CPU', async ({ page }) => {
    // Switch to CPU
    await page.selectOption('#delegate-select', 'CPU');

    // Upload Image
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('#tab-image'); // Switch to Image tab
    await page.click('#image-upload'); // Trigger upload (might need to click label or input)
    // Actually the input is hidden, usually we get file chooser by clicking the dropzone or label.
    // In our HTML: <input type="file" id="image-upload" ...> which is inside .upload-dropzone (or related).
    // Let's force click the dropzone or the input (if label wraps/linked).
    // The previous code had a dropzone.
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(imagePath);

    // Wait for processing
    await expect(page.locator('#status-message')).toContainText('Done', { timeout: 10000 });

    // Check results
    // We expect some text or canvas drawing. 
    // The current implementation draws on canvas. 
    // We can verify inference time is updated.
    // Check results
    await expect(page.locator('#inference-time')).toContainText('Inference Time:');
    await expect(page.locator('#inference-time')).not.toContainText('- ms');

    // Verify detection logic
    const resultsText = await page.locator('#test-results').textContent();
    const detections = JSON.parse(resultsText || '[]');
    expect(detections.length).toBeGreaterThan(0);
    const firstCat = detections[0].categories[0];
    // dog_fluffy.jpg should be a dog
    expect(firstCat.categoryName.toLowerCase()).toContain('dog');
  });

  test('should detect objects on GPU', async ({ page }) => {
    // GPU is default
    await expect(page.locator('#delegate-select')).toHaveValue('GPU');

    // Upload Image
    const fileChooserPromise = page.waitForEvent('filechooser');
    await page.click('#tab-image');
    await page.click('.upload-dropzone'); // Click dropzone to trigger
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(imagePath);

    // Wait for processing
    await expect(page.locator('#status-message')).toContainText('Done', { timeout: 10000 });

    // Check results
    await expect(page.locator('#inference-time')).toContainText('Inference Time:');
    await expect(page.locator('#inference-time')).not.toContainText('- ms');

    // Verify detection properties
    const resultsText = await page.locator('#test-results').textContent();
    const detections = JSON.parse(resultsText || '[]');
    expect(detections.length).toBeGreaterThan(0);
    expect(detections[0].categories[0].categoryName.toLowerCase()).toContain('dog');
  });
});
