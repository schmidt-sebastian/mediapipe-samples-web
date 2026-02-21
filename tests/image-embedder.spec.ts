import { test, expect } from '@playwright/test';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Image Embedder Task', () => {
  let imagePath: string;

  test.beforeAll(() => {
    imagePath = path.resolve(__dirname, '..', 'public', 'dog.jpg');
  });

  test.beforeEach(async ({ page }) => {
    // Route model
    await page.route('**/mobilenet_v3_small.tflite', route => {
      const assetPath = path.join(__dirname, 'assets', 'mobilenet_v3_small.tflite');
      route.fulfill({ path: assetPath });
    });

    await page.goto('#/vision/image_embedder');
    await expect(page.locator('#status-message')).toHaveText(/(Ready)|(Model loaded)/, { timeout: 60000 });
  });

  test('should compute similarity between two images', async ({ page }) => {
    // Choose image 1
    // Upload same image to both slots
    const fileChooserPromise1 = page.waitForEvent('filechooser');
    await page.click('#display-area-1');
    const fileChooser1 = await fileChooserPromise1;
    await fileChooser1.setFiles(imagePath);

    const fileChooserPromise2 = page.waitForEvent('filechooser');
    await page.click('#display-area-2');
    const fileChooser2 = await fileChooserPromise2;
    await fileChooser2.setFiles(imagePath);

    // Check result (should appear automatically)
    const valueEl = page.locator('#similarity-value');
    await expect(valueEl).toBeVisible({ timeout: 30000 });
    
    // Similarity of same image should be close to 1.0
    // Wait for text to not be '--'
    await expect(valueEl).not.toHaveText('--', { timeout: 10000 });
    
    const valueText = await valueEl.innerText();
    const value = parseFloat(valueText);
    expect(value).toBeGreaterThan(0.9);
  });
});
