import { test, expect } from '@playwright/test';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Language Detector Task', () => {
  test.beforeEach(async ({ page }) => {
    // Route model to local asset
    await page.route('**/language_detector.tflite', route => {
      const assetPath = path.join(__dirname, 'assets', 'language_detector.tflite');
      route.fulfill({ path: assetPath });
    });

    await page.goto('#/text/language_detector');
    await expect(page.locator('#status-message')).toHaveText(/(Ready)|(Model loaded)/, { timeout: 30000 });
  });

  test('should detect language of text', async ({ page }) => {
    const detectBtn = page.locator('#detect-btn');
    await expect(detectBtn).toBeEnabled();

    // Input text
    await page.fill('#text-input', 'El mundo es un pañuelo.');
    await detectBtn.click();

    // Wait for results
    const resultsContainer = page.locator('#detection-results');
    await expect(resultsContainer).not.toBeEmpty();

    // Check for Spanish (es)
    await expect(resultsContainer).toContainText('es');

    // Verify inference time
    const inferenceTime = page.locator('#inference-time');
    await expect(inferenceTime).toContainText('Inference Time:');
    await expect(inferenceTime).not.toContainText('- ms');
  });

  test('should handle sample buttons', async ({ page }) => {
    // Click French sample
    // Click French sample
    await page.click('button:has-text("French")');
    // Actually the button text is "French", data-text is the content.
    // Selector: button with text "French"
    const frenchBtn = page.locator('button:has-text("French")');
    await frenchBtn.click();

    // It should auto-detect? 
    // The code says: btn.addEventListener('click', ... detectLanguage(text));
    // So yes.

    const resultsContainer = page.locator('#detection-results');
    await expect(resultsContainer).toContainText('fr', { timeout: 5000 });
  });
});
