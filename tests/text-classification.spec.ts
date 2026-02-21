import { test, expect } from '@playwright/test';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Text Classification Task', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', msg => console.log(`[BROWSER ${msg.type()}] ${msg.text()}`));
    page.on('pageerror', exc => console.log(`[BROWSER UNCAUGHT ERROR] ${exc}`));

    // Route model to local asset if available (downloaded by global-setup)
    await page.route('**/bert_classifier.tflite', route => {
      const assetPath = path.join(__dirname, 'assets', 'bert_classifier.tflite');
      console.log(`Intercepting BERT model request to: ${assetPath}`);
      route.fulfill({ path: assetPath });
    });

    await page.goto('#/text/text_classifier');
  });

  test('should load model and classify text', async ({ page }) => {
    // Wait for button to NOT say "Loading..." or "Error"
    // Ideally it becomes "Classify"
    const classifyBtn = page.locator('#classify-btn');
    await expect(classifyBtn).toHaveText('Classify', { timeout: 30000 });
    await expect(classifyBtn).toBeEnabled();

    // Input text
    await page.fill('#text-input', 'I love this product, it is amazing!');
    await classifyBtn.click();

    // Wait for results
    const resultsContainer = page.locator('#classification-results');
    await expect(resultsContainer).not.toBeEmpty();

    // Check for positive sentiment
    await expect(resultsContainer).toContainText('positive');

    // Verify inference time is displayed
    const inferenceTime = page.locator('#inference-time');
    await expect(inferenceTime).toContainText('ms');
    await expect(inferenceTime).not.toContainText('- ms');
  });
});
