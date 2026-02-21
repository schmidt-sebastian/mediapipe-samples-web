import { test, expect } from '@playwright/test';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Text Embedding Task', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', msg => console.log(`[BROWSER ${msg.type()}] ${msg.text()}`));
    page.on('pageerror', exc => console.log(`[BROWSER UNCAUGHT ERROR] ${exc}`));

    await page.route('**/universal_sentence_encoder.tflite', route => {
      const assetPath = path.join(__dirname, 'assets', 'universal_sentence_encoder.tflite');
      console.log(`Intercepting USE model request to: ${assetPath}`);
      route.fulfill({ path: assetPath });
    });

    await page.goto('#/text/text_embedder');
  });

  test('should load model and compute similarity', async ({ page }) => {
    const embedBtn = page.locator('#embed-btn');
    await expect(embedBtn).toHaveText('Compute Similarity', { timeout: 30000 });
    await expect(embedBtn).toBeEnabled();

    // Use sample chips
    await page.click('.sample-btn:has-text("Positive Pair")');

    // Verify text areas populated
    await expect(page.locator('#text-input-1')).not.toBeEmpty();
    await expect(page.locator('#text-input-2')).not.toBeEmpty();

    // Compute
    // The chip click might trigger compute automatically if logic says so (it does in my code)
    // But let's click button to be safe or check if results appear

    // Wait for results
    const resultsContainer = page.locator('#embedding-results');
    await expect(resultsContainer).toBeVisible();
    await expect(page.locator('#similarity-value')).not.toHaveText('--');

    // Check reasonable similarity for positive pair (should be high)
    const similarity = await page.locator('#similarity-value').innerText();
    console.log('Similarity:', similarity);
    expect(parseFloat(similarity)).toBeGreaterThan(0.5);

    // Verify inference time is displayed
    const inferenceTime = page.locator('#inference-time');
    await expect(inferenceTime).toContainText('ms');
    await expect(inferenceTime).not.toContainText('- ms');
  });
});
