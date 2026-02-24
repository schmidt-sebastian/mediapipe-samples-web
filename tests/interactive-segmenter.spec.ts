import { test, expect } from '@playwright/test';

test.describe('Interactive Segmenter Task', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.click('a[href="#/vision/interactive_segmenter"]');
    await page.waitForSelector('#status-message', { state: 'visible', timeout: 30000 });
  });

  test('should load model and handle click interaction', async ({ page }) => {
    // Wait for model ready
    await expect(page.locator('#status-message')).toHaveText('Ready', { timeout: 30000 });

    // Click on the image to trigger segmentation
    // Wait for Ready status before interaction
    await expect(page.locator('#status-message')).toHaveText('Ready', { timeout: 30000 });

    // We click near the center of the image (assuming cat/dog object is central)
    const testImage = page.locator('#test-image');
    await expect(testImage).toBeVisible();
    await testImage.click({ position: { x: 150, y: 150 } }); // Assuming 300x300 image roughly

    // Wait for "Done in ..." status
    await expect(page.locator('#status-message')).toHaveText(/Done in/, { timeout: 15000 });

    // Check inference time
    await expect(page.locator('#inference-time')).toContainText('Inference Time:');
  });

  test('should handle delegate switching', async ({ page }) => {
    await page.selectOption('#delegate-select', 'CPU');
    await expect(page.locator('#status-message')).toHaveText('Ready', { timeout: 60000 });

    await page.selectOption('#delegate-select', 'GPU');
    await expect(page.locator('#status-message')).toHaveText('Ready', { timeout: 60000 });
  });
});
