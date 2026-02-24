import { test, expect } from '@playwright/test';

test.describe('Holistic Landmarker Task', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the Holistic Landmarker page
    await page.goto('/#/vision/holistic_landmarker');

    // Wait for the page to settle
    await page.waitForLoadState('networkidle');
  });

  test('should load model and handle image inference', async ({ page }) => {
    // 1. Initial State: "Initializing..." then "Loading Model..." then "Ready"
    const status = page.locator('#status-message');

    // Wait for Ready status (model active)
    await expect(status).toHaveText('Ready', { timeout: 30000 });

    // 2. Select Image Tab (should be default)
    const tabImage = page.locator('#tab-image');
    await expect(tabImage).toHaveClass(/active/);

    // 3. Verify Default Image is present
    const testImage = page.locator('#test-image');
    await expect(testImage).toBeVisible();

    // 4. Trigger detection if not already auto-triggered, or verify results
    // The code auto-triggers on 'Ready' if image is present.
    // Wait for "Done in ..." 
    await expect(status).toHaveText(/Done in/, { timeout: 10000 });

    // 5. Verify Inference Time is updated
    const inferenceTime = page.locator('#inference-time');
    await expect(inferenceTime).toHaveText(/Inference Time: \d+\.\d+ ms/);
  });

  test('should handle delegate switching', async ({ page }) => {
    const status = page.locator('#status-message');
    await expect(status).toHaveText('Ready', { timeout: 30000 });

    // Switch to CPU
    await page.selectOption('#delegate-select', 'CPU');

    // Should trigger re-initialization
    await expect(status).toHaveText(/Loading Model|Ready/);
    // Eventually Ready again
    await expect(status).toHaveText('Ready', { timeout: 20000 });
  });
});
