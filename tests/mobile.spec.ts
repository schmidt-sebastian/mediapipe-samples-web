import { test, expect } from '@playwright/test';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Mobile Layout & Navigation', () => {
  let imagePath: string;

  // Mobile viewport (iPhone SE 2020)
  test.use({ viewport: { width: 375, height: 667 } });

  test.beforeAll(() => {
    imagePath = path.resolve(__dirname, 'assets', 'dog.jpg');
  });

  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should hide sidebar and show mobile header', async ({ page }) => {
    // Check sidebar hidden
    const sidebar = page.locator('.sidebar');
    const display = await sidebar.evaluate((el) => window.getComputedStyle(el).display);
    console.log(`Sidebar display: ${display}`);
    await expect(sidebar).toBeHidden();

    // Check mobile header visible
    const mobileHeader = page.locator('.mobile-header');
    await expect(mobileHeader).toBeVisible();
  });

  test('should navigate using dropdown', async ({ page }) => {
    const select = page.locator('#mobile-task-select');
    await expect(select).toBeVisible();

    // Ensure we start at Object Detection (default or explicit)
    await page.goto('/#/vision/object_detector');
    await expect(select).toHaveValue('#/vision/object_detector');

    // Select Image Segmentation
    await select.selectOption('#/vision/image_segmenter');

    // Should update URL and content
    await expect(page).toHaveURL(/.*image_segmenter/);
    await expect(page.locator('.output-header h2')).toHaveText('Image Segmentation');

    // Select back to Object Detection
    await select.selectOption('#/vision/object_detector');
    await expect(page).toHaveURL(/.*object_detector/);
    await expect(page.locator('.output-header h2')).toHaveText('Object Detection');
  });

  test('should stack controls vertically on mobile', async ({ page }) => {
    await page.goto('/#/vision/image_segmenter');

    // .task-container should be column flex direction
    const taskContainer = page.locator('.task-container');
    await expect(taskContainer).toHaveCSS('flex-direction', 'column');

    // Controls panel should be full width (or close to it)
    const controlsPanel = page.locator('.controls-panel');
    const controlsBox = await controlsPanel.boundingBox();
    
    // Output should be below controls
    const outputPanel = page.locator('.output-panel');
    const outputBox = await outputPanel.boundingBox();

    if (controlsBox && outputBox) {
      expect(outputBox.y).toBeGreaterThan(controlsBox.y);
      // Width should be roughly viewport width (minus padding)
      expect(controlsBox.width).toBeGreaterThan(300); // 375 viewport
    }
  });

  test('should show enable webcam button nicely centered', async ({ page }) => {
    await page.goto('/#/vision/image_segmenter');
    
    // Wait for model to load
    await expect(page.locator('#status-message')).toHaveText(/(Ready)|(Done)/, { timeout: 30000 });

    // Switch to Webcam tab
    await page.click('#tab-webcam');
    
    const btn = page.locator('#webcamButton');
    await expect(btn).toBeVisible();
    // Webcam auto-starts on tab switch in this app version
    await expect(btn).toHaveText(/(Enable Webcam)|(Disable Webcam)/);

    // Verify centering styles are applied
    // Verify centering
    await expect(btn).toHaveCSS('position', 'absolute');
    await expect(btn).toHaveCSS('bottom', '30px');

    // Check horizontal centering via bounding box (more robust than computed left %)
    const box = await btn.boundingBox();
    if (box) {
      const viewport = page.viewportSize();
      if (viewport) {
        const centerX = box.x + box.width / 2;
        const viewportCenterX = viewport.width / 2;
        expect(Math.abs(centerX - viewportCenterX)).toBeLessThan(2);
      }
    }
    // The container .cam-container should be visible too
    const camContainer = page.locator('.cam-container');
    await expect(camContainer).toBeVisible();
  });
});
