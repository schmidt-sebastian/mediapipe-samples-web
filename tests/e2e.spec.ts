import { test, expect } from '@playwright/test';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

test.describe('Navigation & UI', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should redirect to object detection by default', async ({ page }) => {
    await expect(page).toHaveURL(/.*#\/vision\/object_detector/);
    await expect(page.locator('.sidebar-nav .active')).toContainText('Object Detection');
  });

  test('should navigate between tasks', async ({ page }) => {
    await page.click('a[data-task="image-segmentation"]');
    await expect(page).toHaveURL(/.*#\/vision\/image_segmenter/);
    await expect(page.locator('h2')).toContainText('Image Segmentation');

    await page.click('a[data-task="object-detection"]');
    await expect(page).toHaveURL(/.*#\/vision\/object_detector/);
    await expect(page.locator('h2')).toContainText('Object Detection');
  });

  test('should have responsive sidebar', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    // Check if sidebar nav is hidden (it might be hidden by CSS, let's check visibility)
    // .sidebar-nav display: none in media query
    await expect(page.locator('.sidebar-nav')).toBeHidden();

    // Toggle menu (using mobile header toggle)
    await page.click('.mobile-header .menu-toggle');
    await expect(page.locator('.sidebar-nav')).toBeVisible();

    // Toggle back (using sidebar toggle, since mobile one is covered)
    await page.click('.sidebar-header .menu-toggle');
    await expect(page.locator('.sidebar-nav')).toBeHidden();
  });
});
