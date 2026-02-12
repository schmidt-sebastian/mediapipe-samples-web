import { test, expect } from '@playwright/test';

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

    // Toggle menu
    await page.click('.menu-toggle');
    await expect(page.locator('.sidebar-nav')).toBeVisible();

    // Toggle back
    await page.click('.menu-toggle');
    await expect(page.locator('.sidebar-nav')).toBeHidden();
  });
});
