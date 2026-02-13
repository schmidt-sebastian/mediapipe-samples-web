export function renderSidebar(container: HTMLElement) {
  container.innerHTML = `
    <div class="sidebar-header" style="justify-content: flex-start; gap: 16px;">
      <button class="menu-toggle material-icons" style="color: var(--text-secondary); background: none; border: none; font-size: 24px; cursor: pointer; padding: 0;">menu_open</button>
      <img src="https://chuoling.github.io/mediapipe/images/mediapipe_small.png" alt="MediaPipe Logo" style="height: 32px;">
    </div>
    <nav class="sidebar-nav">
      <ul>
        <li><a href="#/vision/object_detector" class="nav-button active" data-task="object-detection">Object Detection</a></li>
        <li><a href="#/vision/image_segmenter" class="nav-button" data-task="image-segmentation">Image Segmentation</a></li>
      </ul>
    </nav>
  `;
}
