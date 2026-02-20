export function renderSidebar(container: HTMLElement) {
  container.innerHTML = `
    <div class="sidebar-header">
      <button class="menu-toggle material-icons" style="margin-right: 12px; color: var(--text-secondary); background: none; border: none; font-size: 24px; cursor: pointer;">menu_open</button>
      <div class="sidebar-logo-text">
        <span class="material-icons" style="color: var(--primary); font-size: 28px;">analytics</span>
        <span>MediaPipe Tasks</span>
      </div>
    </div>
    <nav class="sidebar-nav">
      <div class="category-header">Vision</div>
      <ul>
        <li><a href="#/vision/object_detector" class="nav-button" data-task="object-detection">Object Detection</a></li>
        <li><a href="#/vision/image_segmenter" class="nav-button" data-task="image-segmentation">Image Segmentation</a></li>
      </ul>

      <div class="category-header">Audio</div>
      <ul>
        <li><a href="#/audio/audio_classifier" class="nav-button" data-task="audio-classifier">Audio Classifier</a></li>
      </ul>

      <div class="category-header">Text</div>
      <ul>
        <li><a href="#/text/text_classifier" class="nav-button" data-task="text-classification">Text Classification</a></li>
        <li><a href="#/text/text_embedder" class="nav-button" data-task="text-embedding">Text Embedding</a></li>
      </ul>
    </nav>
  `;
}
