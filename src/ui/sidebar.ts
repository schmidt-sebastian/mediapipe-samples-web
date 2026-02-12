export function renderSidebar(container: HTMLElement) {
  container.innerHTML = `
    <div class="sidebar-header" style="justify-content: center;">
      <img src="https://chuoling.github.io/mediapipe/images/mediapipe_small.png" alt="MediaPipe Logo" style="height: 40px;">
    </div>
    <nav class="sidebar-nav">
      <ul>
        <li><a href="#/vision/object_detector" class="nav-button active" data-task="object-detection">Object Detection</a></li>
        <li><a href="#/vision/image_segmenter" class="nav-button" data-task="image-segmentation">Image Segmentation</a></li>
      </ul>
    </nav>
  `;
}
