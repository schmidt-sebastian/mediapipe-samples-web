export function renderMobileNav(container: HTMLElement) {
  container.innerHTML = `
      <img src="https://chuoling.github.io/mediapipe/images/mediapipe_small.png" alt="MediaPipe Logo" style="height: 30px; margin-right: 10px;">
      <select id="mobile-task-select" class="mobile-task-select">
        <option value="#/vision/object_detector">Object Detection</option>
        <option value="#/vision/image_segmenter">Image Segmentation</option>
      </select>
  `;

  const select = document.getElementById('mobile-task-select') as HTMLSelectElement;
  
  // Sync select with current hash
  const updateSelect = () => {
    const hash = window.location.hash || '#/vision/object_detector';
    // Handle potential trailing slashes or varying formats if necessary
    // For now, exact match or default
    if (hash.includes('image_segmenter')) {
      select.value = '#/vision/image_segmenter';
    } else {
      select.value = '#/vision/object_detector';
    }
  };

  updateSelect();
  window.addEventListener('hashchange', updateSelect);

  select.addEventListener('change', (e) => {
    const target = (e.target as HTMLSelectElement).value;
    window.location.hash = target;
  });
}
