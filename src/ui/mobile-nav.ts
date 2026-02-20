export function renderMobileNav(container: HTMLElement) {
  container.innerHTML = `
      <div style="display: flex; align-items: center; margin-right: 10px;">
        <span class="material-icons" style="color: #007f8b; font-size: 24px;">analytics</span>
      </div>
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
