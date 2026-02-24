export function renderMobileNav(container: HTMLElement) {
  container.innerHTML = `
      <div style="display: flex; align-items: center; margin-right: 10px;">
        <span class="material-icons" style="color: #007f8b; font-size: 24px;">analytics</span>
      </div>
      <select id="mobile-task-select" class="mobile-task-select">
        <option value="#/vision/object_detector">Object Detection</option>
        <option value="#/vision/face_detector">Face Detection</option>
        <option value="#/vision/image_classifier">Image Classification</option>
        <option value="#/vision/image_segmenter">Image Segmentation</option>
        <option value="#/vision/interactive_segmenter">Interactive Segmentation</option>
        <option value="#/vision/gesture_recognizer">Gesture Recognition</option>
        <option value="#/vision/hand_landmarker">Hand Landmark Detection</option>
        <option value="#/vision/image_embedder">Image Embedding</option>
        <option value="#/vision/face_landmarker">Face Landmark Detection</option>
        <option value="#/vision/pose_landmarker">Pose Landmark Detection</option>
        <option value="#/audio/audio_classifier">Audio Classifier</option>
        <option value="#/text/text_classifier">Text Classification</option>
        <option value="#/text/language_detector">Language Detection</option>
        <option value="#/text/text_embedder">Text Embedding</option>
      </select>
  `;

  const select = document.getElementById('mobile-task-select') as HTMLSelectElement;
  
  // Sync select with current hash
  const updateSelect = () => {
    const hash = window.location.hash || '#/vision/object_detector';
    const options = Array.from(select.options).map(option => option.value);
    select.value = options.includes(hash) ? hash : '#/vision/object_detector';
  };

  updateSelect();
  window.addEventListener('hashchange', updateSelect);

  select.addEventListener('change', (e) => {
    const target = (e.target as HTMLSelectElement).value;
    window.location.hash = target;
  });
}
