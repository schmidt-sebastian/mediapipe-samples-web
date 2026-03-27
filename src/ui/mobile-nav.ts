/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

export function renderMobileNav(container: HTMLElement) {
  container.innerHTML = `
      <div style="display: flex; align-items: center; margin-right: 10px;">
        <span class="material-icons" style="color: #007f8b; font-size: 24px;">analytics</span>
      </div>
      <select id="mobile-task-select" class="mobile-task-select">
        <option value="#/vision/face_detector">Face Detection</option>
        <option value="#/vision/face_landmarker">Face Landmarker</option>
        <option value="#/vision/hand_landmarker">Hand Landmarker</option>
        <option value="#/vision/pose_landmarker">Pose Landmarker</option>
        <option value="#/vision/holistic_landmarker">Holistic Landmarker</option>
  <option value="#/vision/image_classifier">Image Classifier</option>
        <option value="#/vision/gesture_recognizer">Gesture Recognizer</option>
        <option value="#/vision/image_embedder">Image Embedding</option>
        <option value="#/vision/interactive_segmenter">Interactive Segmenter</option>
        <option value="#/vision/image_segmenter">Image Segmentation</option>
        <option value="#/vision/object_detector">Object Detection</option>
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
    // Handle potential trailing slashes or varying formats if necessary
    // For now, exact match or default
    if (hash.includes('interactive_segmenter')) {
      select.value = '#/vision/interactive_segmenter';
    } else if (hash.includes('image_segmenter')) {
      select.value = '#/vision/image_segmenter';
    } else if (hash.includes('face_landmarker')) {
      select.value = '#/vision/face_landmarker';
    } else if (hash.includes('hand_landmarker')) {
      select.value = '#/vision/hand_landmarker';
    } else if (hash.includes('pose_landmarker')) {
      select.value = '#/vision/pose_landmarker';
    } else if (hash.includes('holistic_landmarker')) {
      select.value = '#/vision/holistic_landmarker';
    } else if (hash.includes('image_classifier')) {
      select.value = '#/vision/image_classifier';
    } else if (hash.includes('gesture_recognizer')) {
      select.value = '#/vision/gesture_recognizer';
    } else if (hash.includes('face_detector')) {
      select.value = '#/vision/face_detector';
    } else if (hash.includes('audio_classifier')) {
      select.value = '#/audio/audio_classifier';
    } else if (hash.includes('text_classifier')) {
      select.value = '#/text/text_classifier';
    } else if (hash.includes('text_embedder')) {
      select.value = '#/text/text_embedder';
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
