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

export interface ClassificationItem {
  label: string;
  score: number; // 0.0 to 1.0
}

export class ClassificationResult {
  private container: HTMLElement;

  constructor(containerId: string) {
    const el = document.getElementById(containerId);
    if (!el) throw new Error(`ClassificationResult: container ${containerId} not found`);
    this.container = el;
  }

  public updateResults(results: ClassificationItem[]) {
    this.container.innerHTML = '';
    
    if (results.length === 0) {
      results = [{ label: 'No results', score: 0 }];
    }

    results.forEach(result => {
      const scorePercent = Math.round(result.score * 100);
      const row = document.createElement('div');
      row.className = 'classification-row';
      row.innerHTML = `
        <div class="category-name">${result.label || 'Unknown'}</div>
        <div class="score-container">
          <div class="score-track">
            <div class="score-fill" style="width: ${scorePercent}%"></div>
          </div>
          <div class="score-text">${scorePercent}%</div>
        </div>
      `;
      this.container.appendChild(row);
    });
  }

  public clear() {
    this.container.innerHTML = '';
  }
}
