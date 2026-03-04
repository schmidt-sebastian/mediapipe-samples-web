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
