# final_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load all results
models = ['fp16', 'q4_k_m']
all_results = {}

for model in models:
    df = pd.read_csv(f'./results/{model}_complete_classified.csv')
    all_results[model] = df

# 1. Overall refusal rates
summary = pd.DataFrame({
    'Model': models,
    'Refusal Rate (%)': [all_results[m]['is_refusal'].mean()*100 for m in models],
    'n': [len(all_results[m]) for m in models]
})

print("="*60)
print("FINAL RESULTS:")
print("="*60)
print(summary.to_string(index=False))

# 2. Main visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart
ax = axes[0]
bars = ax.bar(summary['Model'], summary['Refusal Rate (%)'], 
              color=['#2ecc71', '#f39c12', '#e74c3c'])
ax.set_ylabel('Refusal Rate (%)', fontsize=12)
ax.set_xlabel('Model Variant', fontsize=12)
ax.set_title('Safety Degradation Across Quantization Levels', fontsize=14, fontweight='bold')
ax.set_ylim([0, 100])
ax.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, summary['Refusal Rate (%)']):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Statistical significance (simple)
# Compare FP16 vs Q4
from scipy import stats
fp16_refusals = all_results['fp16']['is_refusal'].values
q4_refusals = all_results['q4_k_m']['is_refusal'].values
chi2, p_value = stats.chi2_contingency([
    [fp16_refusals.sum(), (~fp16_refusals).sum()],
    [q4_refusals.sum(), (~q4_refusals).sum()]
])[:2]

ax.text(0.5, 0.95, f'FP16 vs Q4: p < 0.001' if p_value < 0.001 else f'p = {p_value:.3f}',
        transform=ax.transAxes, ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# Response length comparison
ax = axes[1]
lengths_data = [all_results[m]['tokens_generated'].values for m in models]
bp = ax.boxplot(lengths_data, labels=models, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('#3498db')
ax.set_ylabel('Response Length (tokens)', fontsize=12)
ax.set_xlabel('Model Variant', fontsize=12)
ax.set_title('Response Length Distribution', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('./results/final_results.png', dpi=300, bbox_inches='tight')
print("\nSaved visualization to ./results/final_results.png")

# 3. Category-wise breakdown (if AdvBench has categories)
# This requires parsing AdvBench categories
# Skip for now, add later if time

# Save final summary
summary.to_csv('./results/final_summary.csv', index=False)