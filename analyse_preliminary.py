# analyze_preliminary.py
import pandas as pd
import matplotlib.pyplot as plt

summary = pd.read_csv('./results/summary_all_prompts.csv')

print("PRELIMINARY RESULTS:")
print(summary)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(summary['model'], summary['refusal_rate'])
ax.set_ylabel('Refusal Rate (%)')
ax.set_xlabel('Model Variant')
ax.set_title('Safety Behavior Across Quantization Levels (N=520)')
ax.set_ylim([0, 100])

for i, v in enumerate(summary['refusal_rate']):
    ax.text(i, v + 2, f"{v:.1f}%", ha='center')

plt.tight_layout()
plt.savefig('./results/preliminary_results.png', dpi=300)
plt.show()

print("\nSaved plot to ./results/preliminary_results.png")