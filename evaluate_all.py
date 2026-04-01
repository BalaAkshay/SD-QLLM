import pandas as pd
from evaluate_responses import classify_responses

models = ['q4_k_m', 'fp16']
results = {}

for model in models:
    print(f"\n{'='*60}")
    df = classify_responses(f'./results/{model}_complete.csv')
    results[model] = df


summary = pd.DataFrame({
    'model': models,
    'refusal_rate': [results[m]['is_refusal'].mean()*100 for m in models],
    'compliance_rate': [(~results[m]['is_refusal']).mean()*100 for m in models]
})

print("\n" + "="*60)
print("SUMMARY TABLE:")
print("="*60)
print(summary.to_string(index=False))

summary.to_csv('./results/summary_all_prompts.csv', index=False)