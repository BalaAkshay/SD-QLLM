# create_manual_validation.py
import pandas as pd
import random


all_results = []
for model in ['q4_k_m', 'q8_0', 'fp16']:
    df = pd.read_csv(f'./results/{model}_complete_classified.csv')
    sample = df.sample(n=30, random_state=42)
    all_results.append(sample)

df_manual = pd.concat(all_results)
df_manual['manual_classification'] = ''
df_manual['notes'] = ''


df_manual[['model', 'prompt', 'response', 'classification', 'manual_classification', 'notes']].to_csv(
    './results/manual_validation.csv', index=False
)

print("Created manual_validation.csv with 30 samples")
print("Open in Excel/Google Sheets and add manual classifications")