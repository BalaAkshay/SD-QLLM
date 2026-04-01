# check_classifier_accuracy.py
import pandas as pd

df = pd.read_csv('./results/manual_validation.csv')

# Filter rows where manual classification is done
df_checked = df[df['manual_classification'].isin(['R', 'C', 'P'])]

# Map to boolean
df_checked['manual_refusal'] = df_checked['manual_classification'] == 'R'
df_checked['auto_refusal'] = df_checked['classification'] == 'refusal'

# Calculate agreement
agreement = (df_checked['manual_refusal'] == df_checked['auto_refusal']).mean()
print(f"Classifier accuracy: {agreement*100:.1f}%")

# Confusion matrix
print("\nConfusion matrix:")
print(pd.crosstab(df_checked['manual_refusal'], df_checked['auto_refusal'], 
                  rownames=['Manual'], colnames=['Auto']))