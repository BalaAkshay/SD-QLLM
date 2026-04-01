import pandas as pd
import requests

# Download AdvBench harmful behaviors
url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
df = pd.read_csv(url)

print(f"Total prompts in AdvBench: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 5 prompts:")
print(df.head())

# For initial testing: use subset
# For final experiments: use full dataset or random sample of 300-500

df_subset = df.sample(n=20, random_state=42)
df_subset.to_csv('./data/advbench_subset_20.csv', index=False)

# df_subset = df.sample(n=100, random_state=42)
# df_subset.to_csv('./data/advbench_subset_100.csv', index=False)

# df_full = df.sample(n=520, random_state=42)  # Adjust based on time
# df_full.to_csv('./data/advbench_full.csv', index=False)

print(f"\nSaved subset (100 prompts) to ./data/advbench_subset_100.csv")
print(f"Saved full set (300 prompts) to ./data/advbench_full.csv")

# Inspect prompt format
print("\nExample prompt:")
print(df.iloc[0]['goal'])