import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("ipl_colab.csv")

print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

# Example: top winning teams
if "winner" in df.columns:
    plt.figure(figsize=(10,5))
    sns.countplot(y="winner", data=df, order=df["winner"].value_counts().index)
    plt.title("Most Winning Teams")
    plt.show()

# Save quick summary
df.describe(include='all').to_csv("eda_summary.csv")
print("\nEDA Summary saved as eda_summary.csv")

