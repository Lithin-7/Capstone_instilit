# eda before processing

import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="whitegrid")

# 📊 Univariate Plots
def plot_univariate(df: pd.DataFrame, output_dir: str = "eda_output/plots/univariate"):
    os.makedirs(output_dir, exist_ok=True)
    for col in df.select_dtypes(include=['number', 'object', 'category']):
        plt.figure(figsize=(8, 4))
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            sns.countplot(x=col, data=df)
        else:
            sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f'Univariate Distribution: {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{col}_univariate.png")
        plt.close()

# 🔗 Bivariate with Target
def plot_bivariate(df: pd.DataFrame, target: str, output_dir: str = "eda_output/plots/bivariate"):
    os.makedirs(output_dir, exist_ok=True)
    for col in df.select_dtypes(include=['number', 'object']):
        if col == target:
            continue
        plt.figure(figsize=(8, 4))
        if df[col].dtype == 'object':
            sns.boxplot(x=col, y=target, data=df)
        else:
            sns.scatterplot(x=col, y=target, data=df)
        plt.title(f'Bivariate: {col} vs {target}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{col}_vs_{target}.png")
        plt.close()

# 📦 Boxplots for Numerical Features
def plot_boxplots(df: pd.DataFrame, output_dir: str = "eda_output/plots/boxplots"):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    os.makedirs(output_dir, exist_ok=True)

    for col in df.select_dtypes(include=['number']):
        if df[col].dropna().empty:
            print(f"⚠️ Skipping '{col}' - empty or invalid for boxplot.")
            continue
        plt.figure(figsize=(8, 1.5))
        try:
            sns.boxplot(x=df[col].dropna())
            plt.title(f'Boxplot: {col}')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{col}_boxplot.png")
            plt.close()
        except Exception as e:
            print(f"❌ Skipping '{col}' due to error: {e}")

