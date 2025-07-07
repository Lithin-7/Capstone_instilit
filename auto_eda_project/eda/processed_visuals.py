import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="whitegrid")

# 📊 Univariate Plots for Processed Data
def plot_processed_univariate(df: pd.DataFrame, output_dir: str = "eda_output/processed_eda/univariate"):
    os.makedirs(output_dir, exist_ok=True)
    for col in df.select_dtypes(include='number').columns:
        if df[col].dropna().empty:
            continue
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Processed Univariate: {col}')
        plt.tight_layout()
        safe_col = col.replace('/', '_').replace(' ', '_').replace('|', '_')
        plt.savefig(os.path.join(output_dir, f"{safe_col}_univariate.png"))
        plt.close()


# 📦 Boxplots for Processed Numerical Features
def plot_processed_boxplots(df: pd.DataFrame, output_dir: str = "eda_output/processed_eda/boxplots"):
    os.makedirs(output_dir, exist_ok=True)

    for col in df.select_dtypes(include='number').columns:
        if df[col].dropna().empty:
            continue
        plt.figure(figsize=(8, 1.5))
        try:
            sns.boxplot(x=df[col])
            plt.title(f'Processed Boxplot: {col}')
            plt.tight_layout()
            safe_col = col.replace('/', '_').replace(' ', '_').replace('|', '_')
            plt.savefig(os.path.join(output_dir, f"{safe_col}_boxplot.png"))
            plt.close()
        except Exception as e:
            print(f"❌ Skipping '{col}' due to error: {e}")
