import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_skewness(df, threshold=1.0, save_dir="eda_output/processed_eda/skewness"):
    """
    Computes and plots skewness for numerical features in the dataset.
    
    Parameters:
    - df: Preprocessed DataFrame (numerical)
    - threshold: Absolute skew value above which a feature is considered skewed
    - save_dir: Directory to save skewness plots
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\n📈 Skewness Analysis:")
    skewness = df.skew(numeric_only=True)

    for col, skew in skewness.items():
        if abs(skew) > threshold:
            print(f"⚠️ Skewed Feature: {col} | Skewness: {skew:.2f}")

            # Plot histogram with KDE
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col], kde=True, bins=30, color="orange")
            plt.title(f"Skewed Feature: {col}\nSkewness = {skew:.2f}")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{col}_skew.png")
            plt.close()

def analyze_outliers(df, save_dir="eda_output/processed_eda/outliers", z_thresh=3.0):
    """
    Detects and plots outliers using z-score method.

    Parameters:
    - df: Preprocessed DataFrame (numerical)
    - save_dir: Directory to save boxplots
    - z_thresh: Z-score threshold for outlier detection
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\n📦 Outlier Analysis:")
    for col in df.select_dtypes(include='number').columns:
        col_zscore = (df[col] - df[col].mean()) / df[col].std()
        outliers = df[np.abs(col_zscore) > z_thresh]

        if not outliers.empty:
            print(f"⚠️ Outliers detected in {col}: {len(outliers)} rows")

            # Boxplot
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=df[col], color="salmon")
            plt.title(f"Outliers in {col}")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{col}_outliers.png")
            plt.close()
