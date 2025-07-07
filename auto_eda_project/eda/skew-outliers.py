import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_skewness(df, threshold=1.0):
    print("\n📈 Skewness Analysis:")
    for col in df.select_dtypes(include='number').columns:
        if df[col].dropna().nunique() <= 2:
            continue  # Skip binary columns
        skew_val = df[col].skew()
        if abs(skew_val) > threshold:
            print(f"⚠️ Skewed Feature: {col} | Skewness: {skew_val:.2f}")

def analyze_outliers(df, threshold=1.5):
    print("\n📦 Outlier Analysis:")
    for col in df.select_dtypes(include='number').columns:
        if df[col].dropna().nunique() <= 2:
            continue  # Skip binary columns
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        if len(outliers) > 0:
            print(f"⚠️ Outliers detected in {col}: {len(outliers)} rows")


    
