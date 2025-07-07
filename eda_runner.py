import os
import sys
import pandas as pd

# 🔧 Add project root to sys.path so imports work
sys.path.append(os.path.abspath("auto_eda_project"))

# ✅ Imports from eda modules
from eda.auto_eda_runner import run_autoeda
from eda.eda_visuals import plot_univariate, plot_bivariate, plot_boxplots
from preprocessing.preprocessing import get_preprocessor
from eda.processed_visuals import plot_processed_univariate, plot_processed_boxplots
# 📍 Data path
DATA_PATH = os.path.join("auto_eda_project", "Data", "Software_Salaries.csv")

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ File not found: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)

    # 🔎 Step 1: Run AutoEDA report
    run_autoeda(df, output_path="eda_output", report_name="autoeda_report.html")

    # 📊 Step 2: Standard EDA visuals
    plot_univariate(df)
    plot_boxplots(df)
    plot_bivariate(df, target='adjusted_total_usd')

    # 🧼 Step 3: After preprocessing — visualize transformed distributions
    preprocessor = get_preprocessor(df)
    plot_processed_univariate(df)
    plot_processed_boxplots(df)

    print("✅ EDA completed successfully!")

if __name__ == "__main__":
    main()
