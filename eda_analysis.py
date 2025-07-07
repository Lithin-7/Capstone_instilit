import os
import pandas as pd
import sys

# Add project path
sys.path.append(os.path.abspath("auto_eda_project"))

from preprocessing.preprocessing import get_preprocessor
from eda.processed_eda import analyze_skewness, analyze_outliers

if __name__ == "__main__":
    data_path = os.path.join("auto_eda_project", "Data", "Software_Salaries.csv")
    df = pd.read_csv(data_path)

    # Build and apply preprocessor
    preprocessor = get_preprocessor(df)
    processed_data = preprocessor.fit_transform(df)
    feature_names = preprocessor.get_feature_names_out()
    processed_df = pd.DataFrame(processed_data, columns=feature_names)

    # Run EDA
    analyze_skewness(processed_df)
    analyze_outliers(processed_df)
