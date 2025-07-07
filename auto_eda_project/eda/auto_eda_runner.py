import os
import pandas as pd
from ydata_profiling import ProfileReport

def run_autoeda(df: pd.DataFrame, output_path: str, report_name="autoeda_report.html"):
    """
    Generates an EDA report using ydata-profiling and saves it as an HTML file.
    """
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, report_name)

    profile = ProfileReport(df, title="Auto EDA Report", explorative=True)
    profile.to_file(output_file)

    print(f"AutoEDA report saved at: {output_file}")
