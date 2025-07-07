# auto_eda_project/db/ingest_data.py

import os
import sys
import pandas as pd

# ✅ Add project root dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from auto_eda_project.db.db_connect import get_engine

def ingest_csv_to_postgres(csv_path, table_name, schema='public'):
    """
    Ingest a CSV file into a PostgreSQL table.

    Args:
        csv_path (str): Path to CSV file.
        table_name (str): Destination table name.
        schema (str): Database schema to use (default: 'public').
    """
    try:
        # Get DB engine
        engine = get_engine()

        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"📦 Loaded CSV from {csv_path} with shape: {df.shape}")

        # Ingest to PostgreSQL
        df.to_sql(name=table_name, con=engine, schema=schema,
                  if_exists='replace', index=False)
        print(f"✅ Data written to PostgreSQL table: {schema}.{table_name}")

        # Optional: Preview sample
        preview_query = f"SELECT * FROM {schema}.{table_name} LIMIT 5"
        preview_df = pd.read_sql(preview_query, con=engine)
        print("📄 Sample from ingested table:")
        print(preview_df)

    except Exception as e:
        print("❌ Error during ingestion:", str(e))

# --------------------------------------------
# ✅ Add this to make the script executable
# --------------------------------------------
if __name__ == "__main__":
    csv_path = os.path.join("auto_eda_project", "Data", "Software_Salaries.csv")
    table_name = "software_salaries"
    ingest_csv_to_postgres(csv_path=csv_path, table_name=table_name)
