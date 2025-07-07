# auto_eda_project/data_ingestion/data_loader.py

import os
import pandas as pd
from sqlalchemy import create_engine
from auto_eda_project.db.db_connect import get_engine


class DataIngestionError(Exception):
    """Custom exception for data ingestion issues."""
    pass


def load_data(file_path: str = None, from_db: bool = False, table_name: str = None, schema: str = 'public') -> pd.DataFrame:
    """
    Load data from CSV/Excel/JSON/Parquet or from PostgreSQL database.

    Args:
        file_path (str): File path to CSV, Excel, etc.
        from_db (bool): If True, loads from PostgreSQL table.
        table_name (str): Table name in PostgreSQL (required if from_db=True).
        schema (str): Schema name (default is 'public').

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        DataIngestionError: If loading fails.
    """
    if from_db:
        if not table_name:
            raise DataIngestionError("Table name must be specified when loading from database.")

        try:
            engine = get_engine()
            query = f"SELECT * FROM {schema}.{table_name}"
            df = pd.read_sql(query, con=engine)
            print(f"✅ Loaded data from PostgreSQL: {schema}.{table_name}")
            print(f"Shape: {df.shape}")
            return df
        except Exception as e:
            raise DataIngestionError(f"Failed to load data from PostgreSQL: {str(e)}")

    if not file_path or not os.path.exists(file_path):
        raise DataIngestionError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == '.csv':
            df = pd.read_csv(file_path)
        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        elif ext == '.json':
            df = pd.read_json(file_path)
        elif ext == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise DataIngestionError(f"Unsupported file format: {ext}")

        print(f"📄 Loaded file: {file_path} | Shape: {df.shape}")
        return df

    except Exception as e:
        raise DataIngestionError(f"Failed to load file: {str(e)}")
