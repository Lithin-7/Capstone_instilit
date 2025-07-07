# auto_eda_project/db/connect.py

from sqlalchemy import create_engine

def get_engine(user='postgres', password='lithin', host='localhost', db='instilit'):
    """
    Returns SQLAlchemy engine for given credentials.
    """
    DATABASE_URI = f"postgresql+psycopg2://{user}:{password}@{host}/{db}"
    engine = create_engine(DATABASE_URI)
    return engine
