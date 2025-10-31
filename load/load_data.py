import pandas as pd
from sqlalchemy import create_engine
import os

def init_db(db_path="data/weather_data.db"):
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("se utilizara base local SQLite (modo desarrollo)")
        db_url = "sqlite:///data/weather_data.db"
    else:
        print(f"conectando con postgreSQL: {db_url}")
    engine = create_engine(db_url)
    return engine

def save_to_database(df, table_name, engine):
    df.to_sql(table_name, con=engine, if_exists="append", index=False)

def read_from_database(table_name, engine):
    query = f"SELECT * FROM {table_name}"
    return pd.read_sql(query, con=engine)
