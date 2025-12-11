"""
MODULO DE CARGA
"""

import pandas as pd
from sqlalchemy import create_engine, Engine
from src.config.settings import get_db_url

def init_db_connection() -> Engine:
    """inicializa la conexion a la base de datos."""
    db_url = get_db_url()
    print(f"iniciando conexion a DB: {db_url}")
    return create_engine(db_url)

def save_to_database(df: pd.DataFrame, table_name: str, engine: Engine):
    """
    carga el DataFrame en una tabla SQL
    """
    try:
        df.to_sql(table_name, con=engine, if_exists="append", index=False)
        print(f"datos cargados exitosamente en tabla '{table_name}'")
    except Exception as e:
        print(f"error cargando a base de datos: {e}")

def read_from_database(table_name: str, engine: Engine) -> pd.DataFrame:
    """lee datos desde una tabla SQL."""
    query = f"SELECT * FROM {table_name}"
    return pd.read_sql(query, con=engine)
