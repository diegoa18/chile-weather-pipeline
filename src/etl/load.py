import logging

import pandas as pd
from sqlalchemy import Engine, create_engine

from src.config.settings import get_db_url

logger = logging.getLogger(__name__)


def init_db_connection() -> Engine:
    db_url = get_db_url()
    logger.info("iniciando conexion a DB: %s", db_url)
    return create_engine(db_url)


def save_to_database(df: pd.DataFrame, table_name: str, engine: Engine):
    try:
        df.to_sql(table_name, con=engine, if_exists="append", index=False)
        logger.info("datos cargados exitosamente en tabla '%s'", table_name)
    except Exception as e:
        logger.error("error cargando a base de datos: %s", e)


def read_from_database(table_name: str, engine: Engine) -> pd.DataFrame:
    query = f"SELECT * FROM {table_name}"
    return pd.read_sql(query, con=engine)
