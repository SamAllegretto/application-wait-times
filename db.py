import os
from contextlib import ExitStack, contextmanager

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


@contextmanager
def get_engine():
    with ExitStack() as stack:
        host, port = os.environ["POSTGRES_HOST"], 5432
        if os.getenv("DB_USE_SSH_TUNNEL", "false").lower() == "true":
            from sshtunnel import SSHTunnelForwarder

            tunnel = stack.enter_context(SSHTunnelForwarder(
                (os.environ["SSH_HOST"], 22),
                ssh_username=os.environ["SSH_USER"],
                ssh_pkey=os.environ["SSH_KEY_PATH"],
                remote_bind_address=(host, port),
            ))
            host, port = "127.0.0.1", tunnel.local_bind_port
        engine = create_engine(
            f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
            f"@{host}:{port}/{os.environ['POSTGRES_DB']}",
            connect_args={"sslmode": "require"},
        )
        yield engine
        engine.dispose()


def fetch(schema: str, table: str) -> pd.DataFrame:
    with get_engine() as engine:
        return pd.read_sql_table(table, engine, schema=schema)
