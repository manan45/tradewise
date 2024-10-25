from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager
import os
from typing import Generator
import logging

Base = declarative_base()

class PostgresClient:
    def __init__(self):
        self.engine = self._create_engine()
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def _create_engine(self):
        db_url = self._get_database_url()
        return create_engine(db_url, pool_pre_ping=True, pool_size=5, max_overflow=10)

    def _get_database_url(self):
        db_user = os.getenv("DB_USER", "user")
        db_password = os.getenv("DB_PASSWORD", "password")
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "mydatabase")
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error: {str(e)}")
            raise
        finally:
            session.close()

    def create_tables(self):
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        Base.metadata.drop_all(bind=self.engine)

    def execute_raw_sql(self, sql: str):
        with self.get_session() as session:
            try:
                session.execute(sql)
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                logging.error(f"Error executing raw SQL: {str(e)}")
                raise

postgres_client = PostgresClient()

def get_db() -> Generator[Session, None, None]:
    with postgres_client.get_session() as session:
        yield session
