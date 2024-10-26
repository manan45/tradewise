import time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from contextlib import contextmanager
import os
from typing import Generator
import logging
import alembic.config
import alembic.command
from alembic.config import Config
from app.core.domain.entities.base import Base

class PostgresClient:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.engine = self._create_engine_with_retry()
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.run_migrations()

    def _create_engine_with_retry(self, max_retries=5, delay=5):
        for attempt in range(max_retries):
            try:
                return self._create_engine()
            except OperationalError as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"Failed to connect after {max_retries} attempts.")
                    raise

    def _create_engine(self):
        db_url = self._get_database_url()
        try:
            self.logger.info(f"Attempting to connect to database at {db_url}")
            engine = create_engine(db_url, pool_pre_ping=True, pool_size=5, max_overflow=10)
            engine.connect()
            self.logger.info(f"Successfully connected to database at {db_url}")
            return engine
        except OperationalError as e:
            self.logger.error(f"Failed to connect to database at {db_url}. Error: {str(e)}")
            raise

    def _get_database_url(self):
        db_user = os.getenv("DB_USER", "postgres")
        db_password = os.getenv("DB_PASSWORD", "password")
        db_port = os.getenv("DB_PORT", "5432")
        db_host = os.getenv("DB_HOST", "127.0.0.1")
        db_name = os.getenv("DB_NAME", "stockdb")
        self.logger.info(f"Database connection details: HOST={db_host}, PORT={db_port}, DB={db_name}, USER={db_user}")
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    def run_migrations(self):
        try:
            self.logger.info("Running database migrations...")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
            alembic_ini_path = os.path.join(project_root, 'alembic.ini')
            
            alembic_cfg = Config(alembic_ini_path)
            alembic_cfg.set_main_option("script_location", os.path.join(project_root, "migrations"))
            alembic_cfg.set_main_option("sqlalchemy.url", self._get_database_url())
            
            alembic.command.upgrade(alembic_cfg, "head")
            self.logger.info("Database migrations completed successfully.")
        except Exception as e:
            self.logger.error(f"Error running migrations: {str(e)}")
            raise

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            self.logger.error(f"Database error: {str(e)}")
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
                self.logger.error(f"Error executing raw SQL: {str(e)}")
                raise

postgres_client = PostgresClient()

def get_db() -> Generator[Session, None, None]:
    with postgres_client.get_session() as session:
        yield session
