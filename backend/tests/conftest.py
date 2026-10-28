import pytest
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, make_engine
from app.seed import seed_database


@pytest.fixture
def engine():
    """Fresh in-memory SQLite engine (one shared connection) with all tables created."""
    import app.models  # noqa: F401  (registers tables on Base)

    eng = make_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session_factory(engine):
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@pytest.fixture
def db_session(session_factory):
    with session_factory() as session:
        yield session


@pytest.fixture
def seeded_session(db_session: Session):
    seed_database(db_session)
    return db_session
