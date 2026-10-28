"""Write the seeded dataset as one JSON file for the static (GitHub Pages) frontend.

Run from backend/:  python -m app.export_dataset ../frontend/src/data/dataset.json
"""
import json
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import init_db, make_engine
from app.models import Competition, Event, Fighter, Team
from app.schemas import CompetitionRead, EventRead, FighterRead, TeamRead
from app.seed import seed_database


def build_dataset() -> dict:
    engine = make_engine("sqlite://", poolclass=StaticPool)
    init_db(engine)
    with sessionmaker(engine)() as db:
        seed_database(db)
        dump = lambda model, schema: [  # noqa: E731
            schema.model_validate(row).model_dump(mode="json") for row in db.scalars(select(model).order_by(model.id))
        ]
        return {
            "competitions": dump(Competition, CompetitionRead),
            "teams": dump(Team, TeamRead),
            "fighters": dump(Fighter, FighterRead),
            "events": dump(Event, EventRead),
        }


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "../frontend/src/data/dataset.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_dataset(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
