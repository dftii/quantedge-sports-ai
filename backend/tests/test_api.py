import csv
import io
import math

import pytest
from fastapi.testclient import TestClient

from app.database import get_db
from app.main import app
from app.seed import seed_database

ODDS_1X2 = {"home": 2.10, "draw": 3.40, "away": 3.60}


@pytest.fixture
def client(session_factory):
    """TestClient on an in-memory seeded DB. Used without `with`, so the lifespan never runs
    and the real quantedge.db file is never created."""
    with session_factory() as db:
        seed_database(db)

    def override():
        with session_factory() as db:
            yield db

    app.dependency_overrides[get_db] = override
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["version"] == "1.0.0"
    assert body["counts"]["teams"] == 20 and body["counts"]["simulations"] == 0


def test_list_and_read_catalogue(client):
    assert len(client.get("/api/teams").json()) == 20
    assert len(client.get("/api/competitions").json()) == 2
    assert client.get("/api/teams/1").json()["id"] == 1
    fighters = client.get("/api/fighters").json()
    wc = fighters[0]["weight_class"]
    assert all(f["weight_class"] == wc for f in client.get("/api/fighters", params={"weight_class": wc}).json())
    page = client.get("/api/events", params={"sport": "football", "status": "finished", "limit": 5}).json()
    assert len(page) == 5 and all(e["sport"] == "football" for e in page)


def test_team_create_patch_and_conflicts(client):
    base = client.get("/api/teams/1").json()
    new = {k: v for k, v in base.items() if k not in ("id", "updated_at")} | {"name": "Test FC", "short_name": "TST"}
    created = client.post("/api/teams", json=new)
    assert created.status_code == 201
    team_id = created.json()["id"]
    assert client.patch(f"/api/teams/{team_id}", json={"elo": 1600}).json()["elo"] == 1600
    assert client.post("/api/teams", json=new).status_code == 409  # duplicate name
    assert client.post("/api/teams", json=new | {"name": "Ghost FC", "competition_id": 999}).status_code == 404
    assert client.patch("/api/teams/9999", json={"elo": 1}).status_code == 404


def test_football_simulation_value_bets_and_saved_roundtrip(client):
    body = {"home_team_id": 1, "away_team_id": 2, "n_simulations": 5000, "seed": 7, "odds": ODDS_1X2}
    data = client.post("/api/simulate/football", json=body).json()
    p = data["prediction"]["markets"]
    assert math.isclose(p["home_win"] + p["draw"] + p["away_win"], 1.0, abs_tol=1e-9)
    s = data["simulation"]
    assert math.isclose(s["home_win"] + s["draw"] + s["away_win"], 1.0, abs_tol=1e-9)
    assert {b["selection"] for b in data["value_bets"]} == {"home", "draw", "away"}
    assert data["market_margin"] == pytest.approx(1 / 2.10 + 1 / 3.40 + 1 / 3.60 - 1)

    sim_id = data["simulation_id"]
    saved = client.get(f"/api/simulations/{sim_id}").json()
    assert saved["sport"] == "football" and saved["home_team_id"] == 1 and saved["seed"] == 7
    assert saved["input_params"]["odds"]["home"] == 2.10 and saved["value_bets"]
    assert [x["id"] for x in client.get("/api/simulations", params={"sport": "football"}).json()] == [sim_id]

    assert client.delete(f"/api/simulations/{sim_id}").status_code == 204
    assert client.get(f"/api/simulations/{sim_id}").status_code == 404
    assert client.delete(f"/api/simulations/{sim_id}").status_code == 404


def assert_value_bets_exact(bets, bankroll, kelly_multiplier):
    """EV, Kelly (capped at 10 %) and stake must follow the closed forms for every selection."""
    for b in bets:
        p, o = b["model_prob"], b["decimal_odds"]
        assert b["expected_value"] == pytest.approx(p * o - 1, abs=1e-9)
        assert b["kelly_fraction"] == pytest.approx(min(0.10, max(0.0, kelly_multiplier * (p * o - 1) / (o - 1))), abs=1e-9)
        assert b["stake"] == pytest.approx(bankroll * b["kelly_fraction"], abs=1e-9)
        assert b["is_value"] == (b["expected_value"] > 0)


@pytest.mark.parametrize("odds", [ODDS_1X2, {"home": 4.0, "draw": 6.0, "away": 9.0}])
def test_football_value_bets_follow_closed_forms(client, odds):
    body = {"home_team_id": 1, "away_team_id": 2, "n_simulations": 1000, "seed": 7, "odds": odds,
            "bankroll": 2500.0, "kelly_multiplier": 0.5, "save": False}
    bets = client.post("/api/simulate/football", json=body).json()["value_bets"]
    assert len(bets) == 3
    assert_value_bets_exact(bets, 2500.0, 0.5)
    assert any(b["kelly_fraction"] > 0 for b in bets) or sum(1 / o for o in odds.values()) > 1  # negative margin => some edge


@pytest.mark.parametrize("odds", [{"fighter_a": 1.80, "fighter_b": 2.10}, {"fighter_a": 3.0, "fighter_b": 3.0}])
def test_fight_value_bets_follow_closed_forms(client, odds):
    body = {"fighter_a_id": 1, "fighter_b_id": 2, "n_simulations": 1000, "seed": 3, "odds": odds,
            "bankroll": 800.0, "kelly_multiplier": 0.3, "save": False}
    bets = client.post("/api/simulate/fight", json=body).json()["value_bets"]
    assert len(bets) == 2
    assert_value_bets_exact(bets, 800.0, 0.3)
    assert any(b["kelly_fraction"] > 0 for b in bets) or sum(1 / o for o in odds.values()) > 1


def test_football_simulation_without_odds_or_save(client):
    data = client.post("/api/simulate/football", json={"home_team_id": 3, "away_team_id": 4, "save": False}).json()
    assert data["value_bets"] is None and data["market_margin"] is None and data["simulation_id"] is None
    assert client.get("/api/simulations").json() == []


def test_football_simulation_dixon_coles_fit(client):
    body = {"home_team_id": 1, "away_team_id": 2, "n_simulations": 2000, "seed": 1,
            "use_dixon_coles_fit": True, "lam3": 0.05, "max_goals": 8, "odds": ODDS_1X2, "save": False}
    resp = client.post("/api/simulate/football", json=body)
    assert resp.status_code == 200
    pred = resp.json()["prediction"]
    assert len(pred["matrix"]) == 9 and pred["lam3"] > 0
    assert pred["markets"]["home_win"] + pred["markets"]["draw"] + pred["markets"]["away_win"] == pytest.approx(1.0)


def test_simulation_same_team_and_unknown_ids(client):
    assert client.post("/api/simulate/football", json={"home_team_id": 1, "away_team_id": 1}).status_code == 422
    assert client.post("/api/simulate/football", json={"home_team_id": 1, "away_team_id": 999}).status_code == 404
    assert client.post("/api/simulate/fight", json={"fighter_a_id": 1, "fighter_b_id": 1}).status_code == 422
    assert client.get("/api/teams/9999").status_code == 404
    assert client.get("/api/fighters/9999").status_code == 404


@pytest.mark.parametrize("rounds", [3, 5])
def test_fight_simulation(client, rounds):
    odds = {"fighter_a": 1.80, "fighter_b": 2.10}
    body = {"fighter_a_id": 1, "fighter_b_id": 2, "rounds": rounds, "n_simulations": 5000, "seed": 3, "odds": odds}
    data = client.post("/api/simulate/fight", json=body).json()
    pred = data["prediction"]
    assert pred["win_prob_a"] + pred["win_prob_b"] == pytest.approx(1.0)
    assert data["simulation"]["win_prob_a"] + data["simulation"]["win_prob_b"] == pytest.approx(1.0)
    assert len(data["value_bets"]) == 2 and data["market_margin"] > 0
    saved = client.get(f"/api/simulations/{data['simulation_id']}").json()
    assert saved["sport"] == "mma" and saved["fighter_a_id"] == 1


def test_h2h_football(client):
    data = client.get("/api/h2h/football", params={"team_a_id": 1, "team_b_id": 2}).json()
    assert data["team_a"]["id"] == 1 and data["team_b"]["id"] == 2
    assert data["differentials"]["elo"] == pytest.approx(data["team_a"]["elo"] - data["team_b"]["elo"])
    s = data["summary"]
    assert s["matches"] == len(data["meetings"]) >= 1
    assert s["team_a_wins"] + s["draws"] + s["team_b_wins"] == s["matches"]
    assert data["meetings"][0]["date"] >= data["meetings"][-1]["date"]
    assert set(data["prediction"]) == {"team_a_home", "team_b_home"}
    assert client.get("/api/h2h/football", params={"team_a_id": 1, "team_b_id": 1}).status_code == 422
    assert client.get("/api/h2h/football", params={"team_a_id": 1, "team_b_id": 999}).status_code == 404


def test_h2h_mma(client):
    data = client.get("/api/h2h/mma", params={"fighter_a_id": 1, "fighter_b_id": 2}).json()
    assert data["fighter_a"]["id"] == 1 and set(data["styles"]) == {"fighter_a", "fighter_b"}
    assert data["differentials"]["reach_cm"] == pytest.approx(
        data["fighter_a"]["reach_cm"] - data["fighter_b"]["reach_cm"]
    )
    assert isinstance(data["previous_fights"], list)
    assert set(data["prediction"]) == {"three_rounds", "five_rounds"}
    assert client.get("/api/h2h/mma", params={"fighter_a_id": 2, "fighter_b_id": 2}).status_code == 422


def test_backtest_football(client):
    finished = len(client.get("/api/events", params={"sport": "football", "status": "finished", "limit": 500}).json())
    report = client.post("/api/backtest/football", json={"warmup": 40}).json()
    assert report["n_events"] == finished - 40 == 150 - 40 == len(report["predictions"])
    assert report["n_matches"] == finished and report["warmup"] == 40
    assert math.isfinite(report["brier_score"]) and math.isfinite(report["log_loss"])
    assert isinstance(report["predictions"][0]["date"], str)


def test_backtest_not_enough_matches(client):
    assert client.post("/api/backtest/football", json={"warmup": 5000}).status_code == 422


def test_export_teams_csv(client):
    resp = client.get("/api/export/teams", params={"format": "csv"})
    assert resp.headers["content-type"].startswith("text/csv")
    assert 'filename="quantedge_teams.csv"' in resp.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(resp.text)))
    assert rows[0][:2] == ["name", "short_name"] and len(rows) == 21


def test_export_events_json_and_simulations_csv(client):
    events = client.get("/api/export/events", params={"format": "json"}).json()
    assert len(events) == 197 and {"id", "sport", "event_date"} <= set(events[0])
    client.post("/api/simulate/football", json={"home_team_id": 1, "away_team_id": 2, "n_simulations": 100})
    rows = list(csv.DictReader(io.StringIO(client.get("/api/export/simulations").text)))
    assert len(rows) == 1 and rows[0]["input_params"].startswith("{")


def test_export_unknown_dataset(client):
    assert client.get("/api/export/nope").status_code == 422


def test_admin_seed(client):
    assert client.post("/api/admin/seed").json()["seeded"] is False
    assert client.post("/api/admin/seed", params={"force": True}).json()["seeded"] is True
    assert len(client.get("/api/teams").json()) == 20


def test_defensive_rating_is_not_double_counted(client):
    """defensive_rating is derived from xG conceded, so it must not move the expected goals again."""
    body = {"home_team_id": 1, "away_team_id": 2, "n_simulations": 100, "seed": 1, "save": False}
    before = client.post("/api/simulate/football", json=body).json()["prediction"]
    assert client.patch("/api/teams/2", json={"defensive_rating": 0.3}).status_code == 200
    after = client.post("/api/simulate/football", json=body).json()["prediction"]
    assert after["lambda_home"] == before["lambda_home"]


def test_dixon_coles_fit_is_cached_until_data_changes(client):
    from app.main import fit_dixon_coles

    fit_dixon_coles.cache_clear()
    body = {"home_team_id": 1, "away_team_id": 2, "n_simulations": 100, "seed": 1, "save": False,
            "use_dixon_coles_fit": True}
    first = client.post("/api/simulate/football", json=body).json()["prediction"]
    second = client.post("/api/simulate/football", json=body).json()["prediction"]
    assert first["lambda_home"] == second["lambda_home"]
    assert fit_dixon_coles.cache_info().hits == 1 and fit_dixon_coles.cache_info().misses == 1

    event = {"sport": "football", "competition_id": 1, "event_date": "2026-03-01", "status": "finished",
             "home_team_id": 1, "away_team_id": 2, "home_goals": 0, "away_goals": 5}
    assert client.post("/api/events", json=event).status_code == 201
    third = client.post("/api/simulate/football", json=body).json()["prediction"]
    assert fit_dixon_coles.cache_info().misses == 2
    assert third["lambda_home"] != first["lambda_home"]


def _home_win(client, **body):
    base = {"home_team_id": 1, "away_team_id": 2, "n_simulations": 100, "seed": 1, "save": False}
    resp = client.post("/api/simulate/football", json=base | body)
    assert resp.status_code == 200
    return resp.json()["prediction"]["markets"]["home_win"]


@pytest.mark.parametrize("fit", [False, True])
def test_football_adjustments_move_home_win(client, fit):
    neutral = _home_win(client, use_dixon_coles_fit=fit)
    assert _home_win(client, use_dixon_coles_fit=fit, adjustments={}) == neutral
    assert _home_win(client, use_dixon_coles_fit=fit, adjustments={"home_absences": 3}) < neutral
    assert _home_win(client, use_dixon_coles_fit=fit, adjustments={"home_advantage": 1.3}) > neutral
    assert _home_win(client, use_dixon_coles_fit=fit, adjustments={"home_form": 0.2}) > neutral


def test_football_adjustments_are_saved(client):
    adj = {"home_advantage": 1.2, "home_form": 0.1, "away_form": -0.1, "home_absences": 2, "away_absences": 1}
    sim_id = client.post("/api/simulate/football", json={
        "home_team_id": 1, "away_team_id": 2, "n_simulations": 100, "adjustments": adj}).json()["simulation_id"]
    assert client.get(f"/api/simulations/{sim_id}").json()["input_params"]["adjustments"] == adj


def test_fight_form_adjustment(client):
    def win_a(**adj):
        body = {"fighter_a_id": 1, "fighter_b_id": 2, "n_simulations": 100, "save": False, "adjustments": adj}
        return client.post("/api/simulate/fight", json=body).json()["prediction"]["win_prob_a"]

    assert win_a(form_a=0.2) > win_a() > win_a(form_b=0.2)


@pytest.mark.parametrize("path, body", [
    ("football", {"home_team_id": 1, "away_team_id": 2, "adjustments": {"home_advantage": 1.5}}),
    ("football", {"home_team_id": 1, "away_team_id": 2, "adjustments": {"home_absences": 6}}),
    ("football", {"home_team_id": 1, "away_team_id": 2, "adjustments": {"away_form": -0.3}}),
    ("fight", {"fighter_a_id": 1, "fighter_b_id": 2, "adjustments": {"form_a": 0.25}}),
])
def test_adjustments_out_of_range_422(client, path, body):
    assert client.post(f"/api/simulate/{path}", json=body).status_code == 422
