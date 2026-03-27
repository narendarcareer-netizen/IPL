import typer
from etl.cricsheet import load_cricsheet
from etl.fixtures import sync_upcoming_fixtures_from_odds
from etl.odds import backfill_historical_odds, refresh_odds
from etl.lineups import refresh_lineups

app = typer.Typer(help="ETL commands for local IPL predictor.")

@app.command()
def ingest_cricsheet(path: str):
    load_cricsheet(path)

@app.command()
def refresh_odds_cmd():
    refresh_odds()

@app.command()
def backfill_historical_odds_cmd(limit: int = 25, hours_before_start: int = 6, season_from: int = 2020):
    backfill_historical_odds(
        limit=limit,
        hours_before_start=hours_before_start,
        season_from=season_from,
    )

@app.command()
def sync_upcoming_fixtures_cmd(delete_stale: bool = True):
    sync_upcoming_fixtures_from_odds(delete_stale=delete_stale)

@app.command()
def refresh_lineups_cmd():
    refresh_lineups()

@app.command()
def schedule():
    # Minimal placeholder: implement a simple loop + sleep here,
    # or replace with a proper scheduler later.
    import time
    while True:
        try:
            refresh_odds()
            refresh_lineups()
        except Exception as e:
            print("scheduler error:", e)
        time.sleep(300)

if __name__ == "__main__":
    app()
