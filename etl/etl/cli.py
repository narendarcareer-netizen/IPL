import typer
from etl.cricsheet import load_cricsheet
from etl.odds import refresh_odds
from etl.lineups import refresh_lineups

app = typer.Typer(help="ETL commands for local IPL predictor.")

@app.command()
def ingest_cricsheet(path: str):
    load_cricsheet(path)

@app.command()
def refresh_odds_cmd():
    refresh_odds()

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
