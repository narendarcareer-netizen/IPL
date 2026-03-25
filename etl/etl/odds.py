import os

def refresh_odds():
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("refresh_odds skipped: ODDS_API_KEY not configured")
        return
    print("refresh_odds placeholder: implement odds fetch here")