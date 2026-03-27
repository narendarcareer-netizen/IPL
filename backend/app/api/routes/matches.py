from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.session import get_db

router = APIRouter()

@router.get("/upcoming")
def upcoming_matches(db: Session = Depends(get_db)):
    q = text("""
      SELECT
        m.match_id,
        m.season,
        m.competition,
        m.start_time_utc,
        m.venue_id,
        m.toss_winner_team_id,
        m.toss_decision,
        m.completed,
        t1.name AS team1_name,
        t2.name AS team2_name
      FROM matches m
      JOIN teams t1 ON m.team1_id = t1.team_id
      JOIN teams t2 ON m.team2_id = t2.team_id
      WHERE m.competition = 'Indian Premier League'
        AND COALESCE(m.completed, false) = false
        AND m.start_time_utc >= NOW() - interval '1 day'
      ORDER BY m.start_time_utc ASC
      LIMIT 50
    """)
    rows = db.execute(q).mappings().all()
    return {"items": list(rows)}
