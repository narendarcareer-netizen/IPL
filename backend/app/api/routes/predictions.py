from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.services.predict import predict_match

router = APIRouter()

@router.get("/match/{match_id}")
def predict_for_match(
    match_id: int,
    stage: str = Query("pre_toss", pattern="^(pre_toss|post_toss|post_lineup)$"),
    model_uri: str | None = Query(None),
    db: Session = Depends(get_db),
):
    cutoff = datetime.now(timezone.utc)
    try:
        return predict_match(
            db=db,
            match_id=match_id,
            stage=stage,
            cutoff_time_utc=cutoff,
            model_uri=model_uri,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction unavailable: {str(e)}")

@router.get("")
def list_predictions(
    limit: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db),
):
    rows = db.execute(
        text("""
            SELECT
                p.prediction_id,
                p.match_id,
                p.stage,
                p.model_name,
                p.team1_win_prob,
                p.team2_win_prob,
                p.confidence_score,
                p.created_at_utc,
                t1.name AS team1_name,
                t2.name AS team2_name,
                m.start_time_utc
            FROM predictions p
            JOIN matches m
              ON p.match_id = m.match_id
            JOIN teams t1
              ON m.team1_id = t1.team_id
            JOIN teams t2
              ON m.team2_id = t2.team_id
            ORDER BY p.created_at_utc DESC, p.prediction_id DESC
            LIMIT :limit
        """),
        {"limit": limit},
    ).mappings().all()

    return {"items": list(rows)}
 
@router.get("/explanations/{match_id}")
def get_explanations(match_id: int, db: Session = Depends(get_db)):
    row = db.execute(
        text("""
            SELECT e.top_features_json
            FROM explanations e
            JOIN predictions p
              ON e.prediction_id = p.prediction_id
            WHERE p.match_id = :match_id
            ORDER BY p.created_at_utc DESC, e.explanation_id DESC
            LIMIT 1
        """),
        {"match_id": match_id},
    ).first()

    if not row:
        return {"items": []}

    return {"items": row._mapping["top_features_json"] or []}
