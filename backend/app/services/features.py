from dataclasses import dataclass
from datetime import datetime
from sqlalchemy.orm import Session

@dataclass
class FeatureBundle:
    features: dict
    stage: str
    cutoff_time_utc: datetime

def build_features(db: Session, match_id: int, stage: str, cutoff_time_utc: datetime) -> FeatureBundle:
    # TODO: join matches + team_form_snapshots + player_form_snapshots + odds_snapshots
    # Enforce leakage: only use rows with as_of_time_utc <= cutoff_time_utc.
    feats = {
        "stage": stage,
        "has_toss_info": stage in ("post_toss", "post_lineup"),
        # Placeholders:
        "elo_diff": 0.0,
        "probable_xi_strength_diff": 0.0,
        "book_implied_prob_team1": None,
        "line_move_team1": None,
    }
    return FeatureBundle(features=feats, stage=stage, cutoff_time_utc=cutoff_time_utc)
