import os
from pathlib import Path

from sqlalchemy import create_engine

from ml.feature_builder import build_historical_feature_frame
from ml.feature_config import STAGES, historical_output_for_stage

DATABASE_URL = os.environ["DATABASE_URL"]


def main():
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    for stage in STAGES:
        df = build_historical_feature_frame(engine, stage=stage)
        output_path = historical_output_for_stage(stage)

        if df.empty:
            if os.path.exists(output_path):
                os.remove(output_path)
            print(f"No completed IPL matches found for historical feature generation at stage={stage}.")
            continue

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"Saved {stage} historical features to {output_path}")
        preview_cols = [
            "match_id",
            "stage",
            "team1_name",
            "team2_name",
            "elo_diff",
            "head_to_head_win_pct_diff",
            "probable_xi_batting_form_diff",
            "bookmaker_prob_team1",
            "team1_won",
        ]
        preview_cols = [column for column in preview_cols if column in df.columns]
        print(df[preview_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
