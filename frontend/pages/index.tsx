import { useEffect, useState } from "react";
import Link from "next/link";
import { getUpcomingMatches, getPrediction, getExplanations } from "../lib/api";

export default function Home() {
  const [matches, setMatches] = useState<any[]>([]);
  const [preds, setPreds] = useState<Record<string, any>>({});
  const [loadingMatches, setLoadingMatches] = useState(true);
  const [matchesError, setMatchesError] = useState<string | null>(null);
  const [loadingPredFor, setLoadingPredFor] = useState<number | null>(null);

  const modelUriPre = "models:/ipl_winprob_xgb/Staging";

  useEffect(() => {
    (async () => {
      try {
        setLoadingMatches(true);
        setMatchesError(null);
        const data = await getUpcomingMatches();
        setMatches(data.items || []);
      } catch (error: any) {
        setMatchesError(error?.message || "Failed to load matches");
      } finally {
        setLoadingMatches(false);
      }
    })();
  }, []);

  async function loadPred(matchId: number) {
    try {
      setLoadingPredFor(matchId);

      const pre = await getPrediction(matchId, "pre_toss", modelUriPre);
      const post = await getPrediction(matchId, "post_toss", modelUriPre);
      const explanations = await getExplanations(matchId);

      setPreds((p) => ({
        ...p,
        [matchId]: {
          pre,
          post,
          explanations: explanations.items || [],
        },
      }));
    } catch (error) {
      console.error("Prediction failed for match:", matchId, error);
      alert("Prediction/explanations are not ready yet.");
    } finally {
      setLoadingPredFor(null);
    }
  }

  return (
    <main style={{ padding: 24 }}>
      <h1>IPL Local Predictor</h1>

      <div style={{ marginBottom: 16 }}>
        <Link href="/predictions">Open Prediction Dashboard</Link>
      </div>

      <p>Recent IPL matches</p>

      {loadingMatches && <p>Loading matches...</p>}
      {matchesError && <p style={{ color: "red" }}>{matchesError}</p>}

      {!loadingMatches && !matchesError && matches.length === 0 && (
        <p>No matches found.</p>
      )}

      <ul style={{ paddingLeft: 0, listStyle: "none" }}>
        {matches.map((m) => (
          <li
            key={m.match_id}
            style={{
              marginBottom: 16,
              padding: 16,
              border: "1px solid #ddd",
              borderRadius: 8,
            }}
          >
            <div style={{ fontWeight: 600 }}>
              {m.team1_name} vs {m.team2_name}
            </div>

            <div>Match #{m.match_id}</div>
            <div>Competition: {m.competition || "Indian Premier League"}</div>
            <div>Starts: {m.start_time_utc}</div>

            <button
              onClick={() => loadPred(m.match_id)}
              disabled={loadingPredFor === m.match_id}
              style={{ marginTop: 8 }}
            >
              {loadingPredFor === m.match_id ? "Loading..." : "Predict"}
            </button>

            {preds[m.match_id] && (
              <div style={{ marginTop: 12 }}>
                <div>Model: {preds[m.match_id].pre.model_name}</div>
                <div>
                  {preds[m.match_id].pre.team1_name} win prob:{" "}
                  {Number(preds[m.match_id].pre.team1_win_prob).toFixed(3)}
                </div>
                <div>
                  {preds[m.match_id].pre.team2_name} win prob:{" "}
                  {Number(preds[m.match_id].pre.team2_win_prob).toFixed(3)}
                </div>
                <div>
                  {preds[m.match_id].pre.team1_name} Elo:{" "}
                  {Number(preds[m.match_id].pre.team1_rating).toFixed(1)}
                </div>
                <div>
                  {preds[m.match_id].pre.team2_name} Elo:{" "}
                  {Number(preds[m.match_id].pre.team2_rating).toFixed(1)}
                </div>
                <div>
                  Confidence: {Number(preds[m.match_id].pre.confidence_score).toFixed(3)}
                </div>

                <div style={{ marginTop: 10 }}>
                  <strong>Why this prediction?</strong>
                  {preds[m.match_id].explanations?.length ? (
                    <div style={{ marginTop: 6 }}>
                      {preds[m.match_id].explanations.map((e: any) => (
                        <div key={e.rank}>
                          {e.rank}. {e.feature_name}: {Number(e.impact_value).toFixed(3)}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ marginTop: 6 }}>No explanation available.</div>
                  )}
                </div>
              </div>
            )}
          </li>
        ))}
      </ul>
    </main>
  );
}