import { useEffect, useState } from "react";
import Link from "next/link";
import { getUpcomingMatches, getPrediction, getExplanations } from "../lib/api";
import { PredictionInsightPanel } from "../components/PredictionCard";

export default function Home() {
  const [matches, setMatches] = useState<any[]>([]);
  const [preds, setPreds] = useState<Record<string, any>>({});
  const [stageByMatch, setStageByMatch] = useState<Record<number, string>>({});
  const [loadingMatches, setLoadingMatches] = useState(true);
  const [matchesError, setMatchesError] = useState<string | null>(null);
  const [loadingPredFor, setLoadingPredFor] = useState<number | null>(null);
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
      const stage = stageByMatch[matchId] || "pre_toss";

      const pre = await getPrediction(matchId, stage);
      const explanations = await getExplanations(matchId, stage);

      setPreds((p) => ({
        ...p,
        [matchId]: {
          pre,
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
            <div style={{ marginTop: 8 }}>
              <select
                value={stageByMatch[m.match_id] || "pre_toss"}
                onChange={(e) =>
                  setStageByMatch((current) => ({
                    ...current,
                    [m.match_id]: e.target.value,
                  }))
                }
              >
                <option value="pre_toss">Pre Toss</option>
                <option value="post_toss">Post Toss</option>
                <option value="confirmed_xi">Confirmed XI</option>
              </select>
            </div>

            <button
              onClick={() => loadPred(m.match_id)}
              disabled={loadingPredFor === m.match_id}
              style={{ marginTop: 8 }}
            >
              {loadingPredFor === m.match_id ? "Loading..." : "Predict"}
            </button>

            {preds[m.match_id] && (
              <PredictionInsightPanel
                prediction={preds[m.match_id].pre}
                explanations={preds[m.match_id].explanations}
              />
            )}
          </li>
        ))}
      </ul>
    </main>
  );
}
