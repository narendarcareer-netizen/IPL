import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { getExplanations, getPrediction, listPredictions } from "../lib/api";
import { PredictionInsightPanel } from "../components/PredictionCard";

type PredictionRow = {
  prediction_id: number;
  match_id: number;
  stage: string;
  model_name: string;
  team1_win_prob: number;
  team2_win_prob: number;
  confidence_score: number;
  created_at_utc: string;
  team1_name: string;
  team2_name: string;
  start_time_utc: string;
};

function confidenceLabel(score: number) {
  if (score >= 0.3) return "High";
  if (score >= 0.15) return "Medium";
  return "Low";
}

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

export default function PredictionsPage() {
  const [items, setItems] = useState<PredictionRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [teamFilter, setTeamFilter] = useState("");
  const [detailsByMatch, setDetailsByMatch] = useState<Record<number, any>>({});
  const [loadingDetailMatch, setLoadingDetailMatch] = useState<number | null>(null);

  useEffect(() => {
    (async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await listPredictions(50);
        setItems(data.items || []);
      } catch (err: any) {
        setError(err?.message || "Failed to load predictions");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const filtered = useMemo(() => {
    const q = teamFilter.trim().toLowerCase();
    if (!q) return items;
    return items.filter(
      (x) =>
        x.team1_name.toLowerCase().includes(q) ||
        x.team2_name.toLowerCase().includes(q)
    );
  }, [items, teamFilter]);

  async function loadDetails(matchId: number, stage: string) {
    try {
      setLoadingDetailMatch(matchId);
      const [prediction, explanations] = await Promise.all([
        getPrediction(matchId, stage || "pre_toss"),
        getExplanations(matchId, stage || "pre_toss"),
      ]);
      setDetailsByMatch((current) => ({
        ...current,
        [matchId]: {
          prediction,
          explanations: explanations.items || [],
        },
      }));
    } catch (err) {
      console.error("Failed to load prediction details", err);
      alert("Could not load explanation details for this match.");
    } finally {
      setLoadingDetailMatch(null);
    }
  }

  return (
    <main style={{ maxWidth: 1100, margin: "0 auto", padding: 24 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 20,
          gap: 12,
          flexWrap: "wrap",
        }}
      >
        <div>
          <h1 style={{ margin: 0 }}>IPL Prediction Dashboard</h1>
          <p style={{ marginTop: 8, color: "#555" }}>
            Saved predictions with batting, bowling, venue, and player-form explanations
          </p>
        </div>

        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <Link href="/" style={{ textDecoration: "none" }}>
            Home
          </Link>
          <input
            value={teamFilter}
            onChange={(e) => setTeamFilter(e.target.value)}
            placeholder="Filter by team..."
            style={{
              padding: "10px 12px",
              border: "1px solid #ccc",
              borderRadius: 8,
              minWidth: 220,
            }}
          />
        </div>
      </div>

      {loading && <p>Loading predictions...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {!loading && !error && filtered.length === 0 && <p>No predictions found.</p>}

      <div style={{ display: "grid", gap: 16 }}>
        {filtered.map((item) => {
          const favorite =
            item.team1_win_prob >= item.team2_win_prob
              ? item.team1_name
              : item.team2_name;

          return (
            <section
              key={item.prediction_id}
              style={{
                border: "1px solid #ddd",
                borderRadius: 12,
                padding: 18,
                boxShadow: "0 2px 8px rgba(0,0,0,0.04)",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  gap: 16,
                  flexWrap: "wrap",
                }}
              >
                <div>
                  <h2 style={{ margin: "0 0 8px 0", fontSize: 20 }}>
                    {item.team1_name} vs {item.team2_name}
                  </h2>
                  <div style={{ color: "#666", fontSize: 14 }}>
                    Match #{item.match_id} | {item.stage} | {item.model_name}
                  </div>
                  <div style={{ color: "#666", fontSize: 14, marginTop: 4 }}>
                    Starts: {item.start_time_utc}
                  </div>
                </div>

                <div style={{ textAlign: "right" }}>
                  <div style={{ fontWeight: 600 }}>Favorite: {favorite}</div>
                  <div style={{ marginTop: 6 }}>
                    Confidence:{" "}
                    <span
                      style={{
                        padding: "4px 8px",
                        borderRadius: 999,
                        background: "#f3f4f6",
                        fontSize: 13,
                      }}
                    >
                      {confidenceLabel(item.confidence_score)} ({pct(item.confidence_score)})
                    </span>
                  </div>
                  <button
                    onClick={() => loadDetails(item.match_id, item.stage)}
                    disabled={loadingDetailMatch === item.match_id}
                    style={{
                      marginTop: 10,
                      border: "1px solid #d1d5db",
                      borderRadius: 8,
                      background: "#fff",
                      padding: "8px 12px",
                      cursor: "pointer",
                    }}
                  >
                    {loadingDetailMatch === item.match_id ? "Loading..." : "Show explanation"}
                  </button>
                </div>
              </div>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: 16,
                  marginTop: 16,
                }}
              >
                <div
                  style={{
                    border: "1px solid #eee",
                    borderRadius: 10,
                    padding: 14,
                  }}
                >
                  <div style={{ fontWeight: 600, marginBottom: 8 }}>
                    {item.team1_name}
                  </div>
                  <div style={{ fontSize: 28, fontWeight: 700 }}>
                    {pct(item.team1_win_prob)}
                  </div>
                  <div
                    style={{
                      height: 10,
                      background: "#eee",
                      borderRadius: 999,
                      overflow: "hidden",
                      marginTop: 10,
                    }}
                  >
                    <div
                      style={{
                        width: `${item.team1_win_prob * 100}%`,
                        height: "100%",
                        background: "#2563eb",
                      }}
                    />
                  </div>
                </div>

                <div
                  style={{
                    border: "1px solid #eee",
                    borderRadius: 10,
                    padding: 14,
                  }}
                >
                  <div style={{ fontWeight: 600, marginBottom: 8 }}>
                    {item.team2_name}
                  </div>
                  <div style={{ fontSize: 28, fontWeight: 700 }}>
                    {pct(item.team2_win_prob)}
                  </div>
                  <div
                    style={{
                      height: 10,
                      background: "#eee",
                      borderRadius: 999,
                      overflow: "hidden",
                      marginTop: 10,
                    }}
                  >
                    <div
                      style={{
                        width: `${item.team2_win_prob * 100}%`,
                        height: "100%",
                        background: "#16a34a",
                      }}
                    />
                  </div>
                </div>
              </div>

              <div
                style={{
                  marginTop: 14,
                  fontSize: 13,
                  color: "#666",
                  display: "flex",
                  justifyContent: "space-between",
                  flexWrap: "wrap",
                  gap: 8,
                }}
              >
                <span>Prediction created: {item.created_at_utc}</span>
                <span>ID: {item.prediction_id}</span>
              </div>

              {detailsByMatch[item.match_id] && (
                <PredictionInsightPanel
                  prediction={detailsByMatch[item.match_id].prediction}
                  explanations={detailsByMatch[item.match_id].explanations}
                />
              )}
            </section>
          );
        })}
      </div>
    </main>
  );
}
