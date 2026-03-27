import Link from "next/link";
import { useRouter } from "next/router";
import { useEffect, useState } from "react";
import { PredictionInsightPanel } from "../../components/PredictionCard";
import { getExplanations, getPrediction } from "../../lib/api";

type MatchDetailsState = {
  prediction: any;
  explanations: any[];
};

export default function MatchDetailsPage() {
  const router = useRouter();
  const { matchId } = router.query;
  const [stage, setStage] = useState("pre_toss");
  const [details, setDetails] = useState<MatchDetailsState | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!router.isReady || !matchId) return;

    const numericMatchId = Number(matchId);
    if (!Number.isFinite(numericMatchId)) {
      setError("Invalid match id.");
      return;
    }

    (async () => {
      try {
        setLoading(true);
        setError(null);
        const [prediction, explanations] = await Promise.all([
          getPrediction(numericMatchId, stage),
          getExplanations(numericMatchId, stage),
        ]);
        setDetails({
          prediction,
          explanations: explanations.items || [],
        });
      } catch (err: any) {
        setError(err?.message || "Failed to load match details.");
      } finally {
        setLoading(false);
      }
    })();
  }, [router.isReady, matchId, stage]);

  return (
    <main style={{ maxWidth: 1100, margin: "0 auto", padding: 24 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 12,
          marginBottom: 20,
        }}
      >
        <div>
          <h1 style={{ margin: 0 }}>Match Explanation</h1>
          <p style={{ marginTop: 8, color: "#555" }}>
            Pre-toss prediction with player-form, bowling, batting, venue, and model-driver context
          </p>
          <div style={{ marginTop: 12 }}>
            <select value={stage} onChange={(e) => setStage(e.target.value)}>
              <option value="pre_toss">Pre Toss</option>
              <option value="post_toss">Post Toss</option>
              <option value="confirmed_xi">Confirmed XI</option>
            </select>
          </div>
        </div>
        <Link href="/predictions" style={{ textDecoration: "none" }}>
          Back to dashboard
        </Link>
      </div>

      {loading && <p>Loading match explanation...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {!loading && !error && details && (
        <PredictionInsightPanel
          prediction={details.prediction}
          explanations={details.explanations}
        />
      )}
    </main>
  );
}
