type ExplanationItem = {
  feature_name: string;
  impact_value: number;
  rank?: number;
};

type PlayerCard = {
  player_id: number;
  player_name: string;
  role: string;
  power: number;
  summary: string;
  tags?: string[];
};

type TeamBreakdown = {
  team_name: string;
  batting_power: number;
  bowling_power: number;
  top_batters: PlayerCard[];
  top_bowlers: PlayerCard[];
};

type KeyFactor = {
  id: string;
  title: string;
  edge: number;
  favored_team: string;
  strength: string;
  summary: string;
  bullets: string[];
};

type PredictionPayload = {
  team1_name: string;
  team2_name: string;
  team1_win_prob: number;
  team2_win_prob: number;
  confidence_score: number;
  model_name?: string;
  insights?: {
    overview?: string;
    source_note?: string;
    key_factors?: KeyFactor[];
    team_breakdown?: TeamBreakdown[];
    venue?: {
      venue_name?: string;
      venue_city?: string;
      pitch_type?: string;
    };
  };
};

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

function confidenceLabel(score: number) {
  if (score >= 0.3) return "High";
  if (score >= 0.15) return "Medium";
  return "Low";
}

function roleLabel(role: string) {
  const mapping: Record<string, string> = {
    wk_batter: "WK",
    batter: "Bat",
    bowler: "Bowl",
    all_rounder: "AR",
  };
  return mapping[role] || role;
}

function prettyFeatureName(featureName: string) {
  const mapping: Record<string, string> = {
    probable_xi_batting_form_diff: "Expected XI batting form",
    probable_xi_bowling_form_diff: "Expected XI bowling form",
    probable_xi_runs_avg_diff: "Expected XI runs base",
    probable_xi_strike_rate_diff: "Expected XI strike rate",
    batting_vs_bowling_form_diff: "Batting vs bowling matchup",
    top_order_vs_powerplay_diff: "Top order vs powerplay",
    death_matchup_diff: "Death overs matchup",
    bookmaker_prob_diff: "Bookmaker lean",
    elo_diff: "Elo edge",
    venue_win_bias_diff: "Venue bias",
  };
  return mapping[featureName] || featureName.replace(/_/g, " ");
}

function factorAccent(edge: number) {
  if (edge > 0.08) return "#2563eb";
  if (edge < -0.08) return "#16a34a";
  return "#6b7280";
}

export function PredictionInsightPanel({
  prediction,
  explanations = [],
}: {
  prediction: PredictionPayload;
  explanations?: ExplanationItem[];
}) {
  const insights = prediction.insights;
  const venueLabel =
    insights?.venue?.venue_name || insights?.venue?.venue_city || "Venue not available";

  return (
    <div style={{ marginTop: 16, display: "grid", gap: 16 }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 16,
        }}
      >
        <div
          style={{
            border: "1px solid #e5e7eb",
            borderRadius: 12,
            padding: 14,
            background: "#f8fbff",
          }}
        >
          <div style={{ fontWeight: 700 }}>{prediction.team1_name}</div>
          <div style={{ fontSize: 30, fontWeight: 800, marginTop: 6 }}>
            {pct(prediction.team1_win_prob)}
          </div>
          <div style={{ marginTop: 8, color: "#4b5563" }}>
            Confidence: {confidenceLabel(prediction.confidence_score)} ({pct(prediction.confidence_score)})
          </div>
        </div>

        <div
          style={{
            border: "1px solid #e5e7eb",
            borderRadius: 12,
            padding: 14,
            background: "#f7fcf7",
          }}
        >
          <div style={{ fontWeight: 700 }}>{prediction.team2_name}</div>
          <div style={{ fontSize: 30, fontWeight: 800, marginTop: 6 }}>
            {pct(prediction.team2_win_prob)}
          </div>
          <div style={{ marginTop: 8, color: "#4b5563" }}>
            Model: {prediction.model_name || "local model"}
          </div>
        </div>
      </div>

      {insights?.overview && (
        <section
          style={{
            border: "1px solid #e5e7eb",
            borderRadius: 12,
            padding: 16,
            background: "#fffdf7",
          }}
        >
          <div style={{ fontWeight: 700, marginBottom: 8 }}>Why the model leans this way</div>
          <div style={{ color: "#374151", lineHeight: 1.5 }}>{insights.overview}</div>
        </section>
      )}

      {insights?.key_factors?.length ? (
        <section style={{ display: "grid", gap: 12 }}>
          <div style={{ fontWeight: 700 }}>Key Match Factors</div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
              gap: 12,
            }}
          >
            {insights.key_factors.map((factor) => (
              <div
                key={factor.id}
                style={{
                  border: "1px solid #e5e7eb",
                  borderRadius: 12,
                  padding: 14,
                  background: "#fff",
                }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", gap: 10 }}>
                  <div style={{ fontWeight: 700 }}>{factor.title}</div>
                  <div
                    style={{
                      color: factorAccent(factor.edge),
                      fontWeight: 700,
                      fontSize: 13,
                    }}
                  >
                    {factor.favored_team} | {factor.strength}
                  </div>
                </div>
                <div style={{ marginTop: 8, color: "#4b5563", lineHeight: 1.45 }}>
                  {factor.summary}
                </div>
                <div style={{ marginTop: 10, display: "grid", gap: 6, fontSize: 13, color: "#374151" }}>
                  {factor.bullets.map((bullet) => (
                    <div key={bullet}>- {bullet}</div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      {insights?.team_breakdown?.length ? (
        <section style={{ display: "grid", gap: 12 }}>
          <div style={{ fontWeight: 700 }}>Expected XI Power</div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
              gap: 12,
            }}
          >
            {insights.team_breakdown.map((team) => (
              <div
                key={team.team_name}
                style={{
                  border: "1px solid #e5e7eb",
                  borderRadius: 12,
                  padding: 14,
                  background: "#fff",
                }}
              >
                <div style={{ fontWeight: 700, fontSize: 18 }}>{team.team_name}</div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: 10,
                    marginTop: 12,
                  }}
                >
                  <div
                    style={{
                      padding: 10,
                      borderRadius: 10,
                      background: "#eff6ff",
                    }}
                  >
                    <div style={{ fontSize: 12, color: "#4b5563" }}>Batting Power</div>
                    <div style={{ fontSize: 24, fontWeight: 800 }}>{team.batting_power.toFixed(1)}</div>
                  </div>
                  <div
                    style={{
                      padding: 10,
                      borderRadius: 10,
                      background: "#f0fdf4",
                    }}
                  >
                    <div style={{ fontSize: 12, color: "#4b5563" }}>Bowling Power</div>
                    <div style={{ fontSize: 24, fontWeight: 800 }}>{team.bowling_power.toFixed(1)}</div>
                  </div>
                </div>

                <div style={{ marginTop: 14, fontWeight: 700 }}>Top Batters</div>
                <div style={{ display: "grid", gap: 8, marginTop: 8 }}>
                  {team.top_batters.map((player) => (
                    <div
                      key={`bat-${player.player_id}`}
                      style={{
                        border: "1px solid #eef2f7",
                        borderRadius: 10,
                        padding: 10,
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                        <div style={{ fontWeight: 600 }}>
                          {player.player_name}{" "}
                          <span style={{ color: "#6b7280", fontWeight: 500 }}>
                            ({roleLabel(player.role)})
                          </span>
                        </div>
                        <div style={{ fontWeight: 700 }}>{player.power.toFixed(1)}</div>
                      </div>
                      <div style={{ marginTop: 4, color: "#4b5563", fontSize: 13 }}>{player.summary}</div>
                    </div>
                  ))}
                </div>

                <div style={{ marginTop: 14, fontWeight: 700 }}>Top Bowlers</div>
                <div style={{ display: "grid", gap: 8, marginTop: 8 }}>
                  {team.top_bowlers.map((player) => (
                    <div
                      key={`bowl-${player.player_id}`}
                      style={{
                        border: "1px solid #eef2f7",
                        borderRadius: 10,
                        padding: 10,
                      }}
                    >
                      <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                        <div style={{ fontWeight: 600 }}>
                          {player.player_name}{" "}
                          <span style={{ color: "#6b7280", fontWeight: 500 }}>
                            ({roleLabel(player.role)})
                          </span>
                        </div>
                        <div style={{ fontWeight: 700 }}>{player.power.toFixed(1)}</div>
                      </div>
                      <div style={{ marginTop: 4, color: "#4b5563", fontSize: 13 }}>{player.summary}</div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>
      ) : null}

      <section
        style={{
          border: "1px solid #e5e7eb",
          borderRadius: 12,
          padding: 14,
          background: "#fff",
        }}
      >
        <div style={{ fontWeight: 700 }}>Venue Context</div>
        <div style={{ marginTop: 8, color: "#374151" }}>
          {venueLabel}
          {insights?.venue?.pitch_type ? ` | ${insights.venue.pitch_type} leaning` : ""}
        </div>
        {insights?.source_note && (
          <div style={{ marginTop: 10, color: "#6b7280", fontSize: 13, lineHeight: 1.5 }}>
            {insights.source_note}
          </div>
        )}
      </section>

      {explanations.length ? (
        <section
          style={{
            border: "1px solid #e5e7eb",
            borderRadius: 12,
            padding: 14,
            background: "#fff",
          }}
        >
          <div style={{ fontWeight: 700, marginBottom: 8 }}>Top Model Drivers</div>
          <div style={{ display: "grid", gap: 8 }}>
            {explanations.map((item, index) => (
              <div
                key={`${item.feature_name}-${index}`}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  gap: 12,
                  padding: "8px 10px",
                  borderRadius: 8,
                  background: "#f9fafb",
                }}
              >
                <div>{prettyFeatureName(item.feature_name)}</div>
                <div style={{ fontWeight: 700 }}>{Number(item.impact_value).toFixed(3)}</div>
              </div>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}
