export const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export async function getUpcomingMatches() {
  const res = await fetch(`${BACKEND_URL}/matches/upcoming`);
  if (!res.ok) {
    throw new Error(`Failed to fetch upcoming matches: ${res.status}`);
  }
  return res.json();
}

export async function getPrediction(matchId: number, stage: string, modelUri?: string) {
  const url = new URL(`${BACKEND_URL}/predictions/match/${matchId}`);
  url.searchParams.set("stage", stage);
  if (modelUri) {
    url.searchParams.set("model_uri", modelUri);
  }

  const res = await fetch(url.toString());
  if (!res.ok) {
    throw new Error(`Failed to fetch prediction: ${res.status}`);
  }
  return res.json();
}

export async function getExplanations(matchId: number, stage?: string) {
  const url = new URL(`${BACKEND_URL}/predictions/explanations/${matchId}`);
  if (stage) {
    url.searchParams.set("stage", stage);
  }
  const res = await fetch(url.toString());
  if (!res.ok) {
    throw new Error(`Failed to fetch explanations: ${res.status}`);
  }
  return res.json();
}

export async function listPredictions(limit = 20) {
  const url = new URL(`${BACKEND_URL}/predictions`);
  url.searchParams.set("limit", String(limit));

  const res = await fetch(url.toString());
  if (!res.ok) {
    throw new Error(`Failed to fetch predictions: ${res.status}`);
  }
  return res.json();
}
