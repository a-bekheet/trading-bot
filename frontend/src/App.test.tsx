import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const overview = {
  version: "0.91.0",
  services: [
    { id: "collector", label: "Market collector", status: "complete" },
    { id: "training", label: "Training watcher", status: "waiting" },
    { id: "paper_agents", label: "Paper agents", status: "active" },
  ],
  service_summary: { healthy: 3, total: 3 },
  market: {
    symbol: "AAPL",
    underlying_price: 202,
    session: { provider_state: "REGULAR" },
  },
  agents: { active: 1, total: 5, decisions: 12, executions: 3 },
  account: { cash: 100000 },
  tickers: ["AAPL"],
  jobs: [],
};

const training = {
  defaults: {
    symbols: ["AAPL", "NVDA", "MSFT", "AMZN", "GOOG"],
    episodes: 3,
    hidden_size: 16,
    sequence_length: 4,
    max_steps: 16,
    candidate_count_per_ticker: 12,
    training_seed_count: 3,
  },
  watcher: { status: "waiting" },
  readiness: [],
  jobs: [],
};

describe("Options Control Room", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const path = String(input);
        const body = path.includes("/api/training") ? training : overview;
        return new Response(JSON.stringify(body), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }),
    );
  });

  afterEach(() => vi.unstubAllGlobals());

  it("renders the real command center and first-class training navigation", async () => {
    render(<App />);

    expect(await screen.findByRole("heading", { name: "Command Center" })).toBeVisible();
    const trainingButton = screen.getByRole("button", { name: /Training/ });
    expect(trainingButton).toBeVisible();

    fireEvent.click(trainingButton);

    expect(await screen.findByRole("heading", { name: "Training" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Review and launch training" })).toBeVisible();
    expect(screen.getByText("GRU · LSTM · Mixture · GNN")).toBeVisible();
  });
});
