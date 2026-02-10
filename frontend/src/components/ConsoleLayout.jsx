import { useMemo, useRef, useState } from "react";
import Panel from "./Panel";
import StatusPill from "./StatusPill";
import MetricGrid from "./MetricGrid";
import Sparkline from "./Sparkline";
import MiniBarChart from "./MiniBarChart";
import { runExperimentStream } from "../api";

const formatNumber = (value, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "NA";
  }
  return Number(value).toFixed(digits);
};

const toPercent = (value) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "NA";
  }
  return `${(Number(value) * 100).toFixed(1)}%`;
};

const meanFromEpisodes = (episodes, key) => {
  if (!episodes || episodes.length === 0) return null;
  const values = episodes
    .map((episode) => episode.metrics?.[key])
    .filter((value) => typeof value === "number");
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
};

export default function ConsoleLayout() {
  const MAX_LOGS = 120;
  const [activePanel, setActivePanel] = useState("experiment");
  const [status, setStatus] = useState("idle");
  const [lastSync, setLastSync] = useState("waiting");
  const [logs, setLogs] = useState([
    "> system ready - waiting for command",
  ]);
  const [progressPct, setProgressPct] = useState(null);
  const [form, setForm] = useState({
    scenario: "regime_shift_long",
    agent_type: "ppo",
    reward_mode: "risk_adjusted",
    episodes: 10,
    timesteps: 20000,
    schedule: "",
    schedule_length: 20,
    drawdown_coeff: 0.1,
    volatility_coeff: 0.1,
    trade_penalty_coeff: 0.0,
    invalid_action_penalty: 0.5,
    inactivity_penalty: 0.05,
    trade_size: 50,
  });
  const [lastResult, setLastResult] = useState(null);
  const [prevMetrics, setPrevMetrics] = useState(null);
  const lastProgressRef = useRef(0);

  const appendLog = (message) => {
    setLogs((prev) => {
      const next = [...prev, message];
      if (next.length <= MAX_LOGS) return next;
      return next.slice(next.length - MAX_LOGS);
    });
  };

  const metrics = useMemo(() => {
    if (!lastResult) {
      return [
        { label: "Sharpe", value: "NA", delta: "--" },
        { label: "Max Drawdown", value: "NA", delta: "--" },
        { label: "Turnover", value: "NA", delta: "--" },
        { label: "Final Value", value: "NA", delta: "--" },
      ];
    }

    const metricsSource = lastResult.metrics || lastResult.summary || {};
    const episodeMetrics = lastResult.episodes || [];
    const current = {
      sharpe:
        metricsSource.sharpe ??
        metricsSource.sharpe_mean ??
        meanFromEpisodes(episodeMetrics, "sharpe"),
      max_drawdown:
        metricsSource.max_drawdown ??
        metricsSource.max_drawdown_mean ??
        meanFromEpisodes(episodeMetrics, "max_drawdown"),
      turnover:
        metricsSource.turnover ??
        metricsSource.turnover_mean ??
        meanFromEpisodes(episodeMetrics, "turnover"),
      final_value:
        metricsSource.final_value ??
        metricsSource.final_value_mean ??
        meanFromEpisodes(episodeMetrics, "final_value"),
    };

    const buildDelta = (key, formatter = formatNumber) => {
      if (!prevMetrics || prevMetrics[key] === null) {
        return { text: "--", cls: "" };
      }
      const diff = current[key] - prevMetrics[key];
      const sign = diff > 0 ? "+" : "";
      return {
        text: `${sign}${formatter(diff)}`,
        cls: diff >= 0 ? "positive" : "negative",
      };
    };

    const sharpeDelta = buildDelta("sharpe");
    const drawdownDelta = buildDelta("max_drawdown");
    const turnoverDelta = buildDelta("turnover");
    const finalDelta = buildDelta("final_value", (value) =>
      Number(value).toFixed(0)
    );

    return [
      {
        label: "Sharpe",
        value: formatNumber(current.sharpe),
        delta: sharpeDelta.text,
        deltaClass: sharpeDelta.cls,
      },
      {
        label: "Max Drawdown",
        value: formatNumber(current.max_drawdown),
        delta: drawdownDelta.text,
        deltaClass: drawdownDelta.cls,
      },
      {
        label: "Turnover",
        value: formatNumber(current.turnover),
        delta: turnoverDelta.text,
        deltaClass: turnoverDelta.cls,
      },
      {
        label: "Final Value",
        value: current.final_value
          ? Number(current.final_value).toFixed(0)
          : "NA",
        delta: finalDelta.text,
        deltaClass: finalDelta.cls,
      },
    ];
  }, [lastResult, prevMetrics]);

  const sparklineValues = useMemo(() => {
    if (!lastResult) return [4, 6, 7, 6, 8, 7, 9];
    const trajectory =
      lastResult.trajectory ||
      lastResult.episodes?.[0]?.trajectory ||
      [];
    if (!trajectory.length) return [4, 6, 7, 6, 8, 7, 9];
    const step = Math.max(1, Math.floor(trajectory.length / 12));
    return trajectory.filter((_, idx) => idx % step === 0).slice(0, 12);
  }, [lastResult]);

  const actionBars = useMemo(() => {
    const metricsSource = lastResult?.metrics || lastResult?.summary || {};
    const episodes = lastResult?.episodes || [];
    const hold =
      metricsSource.action_hold_ratio ??
      meanFromEpisodes(episodes, "action_hold_ratio") ??
      0;
    const buy =
      metricsSource.action_buy_ratio ??
      meanFromEpisodes(episodes, "action_buy_ratio") ??
      0;
    const sell =
      metricsSource.action_sell_ratio ??
      meanFromEpisodes(episodes, "action_sell_ratio") ??
      0;
    return [
      { label: "hold", value: hold },
      { label: "buy", value: buy },
      { label: "sell", value: sell },
    ];
  }, [lastResult]);

  const handleChange = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const runSimulation = async () => {
    setStatus("running");
    setProgressPct(0);
    lastProgressRef.current = 0;
    appendLog(
      `- run: ${form.scenario} | ${form.agent_type} | ${form.reward_mode}`
    );

    const payload = {
      scenario: form.scenario,
      agent_type: form.agent_type,
      reward_mode: form.reward_mode,
      episodes: Number(form.episodes),
      timesteps: Number(form.timesteps),
      schedule: form.schedule
        ? form.schedule.split(",").map((s) => s.trim()).filter(Boolean)
        : null,
      schedule_length: Number(form.schedule_length),
      drawdown_coeff: Number(form.drawdown_coeff),
      volatility_coeff: Number(form.volatility_coeff),
      trade_penalty_coeff: Number(form.trade_penalty_coeff),
      invalid_action_penalty: Number(form.invalid_action_penalty),
      inactivity_penalty: Number(form.inactivity_penalty),
      trade_size: Number(form.trade_size),
    };

    try {
      await runExperimentStream(payload, (event) => {
        if (event.type === "progress") {
          const pct = Math.min(100, Math.max(0, event.pct || 0));
          setProgressPct(pct);
          if (pct - lastProgressRef.current >= 5 || pct === 100) {
            lastProgressRef.current = pct;
          }
          return;
        }
        if (event.type === "log") {
          appendLog(event.message);
          return;
        }
        if (event.type === "error") {
          setStatus("error");
          setProgressPct(null);
          appendLog(`! error: ${event.message || "run failed"}`);
          return;
        }
        if (event.type === "result") {
          setPrevMetrics(() => {
            if (!lastResult) return null;
            const source = lastResult.metrics || lastResult.summary || {};
            return {
              sharpe:
                source.sharpe ??
                source.sharpe_mean ??
                meanFromEpisodes(lastResult.episodes, "sharpe"),
              max_drawdown:
                source.max_drawdown ??
                source.max_drawdown_mean ??
                meanFromEpisodes(lastResult.episodes, "max_drawdown"),
              turnover:
                source.turnover ??
                source.turnover_mean ??
                meanFromEpisodes(lastResult.episodes, "turnover"),
              final_value:
                source.final_value ??
                source.final_value_mean ??
                meanFromEpisodes(lastResult.episodes, "final_value"),
            };
          });
          setLastResult(event.result);
          setStatus("live");
          setProgressPct(100);
          setLastSync(new Date().toLocaleTimeString());
          appendLog("- run complete");
          appendLog(
            `- sharpe: ${
              event.result.metrics?.sharpe ??
              event.result.summary?.sharpe_mean ??
              "NA"
            }`
          );
        }
      });
    } catch (error) {
      setStatus("error");
      setProgressPct(null);
      appendLog(`! error: ${error.message || "run failed"}`);
    }
  };

  return (
    <div className="app-shell">
      <header className="console-header panel active">
        <div>
          <div className="eyebrow">PROSPERITY GROVE</div>
          <h1>Research Console</h1>
          <p className="muted">
            Regime stress testing | Reward shaping | Agent diagnostics
          </p>
        </div>
        <div className="header-right">
          <StatusPill
            status={status}
            label={
              status === "running"
                ? "Simulation Running"
                : status === "error"
                ? "Simulation Error"
                : "Simulation Ready"
            }
          />
          <div className="timestamp muted">last sync: {lastSync}</div>
        </div>
      </header>

      <div className="console-grid">
        <div className="column">
          <Panel
            id="experiment"
            title="Experiment Control"
            active={activePanel === "experiment"}
            onSelect={setActivePanel}
          >
            <div className="control-grid">
              <div className="field">
                <label>Scenario</label>
                <select
                  value={form.scenario}
                  onChange={(event) =>
                    handleChange("scenario", event.target.value)
                  }
                >
                  <option value="bull">Bull</option>
                  <option value="bear">Bear</option>
                  <option value="sideways">Sideways</option>
                  <option value="volatile">Volatile</option>
                  <option value="regime_shift_short">
                    Regime Shift (Short)
                  </option>
                  <option value="regime_shift_long">
                    Regime Shift (Long)
                  </option>
                </select>
              </div>
              <div className="field">
                <label>Agent</label>
                <select
                  value={form.agent_type}
                  onChange={(event) =>
                    handleChange("agent_type", event.target.value)
                  }
                >
                  <option value="ppo">PPO</option>
                  <option value="rule_based">Rule Based</option>
                  <option value="buy_and_hold">Buy & Hold</option>
                  <option value="random">Random</option>
                </select>
              </div>
              <div className="field">
                <label>Reward Mode</label>
                <select
                  value={form.reward_mode}
                  onChange={(event) =>
                    handleChange("reward_mode", event.target.value)
                  }
                >
                  <option value="raw">Raw</option>
                  <option value="risk_adjusted">Risk Adjusted</option>
                </select>
              </div>
              <div className="field">
                <label>Timesteps</label>
                <input
                  type="number"
                  value={form.timesteps}
                  onChange={(event) =>
                    handleChange("timesteps", event.target.value)
                  }
                />
              </div>
            </div>
            <div className="control-grid">
              <div className="field">
                <label>Episodes</label>
                <input
                  type="number"
                  value={form.episodes}
                  onChange={(event) =>
                    handleChange("episodes", event.target.value)
                  }
                />
              </div>
              <div className="field">
                <label>Trade Size</label>
                <input
                  type="number"
                  value={form.trade_size}
                  onChange={(event) =>
                    handleChange("trade_size", event.target.value)
                  }
                />
              </div>
              <div className="field">
                <label>Drawdown Coeff</label>
                <input
                  type="number"
                  step="0.01"
                  value={form.drawdown_coeff}
                  onChange={(event) =>
                    handleChange("drawdown_coeff", event.target.value)
                  }
                />
              </div>
              <div className="field">
                <label>Volatility Coeff</label>
                <input
                  type="number"
                  step="0.01"
                  value={form.volatility_coeff}
                  onChange={(event) =>
                    handleChange("volatility_coeff", event.target.value)
                  }
                />
              </div>
              <div className="field">
                <label>Invalid Action Penalty</label>
                <input
                  type="number"
                  step="0.01"
                  value={form.invalid_action_penalty}
                  onChange={(event) =>
                    handleChange("invalid_action_penalty", event.target.value)
                  }
                />
              </div>
              <div className="field">
                <label>Inactivity Penalty</label>
                <input
                  type="number"
                  step="0.01"
                  value={form.inactivity_penalty}
                  onChange={(event) =>
                    handleChange("inactivity_penalty", event.target.value)
                  }
                />
              </div>
              <div className="field">
                <label>Custom Schedule</label>
                <input
                  type="text"
                  placeholder="bull,volatile,bear"
                  value={form.schedule}
                  onChange={(event) =>
                    handleChange("schedule", event.target.value)
                  }
                />
              </div>
              <div className="field">
                <label>Schedule Length</label>
                <input
                  type="number"
                  value={form.schedule_length}
                  onChange={(event) =>
                    handleChange("schedule_length", event.target.value)
                  }
                />
              </div>
            </div>
            <div className="control-actions">
              <button
                className="button-primary"
                onClick={runSimulation}
                disabled={status === "running"}
              >
                {status === "running" ? "Running..." : "Run Experiment"}
              </button>
              <button className="button-ghost">Export Report</button>
            </div>
          </Panel>

          <Panel
            id="agent"
            title="Agent Diagnostics"
            active={activePanel === "agent"}
            onSelect={setActivePanel}
          >
            <div className="diag-grid">
              <div>
                <div className="label">Policy</div>
                <div className="value">
                  {form.agent_type.toUpperCase()} / MLP
                </div>
              </div>
              <div>
                <div className="label">Entropy</div>
                <div className="value">--</div>
              </div>
              <div>
                <div className="label">Trade Size</div>
                <div className="value">{form.trade_size}</div>
              </div>
              <div>
                <div className="label">Inactivity Penalty</div>
                <div className="value">{form.inactivity_penalty}</div>
              </div>
            </div>
            <div className="sparkline-wrap">
              <div className="label">Training Curve</div>
              <Sparkline values={sparklineValues} />
            </div>
          </Panel>

          <Panel
            id="feed"
            title="Data Feed"
            active={activePanel === "feed"}
            onSelect={setActivePanel}
            className="panel-feed"
          >
            <div className="feed-line">
              <span className="tag">synthetic</span>
              <span>{form.scenario}</span>
            </div>
            <div className="feed-line">
              <span className="tag">prices</span>
              <span>length: 240 | step: daily</span>
            </div>
            <div className="feed-line">
              <span className="tag">reward</span>
              <span>
                drawdown: {form.drawdown_coeff} | volatility:{" "}
                {form.volatility_coeff}
              </span>
            </div>
          </Panel>
        </div>

        <div className="column">
          <Panel
            id="metrics"
            title="Live Metrics"
            active={activePanel === "metrics"}
            onSelect={setActivePanel}
          >
            <MetricGrid items={metrics} />
          </Panel>

          <Panel
            id="performance"
            title="Action Mix"
            active={activePanel === "performance"}
            onSelect={setActivePanel}
          >
            <MiniBarChart bars={actionBars} />
            <div className="legend-row">
              <span className="legend-dot" />
              <span className="muted">
                Action ratios per run
              </span>
            </div>
          </Panel>

          <Panel
            id="logs"
            title="Execution Log"
            active={activePanel === "logs"}
            onSelect={setActivePanel}
            className="panel-log"
          >
            {status === "running" && (
              <div className="log-progress">
                <div className="log-progress-label">
                  progress {progressPct?.toFixed(0) ?? 0}%
                </div>
                <div className="log-progress-track">
                  <div
                    className="log-progress-bar"
                    style={{ width: `${progressPct ?? 0}%` }}
                  />
                </div>
              </div>
            )}
            <div className="log">
              {logs.map((line, index) => (
                <div
                  key={`${line}-${index}`}
                  className={`log-line ${
                    line.startsWith("!")
                      ? "log-error"
                      : line.startsWith("-")
                      ? "log-action"
                      : "log-system"
                  }`}
                >
                  {line}
                </div>
              ))}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
