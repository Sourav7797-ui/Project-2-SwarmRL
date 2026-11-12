import { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import SwarmScene from "./SwarmScene.jsx";
import "./style.css";

const API = "http://127.0.0.1:8000";

function Metric({ label, value, unit = "", accent = "" }) {
  return (
    <div className="metric">
      <span className="metric-label">{label}</span>
      <strong className={accent}>{value}{unit}</strong>
    </div>
  );
}

export default function App() {
  const [state, setState] = useState(null);
  const [connection, setConnection] = useState("connecting");
  const [busy, setBusy] = useState(false);
  const socketRef = useRef(null);

  useEffect(() => {
    let active = true;
    let socket;
    let retry;
    let attempts = 0;

    const connect = () => {
      if (!active) return;

      socket = new WebSocket("ws://127.0.0.1:8000/ws/state");
      socketRef.current = socket;

      socket.onopen = () => {
        attempts = 0;
        if (active) setConnection("connected");
      };

      socket.onmessage = (event) => {
        try {
          const next = JSON.parse(event.data);
          if (active) setState(next);
        } catch (error) {
          console.error("Invalid simulation state:", error);
        }
      };

      socket.onerror = () => {
        if (active) setConnection("disconnected");
      };

      socket.onclose = () => {
        if (!active) return;
        setConnection("disconnected");
        const delay = Math.min(1000 * 2 ** attempts, 8000);
        attempts += 1;
        retry = window.setTimeout(connect, delay);
      };
    };

    fetch(`${API}/api/state`)
      .then((response) => {
        if (!response.ok) throw new Error("API unavailable");
        return response.json();
      })
      .then((data) => {
        if (active) setState(data);
      })
      .catch(() => {
        if (active) setConnection("disconnected");
      });

    connect();

    return () => {
      active = false;
      window.clearTimeout(retry);
      if (socket) socket.close();
    };
  }, []);

  const command = useCallback(async (action) => {
    setBusy(true);
    try {
      const response = await fetch(`${API}/api/${action}`, {
        method: "POST",
      });
      if (!response.ok) throw new Error(`Request failed: ${response.status}`);
      const result = await response.json();
      if (action === "reset") setState(null);
      return result;
    } catch (error) {
      console.error(error);
      window.alert(`Could not ${action} the simulation. Is the API server running?`);
    } finally {
      setBusy(false);
    }
  }, []);

  const metrics = state?.metrics;
  const coverage = state?.coverage?.percentage ?? 0;

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">S</div>
          <div>
            <div className="brand-name">SWARMRL</div>
            <div className="brand-subtitle">AUTONOMOUS FLEET COMMAND</div>
          </div>
        </div>

        <div className="topbar-right">
          <span className={`connection-dot ${connection}`} />
          <span className="connection-label">
            {connection === "connected" ? "LIVE CONNECTION" : connection.toUpperCase()}
          </span>
          <span className={`run-status ${state?.running ? "running" : "paused"}`}>
            {state?.running ? "SIMULATION RUNNING" : "SIMULATION PAUSED"}
          </span>
        </div>
      </header>

      <section className="workspace">
        <aside className="sidebar">
          <div className="section-heading">MISSION CONTROL</div>

          <div className="mission-card">
            <span className="eyebrow">ACTIVE MISSION</span>
            <h2>3D Search &amp; Explore</h2>
            <p>Multi-agent autonomous disaster-zone exploration</p>
            <div className="mission-meta">
              <span>WORLD</span>
              <strong>200 × 200 × 40 m</strong>
            </div>
            <div className="mission-meta">
              <span>FLEET</span>
              <strong>{metrics?.agent_count ?? 50} drones</strong>
            </div>
          </div>

          <div className="section-heading">SIMULATION CONTROL</div>
          <div className="controls">
            <button
              className="control-button start"
              disabled={busy || !state || state.running}
              onClick={() => command("start")}
            >
              <span>▶</span> Start mission
            </button>
            <button
              className="control-button pause"
              disabled={busy || !state || !state.running}
              onClick={() => command("pause")}
            >
              <span>Ⅱ</span> Pause mission
            </button>
            <button
              className="control-button reset"
              disabled={busy}
              onClick={() => {
                if (window.confirm("Reset the simulation and clear flight trails?")) {
                  historiesReset();
                  command("reset");
                }
              }}
            >
              <span>↻</span> Reset mission
            </button>
          </div>

          <div className="section-heading">VISUALIZATION KEY</div>
          <div className="legend">
            <div><i className="legend-drone" /> Drone agents</div>
            <div><i className="legend-obstacle" /> Obstacles</div>
            <div><i className="legend-coverage" /> Explored voxels</div>
            <div><i className="legend-trail" /> Flight trajectories</div>
          </div>

          <div className="sidebar-note">
            <span className="note-indicator" />
            DEMONSTRATION CONTROLLER
            <p>Built-in deterministic exploration controller. No trained policy is loaded.</p>
          </div>
        </aside>

        <section className="visual-area">
          <div className="visual-toolbar">
            <div>
              <span className="eyebrow">LIVE ENVIRONMENT</span>
              <h1>Swarm Overview <span className="live-tag">3D</span></h1>
            </div>
            <div className="view-hint">DRAG TO ORBIT · SCROLL TO ZOOM · RIGHT-DRAG TO PAN</div>
          </div>

          <div className="scene-frame">
            {state ? (
              <SwarmScene state={state} />
            ) : (
              <div className="scene-loading">WAITING FOR SIMULATION DATA…</div>
            )}
            <div className="scene-coordinate">WORLD COORDINATES · X / Y / Z</div>
            <div className="scene-step">STEP <strong>{state?.step ?? 0}</strong></div>
          </div>

          <div className="metric-strip">
            <Metric label="FLEET SIZE" value={metrics?.agent_count ?? "--"} />
            <Metric label="ACTIVE DRONES" value={metrics?.active_agents ?? "--"} accent="blue" />
            <Metric label="COVERAGE" value={coverage.toFixed(2)} unit="%" accent="green" />
            <Metric label="MEAN SPEED" value={metrics ? metrics.mean_speed.toFixed(2) : "--"} unit=" m/s" />
            <Metric label="MEAN BATTERY" value={metrics ? (metrics.mean_battery * 100).toFixed(1) : "--"} unit="%" />
            <Metric label="COLLISIONS" value={metrics?.total_collision_events ?? "--"} accent="red" />
          </div>

          <div className="bottom-strip">
            <span>OBSTACLE COLLISION EVENTS <strong>{metrics?.total_obstacle_collision_events ?? 0}</strong></span>
            <span>OBSTACLES <strong>{state?.obstacles?.length ?? 0}</strong></span>
            <span>VOXEL GRID <strong>{state?.coverage?.shape?.join(" × ") ?? "50 × 50 × 10"}</strong></span>
            <span>UPDATE <strong>10 Hz</strong></span>
          </div>
        </section>
      </section>
    </main>
  );
}

function historiesReset() {
  // The scene component resets its history when the viewer is remounted.
  window.dispatchEvent(new Event("swarmrl-reset-trails"));
}

createRoot(document.getElementById("root")).render(<App />);

