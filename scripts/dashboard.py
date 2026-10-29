"""Local, dependency-free SwarmRL training/evaluation dashboard.

Reads:
    checkpoints/day10/progress.csv
    checkpoints/day10/evaluation.csv

Run:
    python scripts/dashboard.py

Then open:
    http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent


def read_csv_rows(path: Path) -> list[dict]:
    """Read a CSV file, returning an empty list when it does not exist."""
    if not path.exists() or path.stat().st_size == 0:
        return []

    try:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def number(value):
    """Convert a CSV value to a finite float, or None if unavailable."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    return result if math.isfinite(result) else None


def latest_value(rows: list[dict], key: str):
    for row in reversed(rows):
        value = number(row.get(key))
        if value is not None:
            return value
    return None


def build_payload(training_csv: Path, evaluation_csv: Path) -> dict:
    training = read_csv_rows(training_csv)
    evaluation = read_csv_rows(evaluation_csv)

    return {
        "generated_at": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "training_csv": str(training_csv),
        "evaluation_csv": str(evaluation_csv),
        "training": training,
        "evaluation": evaluation,
        "summary": {
            "training_rows": len(training),
            "evaluation_rows": len(evaluation),
            "latest_reward": latest_value(
                training, "episode_return_mean_raw"
            ),
            "latest_eval_reward": latest_value(evaluation, "reward_raw"),
            "latest_coverage": latest_value(evaluation, "coverage_pct"),
            "latest_drone_collisions": latest_value(
                evaluation, "drone_drone_collisions"
            ),
            "latest_obstacle_collisions": latest_value(
                evaluation, "drone_obstacle_collisions"
            ),
            "latest_stage": (
                evaluation[-1].get("stage", "Unknown")
                if evaluation
                else "Not evaluated"
            ),
        },
    }


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="15">
<title>SwarmRL Training Dashboard</title>
<style>
:root {
  color-scheme: dark;
  --bg: #0b1020;
  --panel: #141c30;
  --panel2: #10182a;
  --border: #26334d;
  --text: #e8eefb;
  --muted: #9aa9c4;
  --accent: #65d6ad;
  --blue: #7eb6ff;
  --orange: #ffbd75;
  --red: #ff7d91;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.5 system-ui, -apple-system, Segoe UI, sans-serif;
}
header {
  padding: 28px clamp(16px, 4vw, 48px);
  background: linear-gradient(135deg, #172746, #11182b);
  border-bottom: 1px solid var(--border);
}
h1 { margin: 0 0 4px; font-size: clamp(24px, 4vw, 34px); }
h2 { margin: 0 0 12px; font-size: 17px; }
p { margin: 6px 0; color: var(--muted); }
main { padding: 24px clamp(12px, 3vw, 36px) 40px; }
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 14px;
  margin-bottom: 18px;
}
.card, .chart-card, .table-card, .note {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 18px;
  min-width: 0;
}
.metric-label { color: var(--muted); font-size: 13px; }
.metric-value { font-size: 27px; font-weight: 750; margin-top: 5px; }
.metric-detail { color: var(--muted); font-size: 12px; margin-top: 4px; }
.charts {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr));
  gap: 14px;
}
.chart-card canvas { display: block; width: 100%; height: 250px; }
.chart-card p { font-size: 12px; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td {
  text-align: left;
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
th { color: var(--muted); font-weight: 600; }
.pill {
  display: inline-block;
  padding: 3px 9px;
  border: 1px solid #315b50;
  color: var(--accent);
  border-radius: 999px;
  font-size: 12px;
}
.note { margin: 16px 0; background: var(--panel2); }
footer { color: var(--muted); padding: 16px 0 0; font-size: 12px; }
@media (max-width: 520px) {
  .chart-card { padding: 12px; }
  .chart-card canvas { height: 220px; }
}
</style>
</head>
<body>
<header>
  <h1>SwarmRL Training Dashboard</h1>
  <p>Training performance · coverage · collision analysis · curriculum stage evaluations</p>
  <p><span class="pill">Local dashboard</span> <span id="updated">Loading data…</span></p>
</header>
<main>
  <section class="grid">
    <div class="card">
      <div class="metric-label">Latest training raw return</div>
      <div class="metric-value" id="trainReward">—</div>
      <div class="metric-detail">From progress.csv</div>
    </div>
    <div class="card">
      <div class="metric-label">Latest evaluation reward</div>
      <div class="metric-value" id="evalReward">—</div>
      <div class="metric-detail">Unscaled environment reward</div>
    </div>
    <div class="card">
      <div class="metric-label">Latest coverage</div>
      <div class="metric-value" id="coverage">—</div>
      <div class="metric-detail">Percentage of coverage grid explored</div>
    </div>
    <div class="card">
      <div class="metric-label">Drone-to-drone collisions</div>
      <div class="metric-value" id="droneCollisions">—</div>
      <div class="metric-detail">Latest evaluated episode</div>
    </div>
    <div class="card">
      <div class="metric-label">Drone-to-obstacle collisions</div>
      <div class="metric-value" id="obstacleCollisions">—</div>
      <div class="metric-detail">Latest evaluated episode</div>
    </div>
    <div class="card">
      <div class="metric-label">Latest evaluated stage</div>
      <div class="metric-value" id="stage" style="font-size:20px">—</div>
      <div class="metric-detail" id="counts">—</div>
    </div>
  </section>

  <div class="note">
    <strong>Metric availability:</strong>
    training returns come from the existing training CSV. Coverage and split
    collision counts come from checkpoint evaluation. Run evaluations for
    each stage to populate the curriculum chart. A stage label means that
    stage was evaluated; it does not mean PPO automatically promoted to it.
  </div>

  <section class="charts">
    <div class="chart-card">
      <h2>Training reward by iteration</h2>
      <canvas id="rewardChart"></canvas>
      <p>Raw episode return; higher is generally better, but compare alongside coverage and collisions.</p>
    </div>
    <div class="chart-card">
      <h2>Coverage by evaluation</h2>
      <canvas id="coverageChart"></canvas>
      <p>Coverage percentage recorded at the end of each evaluated episode.</p>
    </div>
    <div class="chart-card">
      <h2>Drone-to-drone collisions</h2>
      <canvas id="droneChart"></canvas>
      <p>Event count per evaluated episode. Lower is generally better.</p>
    </div>
    <div class="chart-card">
      <h2>Drone-to-obstacle collisions</h2>
      <canvas id="obstacleChart"></canvas>
      <p>Event count per evaluated episode. Lower is generally better.</p>
    </div>
    <div class="chart-card">
      <h2>Curriculum stage evaluations</h2>
      <canvas id="stageChart"></canvas>
      <p>Stage index: 0 = static_world, 1 = dynamic_obstacles, 2 = wind.</p>
    </div>
    <div class="chart-card">
      <h2>Training time per iteration</h2>
      <canvas id="timeChart"></canvas>
      <p>Iteration wall time in seconds, taken from the training CSV.</p>
    </div>
  </section>

  <section class="table-card" style="margin-top:16px">
    <h2>Recent checkpoint evaluations</h2>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>#</th><th>Stage</th><th>Reward</th><th>Coverage</th>
          <th>Drone collisions</th><th>Obstacle collisions</th><th>Episode steps</th>
        </tr></thead>
        <tbody id="evaluationRows"></tbody>
      </table>
    </div>
  </section>

  <section class="table-card" style="margin-top:16px">
    <h2>Training iterations</h2>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Iteration</th><th>Raw reward</th><th>Seconds</th>
          <th>Policy loss</th><th>Value loss</th><th>Entropy</th>
        </tr></thead>
        <tbody id="trainingRows"></tbody>
      </table>
    </div>
  </section>

  <footer>
    Training CSV: <span id="trainingPath"></span><br>
    Evaluation CSV: <span id="evaluationPath"></span><br>
    Data refreshes when this page reloads; automatic refresh is enabled every 15 seconds.
  </footer>
</main>
<script>
const DATA = __DATA__;

function fmt(value, digits=2) {
  if (value === null || value === undefined || value === "") return "—";
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
}
function numeric(value) {
  if (value === null || value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}
function setText(id, value) {
  document.getElementById(id).textContent = value;
}
function setMetric(id, value, suffix="") {
  setText(id, value === null || value === undefined ? "—" : fmt(value) + suffix);
}

function drawLineChart(canvasId, rows, xKey, series) {
  const canvas = document.getElementById(canvasId);
  const width = Math.max(300, canvas.clientWidth || 400);
  const height = canvas.clientHeight || 250;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);

  const left = 48, right = 14, top = 16, bottom = 34;
  const plotW = width - left - right;
  const plotH = height - top - bottom;

  const points = [];
  for (const item of series) {
    const vals = rows.map((row, index) => ({
      x: numeric(row[xKey]) ?? index + 1,
      y: numeric(row[item.key]),
      label: row[xKey] ?? index + 1
    })).filter(p => p.y !== null);
    if (vals.length) points.push({label:item.label, color:item.color, vals});
  }

  if (!points.length) {
    ctx.fillStyle = "#9aa9c4";
    ctx.font = "13px system-ui";
    ctx.textAlign = "center";
    ctx.fillText("No data yet — run checkpoint evaluation.", width / 2, height / 2);
    return;
  }

  const allY = points.flatMap(s => s.vals.map(p => p.y));
  let minY = Math.min(...allY), maxY = Math.max(...allY);
  if (minY === maxY) { minY -= 1; maxY += 1; }
  const pad = (maxY - minY) * 0.08;
  minY -= pad; maxY += pad;

  const allX = points.flatMap(s => s.vals.map(p => p.x));
  let minX = Math.min(...allX), maxX = Math.max(...allX);
  if (minX === maxX) maxX = minX + 1;

  const xPos = x => left + ((x - minX) / (maxX - minX)) * plotW;
  const yPos = y => top + plotH - ((y - minY) / (maxY - minY)) * plotH;

  ctx.font = "11px system-ui";
  ctx.strokeStyle = "#26334d";
  ctx.fillStyle = "#9aa9c4";
  ctx.lineWidth = 1;
  ctx.textAlign = "right";

  for (let i = 0; i <= 4; i++) {
    const y = top + (plotH * i / 4);
    const value = maxY - (maxY - minY) * i / 4;
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width-right, y); ctx.stroke();
    ctx.fillText(value.toFixed(1), left - 7, y + 4);
  }

  ctx.textAlign = "center";
  const tickCount = Math.min(5, Math.max(2, rows.length));
  for (let i = 0; i < tickCount; i++) {
    const x = left + plotW * i / (tickCount - 1);
    const value = minX + (maxX - minX) * i / (tickCount - 1);
    ctx.fillText(String(Math.round(value)), x, height - 12);
  }

  for (const s of points) {
    ctx.strokeStyle = s.color;
    ctx.fillStyle = s.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    s.vals.forEach((p, i) => {
      const x = xPos(p.x), y = yPos(p.y);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    for (const p of s.vals) {
      ctx.beginPath();
      ctx.arc(xPos(p.x), yPos(p.y), 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  let legendX = left;
  ctx.font = "11px system-ui";
  ctx.textAlign = "left";
  for (const s of points) {
    ctx.fillStyle = s.color;
    ctx.fillRect(legendX, 3, 9, 9);
    ctx.fillStyle = "#e8eefb";
    ctx.fillText(s.label, legendX + 13, 12);
    legendX += ctx.measureText(s.label).width + 34;
  }
}

function renderTables() {
  const evalRows = [...DATA.evaluation].slice(-12).reverse();
  const evalBody = document.getElementById("evaluationRows");

  if (!evalRows.length) {
    evalBody.innerHTML = '<tr><td colspan="7">No evaluation rows yet. Run evaluate_checkpoint.py first.</td></tr>';
  } else {
    evalBody.innerHTML = evalRows.map(r => `<tr>
      <td>${r.evaluation_number ?? "—"}</td>
      <td>${r.stage ?? "—"}</td>
      <td>${fmt(r.reward_raw)}</td>
      <td>${fmt(r.coverage_pct)}%</td>
      <td>${fmt(r.drone_drone_collisions, 0)}</td>
      <td>${fmt(r.drone_obstacle_collisions, 0)}</td>
      <td>${fmt(r.episode_steps, 0)}</td>
    </tr>`).join("");
  }

  const trainRows = [...DATA.training].slice(-12).reverse();
  const trainBody = document.getElementById("trainingRows");

  if (!trainRows.length) {
    trainBody.innerHTML = '<tr><td colspan="6">No training rows found. Start training to populate progress.csv.</td></tr>';
  } else {
    trainBody.innerHTML = trainRows.map(r => `<tr>
      <td>${r.iteration ?? "—"}</td>
      <td>${fmt(r.episode_return_mean_raw)}</td>
      <td>${fmt(r.iteration_seconds)}</td>
      <td>${fmt(r.policy_loss, 4)}</td>
      <td>${fmt(r.value_loss, 4)}</td>
      <td>${fmt(r.entropy, 4)}</td>
    </tr>`).join("");
  }
}

function render() {
  const s = DATA.summary;
  setMetric("trainReward", s.latest_reward);
  setMetric("evalReward", s.latest_eval_reward);
  setMetric("coverage", s.latest_coverage, "%");
  setMetric("droneCollisions", s.latest_drone_collisions, "");
  setMetric("obstacleCollisions", s.latest_obstacle_collisions, "");
  setText("stage", s.latest_stage || "Not evaluated");
  setText("counts", `${s.training_rows} training rows · ${s.evaluation_rows} evaluation episodes`);
  setText("updated", `Updated ${DATA.generated_at}`);

  setText("trainingPath", DATA.training_csv);
  setText("evaluationPath", DATA.evaluation_csv);

  drawLineChart("rewardChart", DATA.training, "iteration", [
    {key:"episode_return_mean_raw", label:"Raw return", color:"#65d6ad"}
  ]);
  drawLineChart("coverageChart", DATA.evaluation, "evaluation_number", [
    {key:"coverage_pct", label:"Coverage %", color:"#7eb6ff"}
  ]);
  drawLineChart("droneChart", DATA.evaluation, "evaluation_number", [
    {key:"drone_drone_collisions", label:"Drone collisions", color:"#ffbd75"}
  ]);
  drawLineChart("obstacleChart", DATA.evaluation, "evaluation_number", [
    {key:"drone_obstacle_collisions", label:"Obstacle collisions", color:"#ff7d91"}
  ]);
  drawLineChart("stageChart", DATA.evaluation, "evaluation_number", [
    {key:"stage_index", label:"Stage index", color:"#65d6ad"}
  ]);
  drawLineChart("timeChart", DATA.training, "iteration", [
    {key:"iteration_seconds", label:"Seconds", color:"#c2a0ff"}
  ]);

  renderTables();
}

render();
window.addEventListener("resize", render);
</script>
</body>
</html>
"""


def make_page(payload: dict) -> bytes:
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    # Prevent any CSV text from terminating the embedded script element.
    serialized = (
        serialized.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    html = PAGE.replace("__DATA__", serialized)
    return html.encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-csv",
        default="checkpoints/day10/progress.csv",
    )
    parser.add_argument(
        "--evaluation-csv",
        default="checkpoints/day10/evaluation.csv",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    def resolve_path(value: str) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

    training_csv = resolve_path(args.training_csv)
    evaluation_csv = resolve_path(args.evaluation_csv)

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if urlparse(self.path).path not in ("/", "/index.html"):
                self.send_error(404)
                return

            payload = build_payload(training_csv, evaluation_csv)
            body = make_page(payload)

            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            # Keep the console readable while the dashboard is open.
            sys.stdout.write(
                "[dashboard] " + fmt % args + "\n"
            )

    try:
        server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    except OSError as exc:
        print(f"ERROR: could not start dashboard server: {exc}")
        return 2

    print("SwarmRL dashboard is running.")
    print(f"Open: http://{args.host}:{args.port}")
    print(f"Training CSV:   {training_csv}")
    print(f"Evaluation CSV: {evaluation_csv}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())