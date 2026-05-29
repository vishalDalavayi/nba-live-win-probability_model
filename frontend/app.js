/**
 * Dashboard: WebSocket replay + Chart.js win probability chart
 */

const els = {
  serverUrl: document.getElementById("serverUrl"),
  connectBtn: document.getElementById("connectBtn"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  status: document.getElementById("status"),
  gameBanner: document.getElementById("gameBanner"),
  gameMatchup: document.getElementById("gameMatchup"),
  gameHelp: document.getElementById("gameHelp"),
  period: document.getElementById("period"),
  timeRemaining: document.getElementById("timeRemaining"),
  score: document.getElementById("score"),
  scoreSub: document.getElementById("scoreSub"),
  homeProbLabel: document.getElementById("homeProbLabel"),
  winProb: document.getElementById("winProb"),
  awayProb: document.getElementById("awayProb"),
  chartTitle: document.getElementById("chartTitle"),
};

let socket = null;
let chart = null;
let tick = 0;
let homeTeam = "HOME";
let awayTeam = "AWAY";

function formatTime(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${String(r).padStart(2, "0")}`;
}

function setStatus(msg, isError = false) {
  els.status.textContent = msg;
  els.status.style.color = isError ? "#f87171" : "";
}

function setTeams(home, away, matchup) {
  homeTeam = home || "HOME";
  awayTeam = away || "AWAY";
  els.gameBanner.classList.remove("hidden");
  els.gameMatchup.textContent = matchup || `${awayTeam} @ ${homeTeam}`;
  els.homeProbLabel.textContent = `${homeTeam} win chance`;
  els.chartTitle.textContent = `${homeTeam} win chance over time`;
  if (chart) {
    chart.data.datasets[0].label = `${homeTeam} win %`;
    chart.update();
  }
}

function initChart() {
  const ctx = document.getElementById("probChart").getContext("2d");
  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Home win %",
          data: [],
          borderColor: "#f97316",
          backgroundColor: "rgba(249, 115, 22, 0.15)",
          fill: true,
          tension: 0.2,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      scales: {
        y: {
          min: 0,
          max: 1,
          ticks: {
            callback: (v) => `${(v * 100).toFixed(0)}%`,
          },
          grid: { color: "#2d3a4f" },
        },
        x: {
          title: { display: true, text: "Play-by-play event", color: "#8b9cb3" },
          grid: { color: "#2d3a4f" },
        },
      },
      plugins: {
        legend: { labels: { color: "#e8eef7" } },
      },
    },
  });
}

function resetChart() {
  tick = 0;
  if (!chart) return;
  chart.data.labels = [];
  chart.data.datasets[0].data = [];
  chart.update();
}

function pushChartPoint(prob) {
  if (!chart || prob == null) return;
  tick += 1;
  chart.data.labels.push(String(tick));
  chart.data.datasets[0].data.push(prob);
  if (chart.data.labels.length > 120) {
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }
  chart.update("none");
}

function updateUI(payload) {
  if (payload.home_team) {
    setTeams(payload.home_team, payload.away_team, payload.matchup);
  }

  els.period.textContent =
    payload.period != null ? `Q${payload.period}` : "—";
  els.timeRemaining.textContent = formatTime(payload.seconds_remaining ?? 0);

  const home = payload.home_score ?? 0;
  const away = payload.away_score ?? 0;
  els.score.textContent = `${homeTeam} ${home} – ${awayTeam} ${away}`;
  const lead = payload.leading_team || (home === away ? "Tied" : null);
  els.scoreSub.textContent = lead
    ? lead === "Tied"
      ? "Tied game"
      : `${lead} leading by ${Math.abs(home - away)}`
    : `${homeTeam} is home (left); ${awayTeam} is away (right)`;

  if (payload.win_probability != null) {
    const homePct = (payload.win_probability * 100).toFixed(1);
    const awayPct = (
      (payload.away_win_probability ?? 1 - payload.win_probability) * 100
    ).toFixed(1);
    els.winProb.textContent = `${homePct}%`;
    els.awayProb.textContent = `${awayTeam}: ${awayPct}%`;
    pushChartPoint(payload.win_probability);
  }
}

function connect() {
  const base = els.serverUrl.value.trim().replace(/\/$/, "");
  if (!base) {
    setStatus("Enter a backend URL", true);
    return;
  }

  if (socket) {
    socket.disconnect();
  }

  setStatus("Connecting…");
  socket = io(base, { transports: ["websocket", "polling"] });

  socket.on("connect", () => {
    setStatus("Connected — click Start Replay");
    els.startBtn.disabled = false;
    els.stopBtn.disabled = false;
  });

  socket.on("disconnect", () => {
    setStatus("Disconnected");
    els.startBtn.disabled = true;
    els.stopBtn.disabled = true;
  });

  socket.on("connect_error", (err) => {
    setStatus(`Connection failed: ${err.message}`, true);
  });

  socket.on("connected", () => {
    setStatus("Connected — click Start Replay");
  });

  socket.on("simulation_started", (data) => {
    resetChart();
    setTeams(data.home_team, data.away_team, data.matchup);
    if (data.help) {
      els.gameHelp.textContent = data.help;
    }
    setStatus(`Replaying: ${data.matchup || data.game_id}`);
  });

  socket.on("game_update", (payload) => {
    if (payload.error) {
      setStatus(`Prediction error: ${payload.error}`, true);
    }
    updateUI(payload);
  });

  socket.on("simulation_complete", () => {
    setStatus(`Replay finished — ${els.gameMatchup.textContent}`);
  });

  socket.on("simulation_error", (data) => {
    setStatus(`Simulation error: ${data.error}`, true);
  });
}

function startReplay() {
  if (!socket?.connected) {
    setStatus("Connect first", true);
    return;
  }
  resetChart();
  socket.emit("start_simulation", { interval_sec: 1.0 });
  setStatus("Starting replay…");
}

function stopReplay() {
  if (socket?.connected) {
    socket.emit("stop_simulation");
    setStatus("Stopped");
  }
}

els.connectBtn.addEventListener("click", connect);
els.startBtn.addEventListener("click", startReplay);
els.stopBtn.addEventListener("click", stopReplay);

initChart();
