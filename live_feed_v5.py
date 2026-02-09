import json
import logging
import threading
import time

import cv2
import numpy as np
import zmq
from flask import Flask, Response, jsonify, render_template_string

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dashboard")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

JETSON_IP = "192.168.3.89"
DATA_PORT = 5555
CONTROL_PORT = 5556  # NEW: port for sending control commands
DEBUG_MODE = False

app = Flask(__name__)

latest_frame = None
latest_telemetry = {
    "raw_steer": 0.0,
    "smooth_steer": 0.0,
    "throttle": 0.0,
    "mode": "NORMAL",
    "incident": "NONE",
    "fps": 0.0,
    "detections": [],
    "is_running": False,
}
lock = threading.Lock()

# NEW: Control socket (global)
control_socket = None


# ──────────────────────────────────────────────
# ZMQ Listener
# ──────────────────────────────────────────────


def _correct_white_balance(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    a_mean, b_mean = a.mean(), b.mean()
    a = a - (a_mean - 128)
    b = b - (b_mean - 128)
    corrected = cv2.merge([l, a, b]).astype(np.uint8)
    return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)


def zmq_listener():
    global latest_frame, latest_telemetry, control_socket

    if DEBUG_MODE:
        log.warning("Running in DEBUG mode — using synthetic data")
        _run_debug_loop()
        return

    log.info("Connecting to Jetson at %s:%d (data) and :%d (control)", 
             JETSON_IP, DATA_PORT, CONTROL_PORT)
    ctx = zmq.Context()
    
    # Data subscription socket
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{JETSON_IP}:{DATA_PORT}")
    sub.setsockopt(zmq.SUBSCRIBE, b"dashboard")
    
    # NEW: Control publishing socket
    control_socket = ctx.socket(zmq.PUB)
    control_socket.connect(f"tcp://{JETSON_IP}:{CONTROL_PORT}")
    time.sleep(0.5)  # Give ZMQ time to establish connection

    while True:
        try:
            _, json_data, img_bytes = sub.recv_multipart()
            telemetry = json.loads(json_data.decode("utf-8"))
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                cv2.rectangle(frame, (80, 0), (560, 479), (0, 255, 0), 1)
                for d in telemetry.get("detections", []):
                    x, y, w, h = d["box"]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    label = f"{d['class']} {d.get('score', 0):.2f}"
                    cv2.rectangle(
                        frame, (x, y - 20), (x + 140, y), (0, 0, 255), -1
                    )
                    cv2.putText(
                        frame, label, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    )

            with lock:
                latest_frame = _correct_white_balance(frame) if frame is not None else None
                latest_telemetry = telemetry

        except Exception as e:
            log.error("ZMQ receive error: %s", e)
            time.sleep(0.1)


def _run_debug_loop():
    global latest_frame, latest_telemetry

    while True:
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            fake_frame, "DEBUG", (240, 250),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2,
        )
        fake_telem = {
            "raw_steer": float(np.sin(time.time())),
            "smooth_steer": float(np.sin(time.time()) * 0.8),
            "throttle": 0.35 + 0.05 * float(np.sin(time.time() * 0.5)),
            "fps": 30.0,
            "mode": "LIMIT" if int(time.time()) % 10 > 5 else "NORMAL",
            "incident": "CHILD" if int(time.time()) % 15 > 12 else "NONE",
            "detections": [
                {"class": "stop_sign", "score": 0.95, "box": [200, 150, 100, 100]}
            ],
            "is_running": int(time.time()) % 20 > 10,
        }
        with lock:
            latest_frame = fake_frame
            latest_telemetry = fake_telem
        time.sleep(0.03)


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Race Telemetry</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/smoothie/1.34.0/smoothie.min.js"></script>
    <style>
        *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

        :root {
            --bg-primary: #09090b;
            --bg-card: #111113;
            --bg-card-alt: #18181b;
            --border: #27272a;
            --border-subtle: #1e1e21;
            --text-primary: #e4e4e7;
            --text-secondary: #71717a;
            --text-dim: #52525b;

            --accent-cyan: #22d3ee;
            --accent-green: #4ade80;
            --accent-rose: #fb7185;
            --accent-amber: #fbbf24;
            --accent-red: #ef4444;

            --font-mono: 'JetBrains Mono', 'Consolas', monospace;
            --font-sans: 'IBM Plex Sans', system-ui, sans-serif;

            --radius: 6px;
        }

        body {
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: var(--font-sans);
            padding: 24px;
            min-height: 100vh;
        }

        .dashboard {
            max-width: 1080px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        /* Header */
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border);
        }
        .header-left {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .header-title {
            font-family: var(--font-mono);
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-secondary);
        }
        .header-dot {
            width: 7px; height: 7px;
            border-radius: 50%;
            background: var(--accent-green);
            box-shadow: 0 0 8px var(--accent-green);
            animation: pulse-dot 2s ease-in-out infinite;
        }
        @keyframes pulse-dot {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        /* Control buttons */
        .control-buttons {
            display: flex;
            gap: 8px;
        }
        .control-btn {
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 8px 20px;
            border: 1px solid var(--border);
            border-radius: 4px;
            background: var(--bg-card);
            color: var(--text-primary);
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .control-btn:hover {
            background: var(--bg-card-alt);
            border-color: var(--text-secondary);
        }
        .control-btn:active {
            transform: scale(0.96);
        }
        .control-btn.start {
            border-color: rgba(74, 222, 128, 0.4);
            color: var(--accent-green);
        }
        .control-btn.start:hover {
            background: rgba(74, 222, 128, 0.08);
            border-color: var(--accent-green);
        }
        .control-btn.stop {
            border-color: rgba(239, 68, 68, 0.4);
            color: var(--accent-red);
        }
        .control-btn.stop:hover {
            background: rgba(239, 68, 68, 0.08);
            border-color: var(--accent-red);
        }
        .control-btn:disabled {
            opacity: 0.3;
            cursor: not-allowed;
        }
        .control-btn:disabled:hover {
            background: var(--bg-card);
            border-color: var(--border);
            transform: none;
        }

        /* Top row */
        .top-row {
            display: flex;
            gap: 16px;
        }

        /* Video */
        .video-panel {
            width: 640px;
            height: 480px;
            flex-shrink: 0;
            border-radius: var(--radius);
            overflow: hidden;
            border: 1px solid var(--border);
            background: #000;
            position: relative;
        }
        .video-panel img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }
        .video-label {
            position: absolute;
            top: 8px; left: 10px;
            font-family: var(--font-mono);
            font-size: 10px;
            font-weight: 500;
            color: var(--text-dim);
            letter-spacing: 0.06em;
            text-transform: uppercase;
            background: rgba(0,0,0,0.6);
            padding: 2px 6px;
            border-radius: 3px;
        }

        /* Stats sidebar */
        .stats-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 12px;
            min-width: 260px;
        }

        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .stat-cell {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 14px 16px;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .stat-cell .stat-key {
            font-family: var(--font-mono);
            font-size: 10px;
            font-weight: 500;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-dim);
        }
        .stat-cell .stat-val {
            font-family: var(--font-mono);
            font-size: 20px;
            font-weight: 600;
            color: var(--text-primary);
        }

        .mode-normal { color: var(--accent-cyan) !important; }
        .mode-limit  { color: var(--accent-amber) !important; }

        /* Graph */
        .graph-panel {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 16px;
        }
        .graph-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 12px;
        }
        .graph-title {
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-secondary);
        }
        .graph-legend {
            display: flex;
            gap: 16px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-secondary);
        }
        .legend-swatch {
            width: 14px;
            height: 3px;
            border-radius: 2px;
        }
        .graph-canvas-wrap {
            width: 100%;
            height: 180px;
        }
        .graph-canvas-wrap canvas {
            width: 100% !important;
            height: 100% !important;
        }

        /* Status colors */
        .status-idle     { color: var(--text-dim) !important; }
        .status-ready    { color: var(--accent-cyan) !important; }
        .status-driving  { color: var(--accent-green) !important; }
        .status-limit    { color: var(--accent-amber) !important; }
        .status-slowing  { color: var(--accent-amber) !important; }
        .status-stopping { color: var(--accent-red) !important; animation: blink-text 0.8s step-end infinite; }
        @keyframes blink-text { 50% { opacity: 0.3; } }

        /* Detection sidebar */
        .det-sidebar {
            flex: 1;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 14px 16px;
            overflow-y: auto;
        }
        .det-title {
            font-family: var(--font-mono);
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--text-dim);
            margin-bottom: 10px;
        }
        .det-list {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        .det-tag {
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 500;
            padding: 4px 10px;
            border-radius: 3px;
            background: var(--bg-card-alt);
            border: 1px solid var(--border);
            color: var(--text-primary);
        }
        .det-tag.det-danger {
            border-color: rgba(239, 68, 68, 0.4);
            color: var(--accent-red);
            background: rgba(239, 68, 68, 0.08);
        }
        .det-empty {
            font-family: var(--font-mono);
            font-size: 11px;
            color: var(--text-dim);
        }
        .det-ago {
            font-size: 10px;
            color: var(--text-dim);
            margin-left: 4px;
        }
        .det-tag.det-stale {
            opacity: 0.45;
        }
        .det-meta {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 10px;
            color: var(--text-dim);
            margin-left: 6px;
        }
        .det-badge {
            font-family: var(--font-mono);
            font-size: 9px;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            padding: 1px 5px;
            border-radius: 2px;
        }
        .det-badge-active {
            background: rgba(74, 222, 128, 0.15);
            color: var(--accent-green);
            border: 1px solid rgba(74, 222, 128, 0.3);
        }
        .det-badge-far {
            background: rgba(113, 113, 122, 0.15);
            color: var(--text-dim);
            border: 1px solid rgba(113, 113, 122, 0.3);
        }
    </style>
</head>
<body>
    <div class="dashboard">

        <div class="header">
            <div class="header-left">
                <span class="header-title">Race Telemetry</span>
                <div class="header-dot" id="connDot"></div>
            </div>
            <div class="control-buttons">
                <button class="control-btn start" id="startBtn" onclick="sendStart()">▶ Start</button>
                <button class="control-btn stop" id="stopBtn" onclick="sendStop()">■ Stop</button>
            </div>
        </div>

        <div class="top-row">
            <div class="video-panel">
                <span class="video-label">Live Feed</span>
                <img src="/video_feed" alt="camera">
            </div>

            <div class="stats-panel">
                <div class="stat-grid">
                    <div class="stat-cell">
                        <span class="stat-key">Status</span>
                        <span class="stat-val status-idle" id="statusText">IDLE</span>
                    </div>
                    <div class="stat-cell">
                        <span class="stat-key">Mode</span>
                        <span class="stat-val mode-normal" id="modeVal">NORMAL</span>
                    </div>
                    <div class="stat-cell">
                        <span class="stat-key">Event</span>
                        <span class="stat-val" id="incidentVal">NONE</span>
                    </div>
                    <div class="stat-cell">
                        <span class="stat-key">Throttle</span>
                        <span class="stat-val" id="thrVal">0.00</span>
                    </div>
                    <div class="stat-cell">
                        <span class="stat-key">Steering</span>
                        <span class="stat-val" id="steerVal">0.00</span>
                    </div>
                    <div class="stat-cell">
                        <span class="stat-key">FPS</span>
                        <span class="stat-val" id="fpsVal">0.0</span>
                    </div>
                </div>

                <div class="det-sidebar">
                    <div class="det-title">Detections</div>
                    <div class="det-list" id="detList">
                        <span class="det-empty">No objects detected</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="graph-panel">
            <div class="graph-header">
                <span class="graph-title">Signal Trace</span>
                <div class="graph-legend">
                    <div class="legend-item">
                        <span class="legend-swatch" style="background:#4ade80"></span>
                        Raw Steer
                    </div>
                    <div class="legend-item">
                        <span class="legend-swatch" style="background:#22d3ee"></span>
                        Kalman Steer
                    </div>
                    <div class="legend-item">
                        <span class="legend-swatch" style="background:#fb7185"></span>
                        Throttle
                    </div>
                </div>
            </div>
            <div class="graph-canvas-wrap">
                <canvas id="graphCanvas"></canvas>
            </div>
        </div>

    </div>

    <script>
        var canvas = document.getElementById("graphCanvas");
        function resizeCanvas() {
            canvas.width = canvas.clientWidth;
            canvas.height = canvas.clientHeight;
        }
        window.addEventListener("resize", resizeCanvas);
        resizeCanvas();

        var chart = new SmoothieChart({
            millisPerPixel: 14,
            interpolation: "bezier",
            grid: {
                fillStyle: "transparent",
                strokeStyle: "rgba(255,255,255,0.04)",
                verticalSections: 4,
                millisPerLine: 2000,
                borderVisible: false
            },
            labels: {
                fillStyle: "rgba(255,255,255,0.25)",
                fontSize: 10,
                fontFamily: "JetBrains Mono, monospace",
                precision: 1
            },
            maxValue: 1.3,
            minValue: -1.3,
            responsive: true,
            tooltip: false
        });

        var tsRaw = new TimeSeries();
        var tsSmooth = new TimeSeries();
        var tsThrottle = new TimeSeries();

        chart.addTimeSeries(tsRaw, {
            strokeStyle: "rgba(74, 222, 128, 0.5)",
            lineWidth: 1.2
        });
        chart.addTimeSeries(tsSmooth, {
            strokeStyle: "#22d3ee",
            lineWidth: 2
        });
        chart.addTimeSeries(tsThrottle, {
            strokeStyle: "rgba(251, 113, 133, 0.7)",
            lineWidth: 1.5
        });
        chart.streamTo(canvas, 500);

        var DANGER_CLASSES = new Set(["stop sign", "stop_sign", "child"]);
        var DET_PERSIST_MS = 5000;
        var detHistory = {};

        function updateDetHistory(dets, now) {
            for (var i = 0; i < dets.length; i++) {
                var key = dets[i]["class"];
                detHistory[key] = {
                    score: dets[i].score || 0,
                    area: dets[i].area || 0,
                    active: dets[i].active !== false,
                    lastSeen: now
                };
            }
            var keys = Object.keys(detHistory);
            for (var j = 0; j < keys.length; j++) {
                if (now - detHistory[keys[j]].lastSeen > DET_PERSIST_MS) {
                    delete detHistory[keys[j]];
                }
            }
        }

        function renderDetList(now) {
            var listEl = document.getElementById("detList");
            var keys = Object.keys(detHistory);
            if (keys.length === 0) {
                listEl.innerHTML = '<span class="det-empty">No objects detected</span>';
                return;
            }
            listEl.innerHTML = keys.map(function(name) {
                var entry = detHistory[name];
                var ago = Math.floor((now - entry.lastSeen) / 1000);
                var cls = DANGER_CLASSES.has(name) ? " det-danger" : "";
                var stale = ago > 0 ? " det-stale" : "";
                var agoText = ago === 0 ? "now" : ago + "s ago";
                var badgeCls = entry.active ? "det-badge-active" : "det-badge-far";
                var badgeText = entry.active ? "active" : "far";
                return '<span class="det-tag' + cls + stale + '">'
                    + name + " " + entry.score.toFixed(2)
                    + '<span class="det-meta">'
                    + entry.area + "px"
                    + '<span class="det-badge ' + badgeCls + '">' + badgeText + "</span>"
                    + '<span class="det-ago">' + agoText + "</span>"
                    + "</span></span>";
            }).join("");
        }

        function sendStart() {
            fetch("/control/start", {method: "POST"})
                .then(function(r) { return r.json(); })
                .then(function(d) {
                    console.log("Start command sent:", d);
                })
                .catch(function(err) {
                    console.error("Failed to send start:", err);
                });
        }

        function sendStop() {
            fetch("/control/stop", {method: "POST"})
                .then(function(r) { return r.json(); })
                .then(function(d) {
                    console.log("Stop command sent:", d);
                })
                .catch(function(err) {
                    console.error("Failed to send stop:", err);
                });
        }

        setInterval(function() {
            fetch("/data").then(function(r) { return r.json(); }).then(function(d) {
                var now = Date.now();

                document.getElementById("fpsVal").textContent = (d.fps || 0).toFixed(1);
                document.getElementById("thrVal").textContent = (d.throttle || 0).toFixed(2);
                document.getElementById("steerVal").textContent = (d.smooth_steer || 0).toFixed(2);

                var modeEl = document.getElementById("modeVal");
                modeEl.textContent = d.mode || "NORMAL";
                modeEl.className = "stat-val " + (d.mode === "LIMIT" ? "mode-limit" : "mode-normal");

                var inc = d.incident || "NONE";
                var incEl = document.getElementById("incidentVal");
                incEl.textContent = inc;

                var st = document.getElementById("statusText");
                st.className = "stat-val";

                var isRunning = d.is_running || false;
                var startBtn = document.getElementById("startBtn");
                var stopBtn = document.getElementById("stopBtn");

                if (!isRunning) {
                    st.textContent = "IDLE";
                    st.classList.add("status-idle");
                    incEl.style.color = "var(--text-dim)";
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                } else {
                    startBtn.disabled = true;
                    stopBtn.disabled = false;

                    if (inc === "STOP") {
                        st.textContent = "STOPPING";
                        st.classList.add("status-stopping");
                        incEl.style.color = "var(--accent-red)";
                    } else if (inc === "CHILD" || inc === "ADULT") {
                        st.textContent = "SLOWING";
                        st.classList.add("status-slowing");
                        incEl.style.color = "var(--accent-amber)";
                    } else if (d.mode === "LIMIT") {
                        st.textContent = "LIMIT";
                        st.classList.add("status-limit");
                        incEl.style.color = "var(--accent-amber)";
                    } else {
                        st.textContent = "DRIVING";
                        st.classList.add("status-driving");
                        incEl.style.color = "var(--text-primary)";
                    }
                }

                tsRaw.append(now, d.raw_steer || 0);
                tsSmooth.append(now, d.smooth_steer || 0);
                tsThrottle.append(now, d.throttle || 0);

                updateDetHistory(d.detections || [], now);
                renderDetList(now);
            });
        }, 50);
    </script>
</body>
</html>
'''


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with lock:
                frame = latest_frame
            if frame is not None:
                _, jpg = cv2.imencode(".jpg", frame)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
                )
            time.sleep(0.04)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/data")
def data():
    with lock:
        return jsonify(latest_telemetry)


# NEW: Control endpoints
@app.route("/control/start", methods=["POST"])
def control_start():
    global control_socket
    if control_socket is not None:
        cmd = {"command": "start"}
        control_socket.send_multipart([b"control", json.dumps(cmd).encode("utf-8")])
        log.info("▶ START command sent to Jetson")
        return jsonify({"status": "ok", "command": "start"})
    return jsonify({"status": "error", "message": "Control socket not initialized"}), 500


@app.route("/control/stop", methods=["POST"])
def control_stop():
    global control_socket
    if control_socket is not None:
        cmd = {"command": "stop"}
        control_socket.send_multipart([b"control", json.dumps(cmd).encode("utf-8")])
        log.info("■ STOP command sent to Jetson")
        return jsonify({"status": "ok", "command": "stop"})
    return jsonify({"status": "error", "message": "Control socket not initialized"}), 500


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    listener = threading.Thread(target=zmq_listener, daemon=True)
    listener.start()
    log.info("Dashboard starting on http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
