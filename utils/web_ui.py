#!/usr/bin/env python
"""pyZeeTom Web UI — Interactive parameter editor and workflow launcher.

Usage
-----
Run from the project root directory::

    python utils/web_ui.py [--host HOST] [--port PORT] [--no-browser]

Then open http://localhost:5000 in your browser.

Features
--------
* Load / edit / save parameter files (.txt or .json format).
* One-click buttons to start the **Forward** or **Inversion** workflow.
* Real-time console output streamed to the browser via Server-Sent Events.
* Stop a running job at any time.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import queue
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that `core` and `pyzeetom` are
# importable regardless of how the script is launched.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # utils/
_ROOT = _HERE.parent                             # project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flask import Flask, Response, jsonify, render_template, request

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, template_folder=str(_HERE / "templates"))
app.config["JSON_SORT_KEYS"] = False

# ---------------------------------------------------------------------------
# In-memory job registry
# ---------------------------------------------------------------------------
_jobs: Dict[str, Dict[str, Any]] = {}  # job_id -> {queue, thread, stopped, success}


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

def _par_to_dict(par) -> Dict[str, Any]:
    """Convert a readParamsTomog object to a JSON-friendly dict."""
    obs_files = []
    for i in range(par.numObs):
        polchan = par.polChannels[i] if i < len(par.polChannels) else "V"
        obs_files.append(
            {
                "path": str(par.fnames[i]),
                "jdate": float(par.jDates[i]),
                "vr": float(par.velRs[i]),
                "polchannel": str(polchan),
            }
        )
    return {
        # Stellar
        "inclination": float(par.inclination),
        "vsini": float(par.vsini),
        "period": float(par.period),
        "pOmega": float(par.pOmega),
        # Disk
        "mass": float(par.mass),
        "radius": float(par.radius),
        "Vmax": float(par.Vmax),
        "r_out": float(getattr(par, "r_out", 0.0)),
        "enable_stellar_occultation": int(getattr(par, "enable_stellar_occultation", 0)),
        # Grid
        "nRingsStellarGrid": int(par.nRingsStellarGrid),
        # Inversion
        "targetForm": str(par.targetForm),
        "targetValue": float(par.targetValue),
        "numIterations": int(par.numIterations),
        "test_aim": float(par.test_aim),
        # Line model
        "lineAmpConst": float(par.lineAmpConst),
        "lineKQU": float(par.lineKQU),
        "lineEnableV": int(par.lineEnableV),
        "lineEnableQU": int(par.lineEnableQU),
        "initTomogFile": int(getattr(par, "initTomogFile", 0)),
        "initModelPath": str(getattr(par, "initModelPath", "") or ""),
        # Fit flags
        "fitBri": int(par.fitBri),
        "fitMag": int(par.fitMag),
        "fitBlos": int(par.fitBlos),
        "fitBperp": int(par.fitBperp),
        "fitChi": int(par.fitChi),
        # Spectral
        "spectralResolution": float(par.spectralResolution),
        "lineParamFile": str(par.lineParamFile),
        "velStart": float(par.velStart),
        "velEnd": float(par.velEnd),
        "obsFileType": str(par.obsFileType),
        "polOut": str(getattr(par, "polOut", "V")),
        "specType": str(getattr(par, "specType", "auto")),
        # Observations
        "jDateRef": float(par.jDateRef),
        "_observations": obs_files,
    }


def _dict_to_json_cfg(p: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a flat params dict (from the web form) to the nested JSON
    schema accepted by readParamsTomog.from_json()."""
    obs_files = p.get("_observations", [])
    return {
        "star": {
            "inclination": float(p.get("inclination", 60.0)),
            "vsini": float(p.get("vsini", 40.0)),
            "period": float(p.get("period", 1.0)),
            "pOmega": float(p.get("pOmega", 0.0)),
            "mass": float(p.get("mass", 1.0)),
            "radius": float(p.get("radius", 1.0)),
        },
        "grid": {
            "nRings": int(p.get("nRingsStellarGrid", 10)),
            "Vmax": float(p.get("Vmax", 0.0)),
            "r_out": float(p.get("r_out", 0.0)),
            "enable_occultation": int(p.get("enable_stellar_occultation", 0)),
        },
        "inversion": {
            "targetForm": str(p.get("targetForm", "C")),
            "targetValue": float(p.get("targetValue", 1.0)),
            "numIterations": int(p.get("numIterations", 5)),
            "test_aim": float(p.get("test_aim", 1e-3)),
        },
        "line_model": {
            "lineAmpConst": float(p.get("lineAmpConst", 1.0)),
            "k_QU": float(p.get("lineKQU", 1.0)),
            "enableV": int(p.get("lineEnableV", 1)),
            "enableQU": int(p.get("lineEnableQU", 1)),
        },
        "initial_model": {
            "initTomogFile": int(p.get("initTomogFile", 0)),
            "initModelPath": str(p.get("initModelPath", "") or ""),
        },
        "fit_flags": {
            "fitBri": int(p.get("fitBri", 1)),
            "fitMag": int(p.get("fitMag", 0)),
            "fitBlos": int(p.get("fitBlos", 0)),
            "fitBperp": int(p.get("fitBperp", 0)),
            "fitChi": int(p.get("fitChi", 0)),
        },
        "spectral": {
            "spectralResolution": float(p.get("spectralResolution", 65000)),
            "lineParamFile": str(p.get("lineParamFile", "input/lines.txt")),
            "velStart": float(p.get("velStart", -400.0)),
            "velEnd": float(p.get("velEnd", 400.0)),
            "obsFileType": str(p.get("obsFileType", "auto")),
            "polOut": str(p.get("polOut", "V")),
            "specType": str(p.get("specType", "auto")),
        },
        "observations": {
            "jDateRef": float(p.get("jDateRef", 0.0)),
            "files": [
                {
                    "path": str(o.get("path", "")),
                    "jdate": float(o.get("jdate", p.get("jDateRef", 0.0))),
                    "vr": float(o.get("vr", 0.0)),
                    "polchannel": str(o.get("polchannel", "V")).upper(),
                }
                for o in obs_files
                if str(o.get("path", "")).strip()
            ],
        },
    }


# ---------------------------------------------------------------------------
# Default parameter snapshot (loaded from the repo's sample file if present)
# ---------------------------------------------------------------------------
_default_params: Optional[Dict[str, Any]] = None


def _load_default_params() -> Optional[Dict[str, Any]]:
    sample = _ROOT / "input" / "params_tomog.txt"
    if not sample.exists():
        return None
    try:
        import core.mainFuncs as mf

        par = mf.readParamsTomog(str(sample), verbose=0)
        return _par_to_dict(par)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/params", methods=["GET"])
def get_default_params():
    """Return the currently-cached default parameter set."""
    global _default_params
    if _default_params is None:
        _default_params = _load_default_params()
    if _default_params:
        return jsonify(_default_params)
    # Return a minimal default so the form still renders
    return jsonify({"_observations": []})


_ALLOWED_PARAM_EXTENSIONS = frozenset({".txt", ".json"})


def _safe_resolve(user_path: str) -> Path:
    """Resolve a user-supplied path to an absolute Path.

    Relative paths are anchored to the project root so that they cannot
    escape via ``../..`` traversal once resolved.
    """
    p = Path(user_path)
    if not p.is_absolute():
        p = _ROOT / p
    return p.resolve()


@app.route("/api/params/load", methods=["POST"])
def load_params():
    """Load a parameter file (.txt or .json) and return its contents."""
    data = request.get_json(force=True, silent=True) or {}
    file_path = (data.get("file_path") or "").strip()
    if not file_path:
        return jsonify({"error": "file_path is required"}), 400

    # Validate extension before touching the filesystem
    if Path(file_path).suffix.lower() not in _ALLOWED_PARAM_EXTENSIONS:
        return jsonify({"error": "Only .txt and .json parameter files are supported"}), 400

    path = _safe_resolve(file_path)

    if not path.exists():
        return jsonify({"error": "File not found"}), 404

    try:
        import core.mainFuncs as mf

        if path.suffix.lower() == ".json":
            par = mf.readParamsTomog.from_json(str(path), verbose=0)
        else:
            par = mf.readParamsTomog(str(path), verbose=0)
        return jsonify(_par_to_dict(par))
    except Exception as exc:
        app.logger.error("load_params error: %s", exc, exc_info=True)
        return jsonify({"error": "Failed to load parameter file. Check server logs."}), 500


@app.route("/api/params/save", methods=["POST"])
def save_params():
    """Save parameters (from form) to a file."""
    data = request.get_json(force=True, silent=True) or {}
    save_path = (data.pop("_save_path", None) or "").strip()
    if not save_path:
        return jsonify({"error": "_save_path is required"}), 400

    # Validate extension before touching the filesystem
    if Path(save_path).suffix.lower() not in _ALLOWED_PARAM_EXTENSIONS:
        return jsonify({"error": "Only .txt and .json parameter files are supported"}), 400

    path = _safe_resolve(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import core.mainFuncs as mf

        cfg = _dict_to_json_cfg(data)
        # Validate: need at least one observation
        if not cfg["observations"]["files"]:
            return jsonify({"error": "At least one observation file is required"}), 400

        # Write via a JSON temp file so we can parse back and use write_params_file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            json.dump(cfg, tmp)
            tmp_json = tmp.name

        try:
            par = mf.readParamsTomog.from_json(tmp_json, verbose=0)
        finally:
            os.unlink(tmp_json)

        if path.suffix.lower() == ".json":
            par.to_json(str(path), verbose=0)
        else:
            par.write_params_file(str(path), verbose=0)

        return jsonify({"ok": True, "saved_to": str(path)})
    except Exception as exc:
        app.logger.error("save_params error: %s", exc, exc_info=True)
        return jsonify({"error": "Failed to save parameter file. Check server logs."}), 500


# ---------------------------------------------------------------------------
# Workflow execution helpers
# ---------------------------------------------------------------------------


class _LogRedirect(io.TextIOBase):
    """TextIO that sends each written line to a queue as an SSE 'log' event."""

    def __init__(self, q: queue.Queue):
        self._q = q
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._q.put(("log", line))
        return len(text)

    def flush(self):
        if self._buf:
            self._q.put(("log", self._buf))
            self._buf = ""


def _run_workflow_thread(
    job_id: str,
    mode: str,
    cfg_json: dict,
    output_dir: str,
):
    """Target function for the background workflow thread."""
    q = _jobs[job_id]["queue"]
    success = False

    # Build a temporary JSON config file
    tmp_json = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            json.dump(cfg_json, tmp)
            tmp_json = tmp.name

        import pyzeetom.tomography as tomo

        # Redirect stdout to the log queue
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected = _LogRedirect(q)
        sys.stdout = redirected
        sys.stderr = redirected

        try:
            if mode == "forward":
                tomo.forward_tomography(
                    param_file=tmp_json,
                    verbose=1,
                    output_dir=output_dir,
                )
            else:
                tomo.inversion_tomography(
                    param_file=tmp_json,
                    verbose=1,
                    output_dir=output_dir,
                )
            success = True
        except Exception as exc:
            import traceback

            q.put(("log", f"\n[ERROR] {exc}"))
            q.put(("log", traceback.format_exc()))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    except Exception as outer_exc:
        q.put(("log", f"[FATAL] Could not start workflow: {outer_exc}"))
    finally:
        if tmp_json and os.path.exists(tmp_json):
            os.unlink(tmp_json)

    _jobs[job_id]["success"] = success
    _jobs[job_id]["done"] = True
    q.put(("done", json.dumps({"success": success})))


@app.route("/api/run", methods=["POST"])
def run_workflow():
    """Start a forward or inversion workflow in a background thread."""
    data = request.get_json(force=True, silent=True) or {}
    mode = (data.get("mode") or "").lower()
    if mode not in ("forward", "inversion"):
        return jsonify({"error": "mode must be 'forward' or 'inversion'"}), 400

    params = data.get("params") or {}
    output_dir = (data.get("output_dir") or "./output").strip()

    # Convert flat form params to nested JSON config
    try:
        cfg_json = _dict_to_json_cfg(params)
        if not cfg_json["observations"]["files"]:
            return (
                jsonify({"error": "At least one valid observation file path is required"}),
                400,
            )
    except Exception as exc:
        app.logger.error("run_workflow parameter error: %s", exc, exc_info=True)
        return jsonify({"error": "Parameter conversion error. Check server logs."}), 400

    job_id = uuid.uuid4().hex
    q: queue.Queue = queue.Queue()
    _jobs[job_id] = {
        "queue": q,
        "success": None,
        "done": False,
        "stopped": False,
        "mode": mode,
    }

    t = threading.Thread(
        target=_run_workflow_thread,
        args=(job_id, mode, cfg_json, output_dir),
        daemon=True,
    )
    _jobs[job_id]["thread"] = t
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/stream/<job_id>")
def stream_logs(job_id: str):
    """SSE endpoint: streams log lines and a final 'done' event."""
    if job_id not in _jobs:
        return Response("Job not found", status=404)

    job = _jobs[job_id]
    q: queue.Queue = job["queue"]

    def generate():
        while True:
            if job.get("stopped"):
                yield "event: done\ndata: {\"success\": false, \"stopped\": true}\n\n"
                break
            try:
                event_type, payload = q.get(timeout=0.3)
            except queue.Empty:
                # Keep-alive comment
                yield ": ping\n\n"
                continue

            if event_type == "log":
                # Escape newlines so the SSE payload stays on one line
                safe = payload.replace("\r\n", " | ").replace("\n", " | ").replace("\r", "")
                yield f"event: log\ndata: {safe}\n\n"
            elif event_type == "progress":
                yield f"event: progress\ndata: {payload}\n\n"
            elif event_type == "done":
                yield f"event: done\ndata: {payload}\n\n"
                break
            else:
                yield f"event: log\ndata: {payload}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/job/<job_id>/status", methods=["GET"])
def job_status(job_id: str):
    if job_id not in _jobs:
        return jsonify({"error": "Job not found"}), 404
    job = _jobs[job_id]
    return jsonify(
        {
            "job_id": job_id,
            "mode": job.get("mode"),
            "done": job.get("done", False),
            "success": job.get("success"),
            "stopped": job.get("stopped", False),
        }
    )


@app.route("/api/job/<job_id>/stop", methods=["POST"])
def stop_job(job_id: str):
    if job_id not in _jobs:
        return jsonify({"error": "Job not found"}), 404
    _jobs[job_id]["stopped"] = True
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="pyZeeTom Web UI — interactive parameter editor and workflow launcher"
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=5000, help="Bind port (default: 5000)")
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    url = f"http://{args.host}:{args.port}"
    print(f"pyZeeTom Web UI starting at {url}")
    print("Press Ctrl+C to stop.\n")

    _BROWSER_OPEN_DELAY = 1.2  # seconds to wait before opening browser

    if not args.no_browser:
        import webbrowser

        threading.Timer(_BROWSER_OPEN_DELAY, lambda: webbrowser.open(url)).start()

    # Change working directory to project root so relative paths resolve
    os.chdir(_ROOT)

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
