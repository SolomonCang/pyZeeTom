"""Tests for JSON-based configuration loading (config.json support).

These tests verify that:
- readParamsTomog.from_json() correctly reads all parameter groups from a JSON
  config file and produces an object identical in interface to the one produced
  by the positional text constructor.
- readParamsTomog.to_json() serialises all parameters back to a dict / file and
  the resulting JSON round-trips without loss.
- forward_tomography() and inversion_tomography() auto-detect a .json extension
  and use from_json() transparently.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on sys.path so that `core` and `pyzeetom` are found.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.mainFuncs import readParamsTomog  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_config(obs_path: str = "/tmp/dummy_obs.lsd") -> dict:
    """Return a minimal but valid config dict."""
    return {
        "star": {
            "inclination": 45.0,
            "vsini": 30.0,
            "period": 2.0,
            "pOmega": 0.0,
            "mass": 1.2,
            "radius": 1.1,
        },
        "grid": {
            "nRings": 5,
            "Vmax": 0.0,
            "r_out": 2.0,
            "enable_occultation": 0,
        },
        "inversion": {
            "targetForm": "C",
            "targetValue": 1.0,
            "numIterations": 3,
            "test_aim": 1e-2,
        },
        "line_model": {
            "lineAmpConst": 0.5,
            "k_QU": 0.8,
            "enableV": 1,
            "enableQU": 0,
        },
        "initial_model": {
            "initTomogFile": 0,
            "initModelPath": "",
        },
        "fit_flags": {
            "fitBri": 1,
            "fitMag": 0,
            "fitBlos": 1,
            "fitBperp": 0,
            "fitChi": 0,
        },
        "spectral": {
            "spectralResolution": 50000,
            "lineParamFile": "input/lines.txt",
            "velStart": -300.0,
            "velEnd": 300.0,
            "obsFileType": "lsd_pol",
            "polOut": "V",
            "specType": "auto",
        },
        "observations": {
            "jDateRef": 2451000.0,
            "files": [
                {"path": obs_path, "jdate": 2451000.0, "vr": 5.0, "polchannel": "V"},
                {"path": obs_path, "jdate": 2451000.5, "vr": -3.0, "polchannel": "V"},
            ],
        },
    }


# ---------------------------------------------------------------------------
# Tests: from_json
# ---------------------------------------------------------------------------

class TestFromJson:
    """Tests for readParamsTomog.from_json()."""

    def test_star_params(self, tmp_path):
        cfg = _minimal_config()
        json_file = tmp_path / "config.json"
        json_file.write_text(json.dumps(cfg))

        par = readParamsTomog.from_json(str(json_file), verbose=0)

        assert par.inclination == pytest.approx(45.0)
        assert par.vsini == pytest.approx(30.0)
        assert par.period == pytest.approx(2.0)
        assert par.pOmega == pytest.approx(0.0)
        assert par.mass == pytest.approx(1.2)
        assert par.radius == pytest.approx(1.1)

    def test_grid_params(self, tmp_path):
        cfg = _minimal_config()
        json_file = tmp_path / "config.json"
        json_file.write_text(json.dumps(cfg))

        par = readParamsTomog.from_json(str(json_file), verbose=0)

        assert par.nRingsStellarGrid == 5

    def test_inversion_params(self, tmp_path):
        cfg = _minimal_config()
        json_file = tmp_path / "config.json"
        json_file.write_text(json.dumps(cfg))

        par = readParamsTomog.from_json(str(json_file), verbose=0)

        assert par.targetForm == "C"
        assert par.targetValue == pytest.approx(1.0)
        assert par.numIterations == 3
        assert par.test_aim == pytest.approx(1e-2)

    def test_line_model_params(self, tmp_path):
        cfg = _minimal_config()
        json_file = tmp_path / "config.json"
        json_file.write_text(json.dumps(cfg))

        par = readParamsTomog.from_json(str(json_file), verbose=0)

        assert par.lineAmpConst == pytest.approx(0.5)
        assert par.lineKQU == pytest.approx(0.8)
        assert par.lineEnableV == 1
        assert par.lineEnableQU == 0

    def test_fit_flags(self, tmp_path):
        cfg = _minimal_config()
        json_file = tmp_path / "config.json"
        json_file.write_text(json.dumps(cfg))

        par = readParamsTomog.from_json(str(json_file), verbose=0)

        assert par.fitBri == 1
        assert par.fitMag == 0
        assert par.fitBlos == 1
        assert par.fitBperp == 0
        assert par.fitChi == 0

    def test_spectral_params(self, tmp_path):
        cfg = _minimal_config()
        json_file = tmp_path / "config.json"
        json_file.write_text(json.dumps(cfg))

        par = readParamsTomog.from_json(str(json_file), verbose=0)

        assert par.spectralResolution == pytest.approx(50000)
        assert par.velStart == pytest.approx(-300.0)
        assert par.velEnd == pytest.approx(300.0)
        assert par.obsFileType == "lsd_pol"
        assert par.polOut == "V"
        assert par.specType == "auto"
        assert par.lineParamFile == "input/lines.txt"

    def test_observations(self, tmp_path):
        cfg = _minimal_config()
        json_file = tmp_path / "config.json"
        json_file.write_text(json.dumps(cfg))

        par = readParamsTomog.from_json(str(json_file), verbose=0)

        assert par.numObs == 2
        assert par.jDateRef == pytest.approx(2451000.0)
        assert par.velRs[0] == pytest.approx(5.0)
        assert par.velRs[1] == pytest.approx(-3.0)
        assert par.polChannels[0] == "V"

    def test_phases_derived(self, tmp_path):
        """Phases should be automatically derived from jDates and jDateRef."""
        import numpy as np

        cfg = _minimal_config()
        json_file = tmp_path / "config.json"
        json_file.write_text(json.dumps(cfg))

        par = readParamsTomog.from_json(str(json_file), verbose=0)

        expected = (par.jDates - par.jDateRef) / par.period
        np.testing.assert_allclose(par.phases, expected)

    def test_missing_observations_raises(self, tmp_path):
        cfg = _minimal_config()
        cfg["observations"]["files"] = []
        json_file = tmp_path / "config.json"
        json_file.write_text(json.dumps(cfg))

        with pytest.raises(ValueError, match="observations.files"):
            readParamsTomog.from_json(str(json_file), verbose=0)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            readParamsTomog.from_json(str(tmp_path / "nonexistent.json"), verbose=0)

    def test_defaults_applied_when_groups_missing(self, tmp_path):
        """from_json should apply sensible defaults for optional groups."""
        cfg = {
            "observations": {
                "jDateRef": 2451000.0,
                "files": [
                    {"path": "/tmp/x.lsd", "jdate": 2451000.0, "vr": 0.0}
                ],
            }
        }
        json_file = tmp_path / "config.json"
        json_file.write_text(json.dumps(cfg))

        par = readParamsTomog.from_json(str(json_file), verbose=0)
        assert par.numObs == 1
        assert par.inclination == pytest.approx(60.0)   # default
        assert par.nRingsStellarGrid == 10              # default


# ---------------------------------------------------------------------------
# Tests: to_json
# ---------------------------------------------------------------------------

class TestToJson:
    """Tests for readParamsTomog.to_json()."""

    def test_returns_dict(self, tmp_path):
        cfg = _minimal_config()
        json_file = tmp_path / "config.json"
        json_file.write_text(json.dumps(cfg))

        par = readParamsTomog.from_json(str(json_file), verbose=0)
        result = par.to_json(verbose=0)

        assert isinstance(result, dict)
        for key in ("star", "grid", "inversion", "line_model",
                    "initial_model", "fit_flags", "spectral", "observations"):
            assert key in result

    def test_writes_file(self, tmp_path):
        cfg = _minimal_config()
        in_json = tmp_path / "config.json"
        in_json.write_text(json.dumps(cfg))

        par = readParamsTomog.from_json(str(in_json), verbose=0)

        out_json = tmp_path / "out_config.json"
        par.to_json(str(out_json), verbose=0)

        assert out_json.exists()
        loaded = json.loads(out_json.read_text())
        assert loaded["star"]["inclination"] == pytest.approx(45.0)

    def test_round_trip(self, tmp_path):
        """from_json -> to_json -> from_json must preserve all scalar params."""
        cfg = _minimal_config()
        json_file1 = tmp_path / "config1.json"
        json_file1.write_text(json.dumps(cfg))

        par1 = readParamsTomog.from_json(str(json_file1), verbose=0)

        json_file2 = tmp_path / "config2.json"
        par1.to_json(str(json_file2), verbose=0)

        par2 = readParamsTomog.from_json(str(json_file2), verbose=0)

        assert par2.inclination == pytest.approx(par1.inclination)
        assert par2.vsini == pytest.approx(par1.vsini)
        assert par2.period == pytest.approx(par1.period)
        assert par2.pOmega == pytest.approx(par1.pOmega)
        assert par2.nRingsStellarGrid == par1.nRingsStellarGrid
        assert par2.targetForm == par1.targetForm
        assert par2.numIterations == par1.numIterations
        assert par2.lineAmpConst == pytest.approx(par1.lineAmpConst)
        assert par2.fitBri == par1.fitBri
        assert par2.spectralResolution == pytest.approx(par1.spectralResolution)
        assert par2.velStart == pytest.approx(par1.velStart)
        assert par2.numObs == par1.numObs

    def test_no_file_written_when_none(self, tmp_path):
        cfg = _minimal_config()
        json_file = tmp_path / "config.json"
        json_file.write_text(json.dumps(cfg))

        par = readParamsTomog.from_json(str(json_file), verbose=0)
        result = par.to_json(json_file=None, verbose=0)

        # Only the input file should exist, no new files
        assert isinstance(result, dict)
        assert len(list(tmp_path.glob("*.json"))) == 1


# ---------------------------------------------------------------------------
# Tests: JSON detection in high-level API
# ---------------------------------------------------------------------------

class TestAutoDetectJson:
    """Verify that forward_tomography() / inversion_tomography() route .json
    files through from_json transparently."""

    def test_forward_tomography_accepts_json_extension(self, tmp_path, monkeypatch):
        """forward_tomography should call from_json when param_file ends in .json."""
        import pyzeetom.tomography as tomo

        calls = []

        original_from_json = readParamsTomog.from_json

        @classmethod  # type: ignore[misc]
        def mock_from_json(cls, path, verbose=1):
            calls.append(path)
            return original_from_json.__func__(cls, path, verbose=0)

        monkeypatch.setattr(
            "core.mainFuncs.readParamsTomog.from_json", mock_from_json
        )

        cfg = _minimal_config()
        json_file = tmp_path / "config.json"
        json_file.write_text(json.dumps(cfg))

        # We only check that from_json was invoked; we don't need the full
        # pipeline to run (obs files don't exist), so catch the downstream error.
        try:
            tomo.forward_tomography(param_file=str(json_file), verbose=0)
        except Exception:
            pass

        assert any(str(json_file) in str(c) for c in calls), (
            "from_json was not called for a .json param_file"
        )

    def test_inversion_tomography_accepts_json_extension(self, tmp_path, monkeypatch):
        """inversion_tomography should call from_json when param_file ends in .json."""
        import pyzeetom.tomography as tomo

        calls = []

        original_from_json = readParamsTomog.from_json

        @classmethod  # type: ignore[misc]
        def mock_from_json(cls, path, verbose=1):
            calls.append(path)
            return original_from_json.__func__(cls, path, verbose=0)

        monkeypatch.setattr(
            "core.mainFuncs.readParamsTomog.from_json", mock_from_json
        )

        cfg = _minimal_config()
        json_file = tmp_path / "config.json"
        json_file.write_text(json.dumps(cfg))

        try:
            tomo.inversion_tomography(param_file=str(json_file), verbose=0)
        except Exception:
            pass

        assert any(str(json_file) in str(c) for c in calls), (
            "from_json was not called for a .json param_file"
        )
