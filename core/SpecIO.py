from __future__ import annotations
# -----------------------------------------------------------------------------
# Grid (diskGrid) structure IO for tomography
# -----------------------------------------------------------------------------
import json
import numpy as np


def write_model_grid(filename: str, grid, meta: dict = None):
    """
    Save diskGrid structure to file, supporting all pixel attributes (r, phi, area, ring_id, phi_id, ...).
    meta is optional, written to file header (json).
    """
    meta = meta or {}
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('# TOMOGRID 1.0\n')
        f.write('#META ' + json.dumps(meta, ensure_ascii=False) + '\n')
        cols = ["r", "phi", "area", "ring_id", "phi_id"]
        for attr in ["dr_cell", "dphi_cell"]:
            if hasattr(grid, attr):
                cols.append(attr)
        f.write('# ' + ' '.join(cols) + '\n')
        for i in range(grid.numPoints):
            vals = [getattr(grid, k)[i] for k in cols]
            f.write(' '.join(f'{v:.8g}' if isinstance(v, float) else str(v)
                             for v in vals) + '\n')


def load_model_grid(filename: str):
    """
    Read diskGrid structure file, return grid-like object and meta.
    """
    from types import SimpleNamespace as _NS
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    meta = {}
    cols = []
    data = []
    for ln in lines:
        if ln.startswith('#META'):
            meta = json.loads(ln[5:].strip())
        elif ln.startswith('#') and 'TOMOGRID' in ln:
            continue
        elif ln.startswith('#'):
            cols = ln[1:].strip().split()
        elif ln.strip():
            data.append([float(x) for x in ln.strip().split()])
    arr = np.array(data)
    grid = _NS()
    for i, k in enumerate(cols):
        grid.__setattr__(k, arr[:, i])
    grid.numPoints = arr.shape[0]
    return grid, meta


"""SpecIO.py — Spectral IO utilities

- Input: Supports LSD/spec (I/pol/simple) spectral data as ObservationProfile
- Output: Supports writing model spectra (velocity or wavelength domain) based on model integration results

Compatible with old interfaces (loadObsProfile/obsProfSetInRange/getObservedEW).
"""

import io
import pandas as pd
from typing import Optional, Tuple, List, Dict

# -----------------------------------------------------------------------------
# Core parsing utilities (from former readObs.py)
# -----------------------------------------------------------------------------


def parse_text_to_df(text: str) -> Optional[pd.DataFrame]:
    buf = io.StringIO(text)
    lines = buf.getvalue().splitlines()

    non_comment = [ln for ln in lines if not ln.lstrip().startswith("*")]

    start_idx = None
    for i, ln in enumerate(non_comment):
        parts = ln.strip().split()
        nums = 0
        for p in parts:
            try:
                float(p)
                nums += 1
            except Exception:
                pass
        if nums >= 2:
            start_idx = i
            break

    try:
        if start_idx is not None:
            first_line = non_comment[start_idx].strip().split()
            if len(first_line) == 2:
                try:
                    float(first_line[0])
                    float(first_line[1])
                    data_text = "\n".join(non_comment[start_idx + 2:])
                except ValueError:
                    data_text = "\n".join(non_comment[start_idx:])
            else:
                data_text = "\n".join(non_comment[start_idx:])

            df = pd.read_csv(io.StringIO(data_text),
                             sep=r"\s+",
                             header=None,
                             engine="python")
            return df
        else:
            df = pd.read_csv(io.StringIO(text),
                             sep=r"\s+",
                             comment='*',
                             skiprows=2,
                             header=None,
                             engine="python")
            return df
    except Exception as e:
        print(f"Warning: parse_text_to_df failed with error: {e}")
        return None


def _assign_columns_by_type(df: pd.DataFrame, file_type_hint: str):
    ncol = df.shape[1]

    if file_type_hint == "spec_pol":
        if ncol != 6:
            return None, f"Spec (pol) expects 6 columns, but got {ncol}"
        df = df.copy()
        df.columns = ["Wav", "Int", "Pol", "Null1", "Null2", "sigma_int"]
        x_col = "Wav"
    elif file_type_hint == "spec_i":
        if ncol != 3:
            return None, f"Spec (I) expects 3 columns, but got {ncol}"
        df = df.copy()
        df.columns = ["Wav", "Int", "sigma_int"]
        x_col = "Wav"
    elif file_type_hint == "spec_i_simple":
        if ncol != 2:
            return None, f"Spec (I, simple) expects 2 columns, but got {ncol}"
        df = df.copy()
        df.columns = ["Wav", "Int"]
        x_col = "Wav"
    elif file_type_hint == "lsd_pol":
        if ncol != 7:
            return None, f"LSD (pol) expects 7 columns, but got {ncol}"
        df = df.copy()
        df.columns = [
            "RV", "Int", "sigma_int", "Pol", "sigma_pol", "Null1",
            "sigma_null1"
        ]
        x_col = "RV"
    elif file_type_hint == "lsd_i":
        if ncol != 3:
            return None, f"LSD (I) expects 3 columns, but got {ncol}"
        df = df.copy()
        df.columns = ["RV", "Int", "sigma_int"]
        x_col = "RV"
    elif file_type_hint == "lsd_i_simple":
        if ncol != 2:
            return None, f"LSD (I, simple) expects 2 columns, but got {ncol}"
        df = df.copy()
        df.columns = ["RV", "Int"]
        x_col = "RV"
    else:
        return None, f"Unknown file type: {file_type_hint}"
    return (df, x_col), None


def _heuristic_guess(df: pd.DataFrame):
    if df is None or df.empty:
        return None, None
    df = df.dropna(axis=1, how="all")
    ncol = df.shape[1]
    if ncol not in (2, 3, 6, 7):
        return None, None

    if ncol == 6:
        # Convention: 6 columns can only be wavelength domain spec_pol (Wav Int Pol Null1 Null2 sigma_int)
        return "spec_pol", _assign_columns_by_type(df, "spec_pol")[0]
    if ncol == 7:
        return "lsd_pol", _assign_columns_by_type(df, "lsd_pol")[0]

    if ncol == 3:
        x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        if x.isna().all():
            return None, None
        xmin, xmax = x.min(), x.max()
        is_wav = (xmin >= 200) and (xmax <= 5000)
        is_rv = (xmin < 0) or (abs(xmin) <= 10000 and abs(xmax) <= 10000
                               and xmax < 200)
        if is_wav and not is_rv:
            return "spec_i", _assign_columns_by_type(df, "spec_i")[0]
        if is_rv and not is_wav:
            return "lsd_i", _assign_columns_by_type(df, "lsd_i")[0]

    if ncol == 2:
        x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        if x.isna().all():
            return None, None
        xmin, xmax = x.min(), x.max()
        is_wav = (xmin >= 200) and (xmax <= 5000)
        is_rv = (xmin < 0) or (abs(xmin) <= 10000 and abs(xmax) <= 10000
                               and xmax < 200)
        if is_wav and not is_rv:
            return "spec_i_simple", _assign_columns_by_type(
                df, "spec_i_simple")[0]
        if is_rv and not is_wav:
            return "lsd_i_simple", _assign_columns_by_type(df,
                                                           "lsd_i_simple")[0]

    for ft in ["spec_i", "lsd_i", "spec_i_simple", "lsd_i_simple"]:
        res, err = _assign_columns_by_type(df, ft)
        if not err:
            return ft, res
    return None, None


def detect_and_assign_columns(df: pd.DataFrame, file_type_hint: str = "auto"):
    if file_type_hint == "auto":
        guessed_type, res = _heuristic_guess(df)
        if guessed_type is None or res is None:
            return None, "Cannot automatically determine data type, please specify file type manually"
        (df_named, x_col) = res
        return (df_named, x_col, guessed_type), None
    else:
        res, err = _assign_columns_by_type(df, file_type_hint)
        if err:
            return None, err
        df_named, x_col = res
        return (df_named, x_col, file_type_hint), None


# -----------------------------------------------------------------------------
# High-level observation data structures
# -----------------------------------------------------------------------------


class ObservationProfile:

    def __init__(self,
                 wl: np.ndarray,
                 specI: np.ndarray,
                 specIsig: Optional[np.ndarray] = None,
                 specV: Optional[np.ndarray] = None,
                 specVsig: Optional[np.ndarray] = None,
                 specQ: Optional[np.ndarray] = None,
                 specU: Optional[np.ndarray] = None,
                 null: Optional[np.ndarray] = None,
                 profile_type: str = "unknown",
                 pol_channel: str = "V"):
        self.wl = np.asarray(wl, dtype=float)
        self.specI = np.asarray(specI, dtype=float)
        self.specIsig = np.asarray(
            specIsig, dtype=float
        ) if specIsig is not None else np.ones_like(specI) * 1e-4
        self.specV = np.asarray(
            specV, dtype=float) if specV is not None else np.zeros_like(specI)
        self.specVsig = np.asarray(
            specVsig, dtype=float
        ) if specVsig is not None else np.ones_like(specI) * 1e-5
        self.specQ = np.asarray(
            specQ, dtype=float) if specQ is not None else np.zeros_like(specI)
        self.specU = np.asarray(
            specU, dtype=float) if specU is not None else np.zeros_like(specI)
        self.null = np.asarray(
            null, dtype=float) if null is not None else np.zeros_like(specI)
        self.profile_type = profile_type
        # New: Record polarization channel (I/V/Q/U)
        self.pol_channel = pol_channel.upper() if pol_channel else "V"
        # Mark available polarization components
        comps = {'I'}
        if self.specV is not None and np.any(self.specV != 0.0):
            comps.add('V')
        if self.specQ is not None and np.any(self.specQ != 0.0):
            comps.add('Q')
        if self.specU is not None and np.any(self.specU != 0.0):
            comps.add('U')
        self.components_present = comps
        self.hasV = 'V' in comps
        self.hasQ = 'Q' in comps
        self.hasU = 'U' in comps

    def scaleIsig(self, scale_factor: float):
        self.specIsig *= scale_factor


def loadObsProfile(filename: str,
                   file_type: str = "auto",
                   vel_start: Optional[float] = None,
                   vel_end: Optional[float] = None,
                   vel_shift: float = 0.0,
                   pol_channel: str = "V") -> Optional[ObservationProfile]:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None

    df_raw = parse_text_to_df(text)
    if df_raw is None:
        print(f"Failed to parse {filename} as tabular data")
        return None

    result, err = detect_and_assign_columns(df_raw, file_type)
    if err:
        print(f"Error detecting columns in {filename}: {err}")
        return None

    df, x_col, resolved_type = result

    wl = df[x_col].values
    specI = df["Int"].values if "Int" in df.columns else df.iloc[:, 1].values

    if resolved_type.startswith("lsd"):
        wl = wl + vel_shift

    if vel_start is not None and vel_end is not None:
        if resolved_type.startswith("lsd"):
            mask = (wl >= vel_start) & (wl <= vel_end)
            wl = wl[mask]
            specI = specI[mask]
            df = df[mask].copy()

    specIsig = df["sigma_int"].values if "sigma_int" in df.columns else None
    specV = df["Pol"].values if "Pol" in df.columns else (
        df["V"].values if "V" in df.columns else None)
    specVsig = df["sigma_pol"].values if "sigma_pol" in df.columns else None
    null = df["Null1"].values if "Null1" in df.columns else (
        df["Null"].values if "Null" in df.columns else None)

    return ObservationProfile(wl=wl,
                              specI=specI,
                              specIsig=specIsig,
                              specV=specV,
                              specVsig=specVsig,
                              null=null,
                              profile_type=resolved_type,
                              pol_channel=pol_channel)


def obsProfSetInRange(
        fnames: List[str],
        vel_start: float,
        vel_end: float,
        vel_shifts: Optional[np.ndarray] = None,
        file_type: str = "auto",
        pol_channels: Optional[List[str]] = None,
        phases: Optional[np.ndarray] = None) -> List[ObservationProfile]:
    if vel_shifts is None:
        vel_shifts = np.zeros(len(fnames))
    if pol_channels is None:
        pol_channels = ["V"] * len(fnames)

    obsSet = []
    for i, fname in enumerate(fnames):
        pol_chan = pol_channels[i] if i < len(pol_channels) else "V"
        obs = loadObsProfile(fname,
                             file_type=file_type,
                             pol_channel=pol_chan,
                             vel_start=vel_start,
                             vel_end=vel_end,
                             vel_shift=vel_shifts[i])
        if obs is not None:
            # Inject phase if available
            if phases is not None and i < len(phases):
                obs.phase = float(phases[i])
            obsSet.append(obs)
        else:
            print(f"Warning: failed to load {fname}, skipping")
    return obsSet


def getObservedEW(obsSet: List[ObservationProfile],
                  lineData,
                  verbose: int = 1) -> float:
    total_ew = 0.0
    count = 0
    for obs in obsSet:
        if obs.profile_type.startswith("lsd"):
            ew = np.trapz(1.0 - obs.specI, obs.wl)
        else:
            ew = np.trapz(1.0 - obs.specI, obs.wl)
        total_ew += ew
        count += 1
    mean_ew = total_ew / count if count > 0 else 0.0
    if verbose:
        print(f"Mean equivalent width: {mean_ew:.6f}")
    return mean_ew


# -----------------------------------------------------------------------------
# Model spectrum output
# -----------------------------------------------------------------------------


def write_model_spectrum(filename: str,
                         x: np.ndarray,
                         Iprof: np.ndarray,
                         V: Optional[np.ndarray] = None,
                         Q: Optional[np.ndarray] = None,
                         U: Optional[np.ndarray] = None,
                         sigmaI: Optional[np.ndarray] = None,
                         fmt: str = "lsd",
                         header: Optional[Dict[str, str]] = None,
                         pol_channel: str = 'V',
                         include_null: bool = False,
                         file_type_hint: Optional[str] = None) -> None:
    """Write model spectrum to file.

    Default output:
      - fmt="lsd": Velocity domain (km/s) columns: RV Int sigma_int V Q U or extended 7 columns LSD(pol) format
      - fmt="spec": Wavelength domain (nm) columns: Wav Int sigma_int V Q U

        To ensure consistency with parsing types, it is recommended to explicitly pass file_type_hint (e.g., 'spec_pol') when a specific structure is needed.
    """
    x = np.asarray(x, dtype=float)
    Iprof = np.asarray(Iprof, dtype=float)
    V = np.zeros_like(Iprof) if V is None else np.asarray(V, dtype=float)
    Q = np.zeros_like(Iprof) if Q is None else np.asarray(Q, dtype=float)
    U = np.zeros_like(Iprof) if U is None else np.asarray(U, dtype=float)
    sigmaI = np.zeros_like(Iprof) if sigmaI is None else np.asarray(
        sigmaI, dtype=float)

    name_x = "RV" if fmt == "lsd" else "Wav"
    pol_channel = (pol_channel or 'V').upper()

    # ------------------------------------------------------------------
    # Automatically infer output format (if file_type_hint is not explicitly specified)
    # ------------------------------------------------------------------
    if file_type_hint is None:
        # Automatically select appropriate output format based on pol_channel and fmt
        if pol_channel == 'I':
            # I channel: output spec_i or lsd_i format (3 columns)
            file_type_hint = 'lsd_i' if fmt == 'lsd' else 'spec_i'
        else:
            # V/Q/U channel: output spec_pol or lsd_pol format
            file_type_hint = 'lsd_pol' if fmt == 'lsd' else 'spec_pol'

    # ------------------------------------------------------------------
    # Explicit file type output (priority higher than fmt/force_input_structure)
    # Supports: spec_pol, spec_i, spec_i_simple, lsd_pol, lsd_i, lsd_i_simple
    # ------------------------------------------------------------------
    if file_type_hint is not None:
        fth = file_type_hint.lower()
        if fth == 'spec_pol':
            # Wav(nm) Int Pol Null1 Null2 sigma_int
            if pol_channel == 'V':
                pol = V
            elif pol_channel == 'Q':
                pol = Q
            else:
                pol = U
            pol = np.zeros_like(Iprof) if pol is None else np.asarray(
                pol, dtype=float)
            sigma_int = np.zeros_like(Iprof) if sigmaI is None else sigmaI
            null1 = np.zeros_like(Iprof)
            null2 = np.zeros_like(Iprof)
            cols = ["Wav(nm)", "Int", "Pol", "Null1", "Null2", "sigma_int"]
            with open(filename, 'w', encoding='utf-8') as f:
                if header:
                    for k, v in header.items():
                        f.write(f"# {k}: {v}\n")
                f.write('# ' + ' '.join(cols) + '\n')
                for i in range(x.size):
                    f.write(
                        f"{x[i]:.6f} {Iprof[i]:.8e} {pol[i]:.8e} {null1[i]:.1f} {null2[i]:.1f} {sigma_int[i]:.3e}\n"
                    )
            return
        elif fth == 'spec_i':
            # Wav Int sigma_int
            sigma_int = np.zeros_like(Iprof) if sigmaI is None else sigmaI
            cols = ["Wav", "Int", "sigma_int"]
            with open(filename, 'w', encoding='utf-8') as f:
                if header:
                    for k, v in header.items():
                        f.write(f"# {k}: {v}\n")
                f.write('# ' + ' '.join(cols) + '\n')
                for i in range(x.size):
                    f.write(f"{x[i]:.6f} {Iprof[i]:.8e} {sigma_int[i]:.3e}\n")
            return
        elif fth == 'spec_i_simple':
            # Wav Int
            cols = ["Wav", "Int"]
            with open(filename, 'w', encoding='utf-8') as f:
                if header:
                    for k, v in header.items():
                        f.write(f"# {k}: {v}\n")
                f.write('# ' + ' '.join(cols) + '\n')
                for i in range(x.size):
                    f.write(f"{x[i]:.6f} {Iprof[i]:.8e}\n")
            return
        elif fth == 'lsd_pol':
            # RV Int sigma_int Pol sigma_pol Null1 sigma_null1
            if pol_channel == 'V':
                pol = V
            elif pol_channel == 'Q':
                pol = Q
            elif pol_channel == 'U':
                pol = U
            pol = np.zeros_like(Iprof) if pol is None else np.asarray(
                pol, dtype=float)
            sigma_pol = np.zeros_like(Iprof) if sigmaI is None else sigmaI
            null1 = np.zeros_like(Iprof)
            sigma_null1 = sigma_pol
            cols = [
                "RV", "Int", "sigma_int", "Pol", "sigma_pol", "Null1",
                "sigma_null1"
            ]
            with open(filename, 'w', encoding='utf-8') as f:
                if header:
                    for k, v in header.items():
                        f.write(f"# {k}: {v}\n")
                f.write('# ' + ' '.join(cols) + '\n')
                for i in range(x.size):
                    f.write(
                        f"{x[i]:.6f} {Iprof[i]:.8e} {sigmaI[i]:.3e} {pol[i]:.8e} {sigma_pol[i]:.3e} {null1[i]:.8e} {sigma_null1[i]:.3e}\n"
                    )
            return
        elif fth == 'lsd_i':
            # RV Int sigma_int （严格 3 列）
            sigma_int = np.zeros_like(Iprof) if sigmaI is None else sigmaI
            cols = ["RV", "Int", "sigma_int"]
            with open(filename, 'w', encoding='utf-8') as f:
                if header:
                    for k, v in header.items():
                        f.write(f"# {k}: {v}\n")
                f.write('# ' + ' '.join(cols) + '\n')
                for i in range(x.size):
                    f.write(f"{x[i]:.6f} {Iprof[i]:.8e} {sigma_int[i]:.3e}\n")
            return
        elif fth == 'lsd_i_simple':
            # RV Int
            cols = ["RV", "Int"]
            with open(filename, 'w', encoding='utf-8') as f:
                if header:
                    for k, v in header.items():
                        f.write(f"# {k}: {v}\n")
                f.write('# ' + ' '.join(cols) + '\n')
                for i in range(x.size):
                    f.write(f"{x[i]:.6f} {Iprof[i]:.8e}\n")
            return

    # Deprecated: No longer force matching input file structure; please use file_type_hint to specify output structure.

    if fmt == 'lsd' and pol_channel in ('V', 'Q', 'U') and include_null:
        # Write LSD(pol) 7-column format: RV Int sigma_int Pol sigma_pol Null1 sigma_null1
        if pol_channel == 'V':
            pol = V if V is not None else np.zeros_like(Iprof)
        elif pol_channel == 'Q':
            pol = Q if Q is not None else np.zeros_like(Iprof)
        else:
            pol = U if U is not None else np.zeros_like(Iprof)
        sigma_pol = sigmaI if sigmaI is not None else np.zeros_like(Iprof)
        null1 = np.zeros_like(Iprof)
        sigma_null1 = sigma_pol
        cols = [
            name_x, "Int", "sigma_int", "Pol", "sigma_pol", "Null1",
            "sigma_null1"
        ]
        with open(filename, "w", encoding="utf-8") as f:
            if header:
                for k, v in header.items():
                    f.write(f"# {k}: {v}\n")
            f.write("# " + " ".join(cols) + "\n")
            for i in range(x.size):
                f.write(
                    f"{x[i]:.6f} {Iprof[i]:.8e} {sigmaI[i]:.3e} {pol[i]:.8e} {sigma_pol[i]:.3e} {null1[i]:.8e} {sigma_null1[i]:.3e}\n"
                )
    else:
        # Generic format: RV/Wav Int sigma_int V Q U (Always write V/Q/U columns, fill with 0 if missing)
        cols = [name_x, "Int", "sigma_int", "V", "Q", "U"]
        Vw = np.zeros_like(Iprof) if V is None else V
        Qw = np.zeros_like(Iprof) if Q is None else Q
        Uw = np.zeros_like(Iprof) if U is None else U
        with open(filename, "w", encoding="utf-8") as f:
            if header:
                for k, v in header.items():
                    f.write(f"# {k}: {v}\n")
            f.write("# " + " ".join(cols) + "\n")
            for i in range(x.size):
                f.write(
                    f"{x[i]:.6f} {Iprof[i]:.8e} {sigmaI[i]:.3e} {Vw[i]:.8e} {Qw[i]:.8e} {Uw[i]:.8e}\n"
                )


def save_results_series(results: List[Tuple[np.ndarray, np.ndarray,
                                            np.ndarray]],
                        basepath: str = "model_phase",
                        fmt: str = "lsd") -> List[str]:
    """
    Save tomography.main result list [(x, I, V), ...] to multiple files.
    Return list of written file paths.
    """
    paths = []
    for i, (x, I, V) in enumerate(results):
        ext = 'lsd' if fmt == 'lsd' else 'spec'
        fname = f"{basepath}_{i:02d}.{ext}"
        header = {"phase_index": str(i)}
        write_model_spectrum(fname, x, I, V=V, fmt=fmt, header=header)
        paths.append(fname)
    return paths


def save_model_spectra_with_polchannel(
    phase_results,
    pol_channel="I",
    output_dir="output/outModelSpec",
    file_type_hint="spec",
    verbose=0,
):
    """Save synthesized spectra, selecting Stokes component by pol_channel.

    Save synthesized spectra for each phase in phase_results as files according to the specified polarization channel.
    One file per phase:
    phase_{phase_index:04d}_{HJD}_{VRinfo}_{channel}.{ext}

    Parameters
    ----------
    phase_results : list of dict
        List of results for each phase, each item is a dictionary containing:
        - 'phase': int, phase index
        - 'wl': np.ndarray, wavelength
        - 'I': np.ndarray, Stokes I
        - 'V': np.ndarray or None, Stokes V
        - 'Q': np.ndarray or None, Stokes Q
        - 'U': np.ndarray or None, Stokes U
        - 'sigma': np.ndarray, error
        - 'HJD': float, Heliocentric Julian Date
        - 'VRinfo': str, velocity info string
    pol_channel : str, default="I"
        Polarization channel selection, options are "I", "V", "Q", "U"
    output_dir : str, default="output/outModelSpec"
        Output directory path
    file_type_hint : str, default="spec"
        File format hint, passed to write_model_spectrum
        (supports "spec_pol", "spec_i", "lsd_pol", "lsd_i", etc.)
    verbose : int, default=0
        Verbosity level (0=silent, 1=normal, 2=detailed)

    Returns
    -------
    list of str
        List of written file paths
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Map pol_channel to data fields
    channel_map = {"I": "I", "V": "V", "Q": "Q", "U": "U"}
    if pol_channel not in channel_map:
        raise ValueError(
            f"pol_channel must be 'I', 'V', 'Q', or 'U', got '{pol_channel}'")

    paths = []
    for res in phase_results:
        phase_idx = res.get("phase", 0)
        wl = res.get("wl")
        sigma = res.get("sigma")
        HJD = res.get("HJD", 0.0)
        VRinfo = res.get("VRinfo", "VRp0p00")

        if wl is None:
            if verbose:
                print(
                    f"[SpecIO] Skipping phase {phase_idx}: No wavelength data")
            continue

        # Select specified Stokes component
        if pol_channel == "I":
            data = res.get("I")
        elif pol_channel == "V":
            data = res.get("V")
        elif pol_channel == "Q":
            data = res.get("Q")
        elif pol_channel == "U":
            data = res.get("U")
        else:
            data = None

        if data is None:
            if verbose:
                print(
                    f"[SpecIO] Phase {phase_idx} missing {pol_channel} data, skipping"
                )
            continue

        # Construct filename
        HJD_str = f"{HJD:.3f}".replace(".", "p")
        filename = (
            f"phase_{phase_idx:04d}_HJD{HJD_str}_{VRinfo}_{pol_channel}.spec")
        filepath = output_path / filename

        # Prepare other Stokes components (None if missing)
        V_data = res.get("V") if pol_channel != "V" else None
        Q_data = res.get("Q") if pol_channel != "Q" else None
        U_data = res.get("U") if pol_channel != "U" else None

        # Call write_model_spectrum to save
        header = {
            "phase": str(phase_idx),
            "HJD": str(HJD),
            "pol_channel": pol_channel,
            "VRinfo": VRinfo,
        }
        try:
            write_model_spectrum(
                str(filepath),
                wl,
                data,
                V=V_data,
                Q=Q_data,
                U=U_data,
                sigmaI=sigma,
                fmt=file_type_hint,
                header=header,
            )
            paths.append(str(filepath))
            if verbose:
                print(
                    f"[SpecIO] Saved phase {phase_idx} {pol_channel} channel to {filename}"
                )
        except Exception as e:
            if verbose:
                print(f"[SpecIO] Failed to save phase {phase_idx}: {e}")
            continue

    if verbose:
        print(f"[SpecIO] Saved {len(paths)} spectrum files")

    return paths
