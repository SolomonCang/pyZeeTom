"""SpecIO.py — Spectral IO utilities

- 读入：支持 LSD/spec (I/pol/simple) 光谱数据为 ObservationProfile
- 写出：支持根据模型积分结果写出模型光谱（速度或波长域）

兼容老接口（loadObsProfile/obsProfSetInRange/getObservedEW）。
"""

from __future__ import annotations

import io
import numpy as np
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
            return None, f"Spec (pol) 期望6列，但有 {ncol} 列"
        df = df.copy()
        df.columns = ["Wav", "Int", "Pol", "Null1", "Null2", "sigma_int"]
        x_col = "Wav"
    elif file_type_hint == "spec_i":
        if ncol != 3:
            return None, f"Spec (I) 期望3列，但有 {ncol} 列"
        df = df.copy()
        df.columns = ["Wav", "Int", "sigma_int"]
        x_col = "Wav"
    elif file_type_hint == "spec_i_simple":
        if ncol != 2:
            return None, f"Spec (I, simple) 期望2列，但有 {ncol} 列"
        df = df.copy()
        df.columns = ["Wav", "Int"]
        x_col = "Wav"
    elif file_type_hint == "lsd_pol":
        if ncol != 7:
            return None, f"LSD (pol) 期望7列，但有 {ncol} 列"
        df = df.copy()
        df.columns = [
            "RV", "Int", "sigma_int", "Pol", "sigma_pol", "Null1",
            "sigma_null1"
        ]
        x_col = "RV"
    elif file_type_hint == "lsd_i":
        if ncol != 3:
            return None, f"LSD (I) 期望3列，但有 {ncol} 列"
        df = df.copy()
        df.columns = ["RV", "Int", "sigma_int"]
        x_col = "RV"
    elif file_type_hint == "lsd_i_simple":
        if ncol != 2:
            return None, f"LSD (I, simple) 期望2列，但有 {ncol} 列"
        df = df.copy()
        df.columns = ["RV", "Int"]
        x_col = "RV"
    else:
        return None, f"未知文件类型: {file_type_hint}"
    return (df, x_col), None


def _heuristic_guess(df: pd.DataFrame):
    if df is None or df.empty:
        return None, None
    df = df.dropna(axis=1, how="all")
    ncol = df.shape[1]
    if ncol not in (2, 3, 6, 7):
        return None, None

    if ncol == 6:
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
            return None, "无法自动判别数据类型，请手动指定文件类型"
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
                 profile_type: str = "unknown"):
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

    def scaleIsig(self, scale_factor: float):
        self.specIsig *= scale_factor


def loadObsProfile(filename: str,
                   file_type: str = "auto",
                   vel_start: Optional[float] = None,
                   vel_end: Optional[float] = None,
                   vel_shift: float = 0.0) -> Optional[ObservationProfile]:
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
                              profile_type=resolved_type)


def obsProfSetInRange(fnames: List[str],
                      vel_start: float,
                      vel_end: float,
                      vel_shifts: Optional[np.ndarray] = None,
                      file_type: str = "auto") -> List[ObservationProfile]:
    if vel_shifts is None:
        vel_shifts = np.zeros(len(fnames))

    obsSet = []
    for i, fname in enumerate(fnames):
        obs = loadObsProfile(fname,
                             file_type=file_type,
                             vel_start=vel_start,
                             vel_end=vel_end,
                             vel_shift=vel_shifts[i])
        if obs is not None:
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
                         header: Optional[Dict[str, str]] = None) -> None:
    """
    将模型光谱写入文件。
    - fmt="lsd": 速度域 (km/s) 列：RV Int [sigma_int] [V] [Q] [U]
    - fmt="spec": 波长域 (nm) 列：Wav Int [sigma_int] [V] [Q] [U]
    """
    x = np.asarray(x, dtype=float)
    Iprof = np.asarray(Iprof, dtype=float)
    V = np.zeros_like(Iprof) if V is None else np.asarray(V, dtype=float)
    Q = np.zeros_like(Iprof) if Q is None else np.asarray(Q, dtype=float)
    U = np.zeros_like(Iprof) if U is None else np.asarray(U, dtype=float)
    sigmaI = np.zeros_like(Iprof) if sigmaI is None else np.asarray(
        sigmaI, dtype=float)

    name_x = "RV" if fmt == "lsd" else "Wav"
    cols = [name_x, "Int", "sigma_int", "V", "Q", "U"]

    with open(filename, "w", encoding="utf-8") as f:
        if header:
            for k, v in header.items():
                f.write(f"# {k}: {v}\n")
        f.write("# " + " ".join(cols) + "\n")
        for i in range(x.size):
            f.write(
                f"{x[i]:.6f} {Iprof[i]:.8e} {sigmaI[i]:.3e} {V[i]:.8e} {Q[i]:.8e} {U[i]:.8e}\n"
            )


def save_results_series(results: List[Tuple[np.ndarray, np.ndarray,
                                            np.ndarray]],
                        basepath: str = "model_phase",
                        fmt: str = "lsd") -> List[str]:
    """
    将 tomography.main 的结果列表 [(x, I, V), ...] 保存为多文件。
    返回写出的文件路径列表。
    """
    paths = []
    for i, (x, I, V) in enumerate(results):
        ext = 'lsd' if fmt == 'lsd' else 'spec'
        fname = f"{basepath}_{i:02d}.{ext}"
        header = {"phase_index": str(i)}
        write_model_spectrum(fname, x, I, V=V, fmt=fmt, header=header)
        paths.append(fname)
    return paths
