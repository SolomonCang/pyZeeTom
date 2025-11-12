"""readObs.py — Observation data reader for spectroscopic profiles

This module provides functions to read and parse spectral/LSD profile data
from text files. It supports multiple formats:
- LSD profiles (pol/I/simple): velocity-space profiles
- Spec profiles (pol/I/simple): wavelength-space spectra

Adapted from SpectrumViewer.py parse functions with enhancements for
ZDI/tomography workflows.
"""

import io
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


# -----------------------------------------------------------------------------
# Core parsing utilities
# -----------------------------------------------------------------------------

def parse_text_to_df(text: str) -> Optional[pd.DataFrame]:
    """
    智能解析文本为 DataFrame：
    - 空白分隔
    - 自动探测并跳过前导注释/元信息行
    - 忽略以 * 开头的注释行
    - 支持2列简单格式（跳过前2行数据）
    
    Returns:
        pd.DataFrame or None if parsing fails
    """
    buf = io.StringIO(text)
    lines = buf.getvalue().splitlines()

    # 过滤掉以 * 开头的行（注释）
    non_comment = [ln for ln in lines if not ln.lstrip().startswith("*")]

    # 从头找到首个"更像数据"的行：至少2个数值（支持简单格式）
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
        if nums >= 2:  # 至少2个数值
            start_idx = i
            break

    try:
        if start_idx is not None:
            # 检查是否为2列简单格式
            first_line = non_comment[start_idx].strip().split()
            if len(first_line) == 2:
                try:
                    float(first_line[0])
                    float(first_line[1])
                    # 是2列格式，跳过前2行数据
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
            # 回退方案：沿用原逻辑
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


def _assign_columns_by_type(df: pd.DataFrame, file_type_hint: str) -> Tuple[Optional[Tuple[pd.DataFrame, str]], Optional[str]]:
    """
    按给定类型重命名列并返回 (df, x_col)；若不匹配则返回 (None, 错误信息)
    
    Supported formats:
    - spec_pol: 6 columns (Wav, Int, Pol, Null1, Null2, sigma_int)
    - spec_i: 3 columns (Wav, Int, sigma_int)
    - spec_i_simple: 2 columns (Wav, Int)
    - lsd_pol: 7 columns (RV, Int, sigma_int, Pol, sigma_pol, Null1, sigma_null1)
    - lsd_i: 3 columns (RV, Int, sigma_int)
    - lsd_i_simple: 2 columns (RV, Int)
    """
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
            "RV", "Int", "sigma_int", "Pol", "sigma_pol", "Null1", "sigma_null1"
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


def _heuristic_guess(df: pd.DataFrame) -> Tuple[Optional[str], Optional[Tuple[pd.DataFrame, str]]]:
    """
    启发式自动判别数据类型：
    - 先用列数硬规则
    - 3 列时根据第一列数值范围判断是 Wav(nm) 还是 RV(km/s)
    - 2 列时判断简单格式
    """
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

    # 3 列：区分 spec_i vs lsd_i
    if ncol == 3:
        x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        if x.isna().all():
            return None, None

        xmin, xmax = x.min(), x.max()
        is_wav = (xmin >= 200) and (xmax <= 5000)
        is_rv = (xmin < 0) or (abs(xmin) <= 10000 and abs(xmax) <= 10000 and xmax < 200)

        if is_wav and not is_rv:
            return "spec_i", _assign_columns_by_type(df, "spec_i")[0]
        if is_rv and not is_wav:
            return "lsd_i", _assign_columns_by_type(df, "lsd_i")[0]

    # 2 列时判断简单格式
    if ncol == 2:
        x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        if x.isna().all():
            return None, None

        xmin, xmax = x.min(), x.max()
        is_wav = (xmin >= 200) and (xmax <= 5000)
        is_rv = (xmin < 0) or (abs(xmin) <= 10000 and abs(xmax) <= 10000 and xmax < 200)

        if is_wav and not is_rv:
            return "spec_i_simple", _assign_columns_by_type(df, "spec_i_simple")[0]
        if is_rv and not is_wav:
            return "lsd_i_simple", _assign_columns_by_type(df, "lsd_i_simple")[0]

    # 尝试所有可能的格式
    for ft in ["spec_i", "lsd_i", "spec_i_simple", "lsd_i_simple"]:
        res, err = _assign_columns_by_type(df, ft)
        if not err:
            return ft, res

    return None, None


def detect_and_assign_columns(df: pd.DataFrame, file_type_hint: str = "auto") -> Tuple[Optional[Tuple[pd.DataFrame, str, str]], Optional[str]]:
    """
    根据列数和用户选择的 file_type 提示，给 DataFrame 设定标准列名。
    - file_type_hint == "auto" 时自动探测
    - 其他值时严格按照指定类型验证
    
    Returns:
        ((df_named, x_col, resolved_type), err_msg)
    """
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
    """
    Container for a single observation profile (LSD or spectral).
    
    Attributes:
        wl: wavelength or velocity array (km/s for LSD, nm for spec)
        specI: Stokes I profile
        specIsig: sigma for Stokes I
        specV: Stokes V profile (optional)
        specVsig: sigma for Stokes V (optional)
        specQ: Stokes Q profile (optional)
        specU: Stokes U profile (optional)
        null: null profile (optional)
        profile_type: detected type (e.g., 'lsd_i', 'spec_pol')
    """
    def __init__(self, wl: np.ndarray, specI: np.ndarray, 
                 specIsig: Optional[np.ndarray] = None,
                 specV: Optional[np.ndarray] = None,
                 specVsig: Optional[np.ndarray] = None,
                 specQ: Optional[np.ndarray] = None,
                 specU: Optional[np.ndarray] = None,
                 null: Optional[np.ndarray] = None,
                 profile_type: str = "unknown"):
        self.wl = np.asarray(wl, dtype=float)
        self.specI = np.asarray(specI, dtype=float)
        self.specIsig = np.asarray(specIsig, dtype=float) if specIsig is not None else np.ones_like(specI) * 1e-4
        self.specV = np.asarray(specV, dtype=float) if specV is not None else np.zeros_like(specI)
        self.specVsig = np.asarray(specVsig, dtype=float) if specVsig is not None else np.ones_like(specI) * 1e-5
        self.specQ = np.asarray(specQ, dtype=float) if specQ is not None else np.zeros_like(specI)
        self.specU = np.asarray(specU, dtype=float) if specU is not None else np.zeros_like(specI)
        self.null = np.asarray(null, dtype=float) if null is not None else np.zeros_like(specI)
        self.profile_type = profile_type
    
    def scaleIsig(self, scale_factor: float):
        """Scale the Stokes I error bars by a factor (for chi^2 adjustment)."""
        self.specIsig *= scale_factor


def loadObsProfile(filename: str, file_type: str = "auto", 
                   vel_start: Optional[float] = None, 
                   vel_end: Optional[float] = None,
                   vel_shift: float = 0.0) -> Optional[ObservationProfile]:
    """
    Load a single observation profile from file.
    
    Args:
        filename: path to observation file
        file_type: format hint ('auto', 'lsd_i', 'lsd_pol', 'spec_i', etc.)
        vel_start: optional velocity range start (km/s, for filtering)
        vel_end: optional velocity range end (km/s, for filtering)
        vel_shift: radial velocity shift to apply (km/s, for barycentric correction)
    
    Returns:
        ObservationProfile or None if loading fails
    """
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
    
    # Extract data arrays
    wl = df[x_col].values
    specI = df["Int"].values if "Int" in df.columns else df.iloc[:, 1].values
    
    # Apply velocity shift if this is velocity-space data
    if resolved_type.startswith("lsd"):
        wl = wl + vel_shift
    
    # Apply velocity range filter
    if vel_start is not None and vel_end is not None:
        if resolved_type.startswith("lsd"):
            mask = (wl >= vel_start) & (wl <= vel_end)
            wl = wl[mask]
            specI = specI[mask]
            df = df[mask].copy()
    
    # Extract optional columns
    specIsig = df["sigma_int"].values if "sigma_int" in df.columns else None
    specV = df["Pol"].values if "Pol" in df.columns else (df["V"].values if "V" in df.columns else None)
    specVsig = df["sigma_pol"].values if "sigma_pol" in df.columns else None
    null = df["Null1"].values if "Null1" in df.columns else (df["Null"].values if "Null" in df.columns else None)
    
    return ObservationProfile(
        wl=wl,
        specI=specI,
        specIsig=specIsig,
        specV=specV,
        specVsig=specVsig,
        null=null,
        profile_type=resolved_type
    )


def obsProfSetInRange(fnames: List[str], vel_start: float, vel_end: float, 
                      vel_shifts: Optional[np.ndarray] = None,
                      file_type: str = "auto") -> List[ObservationProfile]:
    """
    Load a set of observation profiles from multiple files.
    
    Args:
        fnames: list of filenames
        vel_start: velocity range start (km/s)
        vel_end: velocity range end (km/s)
        vel_shifts: optional radial velocity shifts per file (km/s)
        file_type: format hint for all files
    
    Returns:
        List of ObservationProfile objects
    """
    if vel_shifts is None:
        vel_shifts = np.zeros(len(fnames))
    
    obsSet = []
    for i, fname in enumerate(fnames):
        obs = loadObsProfile(fname, file_type=file_type, 
                            vel_start=vel_start, vel_end=vel_end,
                            vel_shift=vel_shifts[i])
        if obs is not None:
            obsSet.append(obs)
        else:
            print(f"Warning: failed to load {fname}, skipping")
    
    return obsSet


def getObservedEW(obsSet: List[ObservationProfile], lineData, verbose: int = 1) -> float:
    """
    Calculate mean equivalent width from observation set.
    (Placeholder implementation; adapt to your line model.)
    
    Args:
        obsSet: list of ObservationProfile
        lineData: line model data (from lineprofile or linemodel_basic)
        verbose: verbosity level
    
    Returns:
        mean equivalent width
    """
    total_ew = 0.0
    count = 0
    
    for obs in obsSet:
        # Simple trapezoidal integration of (1 - I)
        if obs.profile_type.startswith("lsd"):
            # For LSD: integrate over velocity
            # Assuming wl is in km/s and specI is normalized
            ew = np.trapz(1.0 - obs.specI, obs.wl)
        else:
            # For spec: integrate over wavelength
            ew = np.trapz(1.0 - obs.specI, obs.wl)
        
        total_ew += ew
        count += 1
    
    mean_ew = total_ew / count if count > 0 else 0.0
    
    if verbose:
        print(f"Mean equivalent width: {mean_ew:.6f}")
    
    return mean_ew
