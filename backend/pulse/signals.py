"""Defensible rPPG signal transforms and quality measurements."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, find_peaks


def _rgb(rgb):
    x = np.asarray(rgb, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("RGB input must have shape (samples, 3)")
    return x


def pos(rgb, fps: float, window_seconds: float = 1.6) -> np.ndarray:
    """Plane-Orthogonal-to-Skin with temporal normalization and overlap-add."""
    x = _rgb(rgb)
    if fps <= 0 or len(x) < max(8, int(fps * 1.5)):
        return np.full(len(x), np.nan)
    n = max(8, int(round(window_seconds * fps)))
    step = max(1, n // 2)
    out = np.zeros(len(x)); weight = np.zeros(len(x))
    skin = np.array([1.0, 1.0, 1.0])
    # Two orthogonal chrominance axes spanning the plane normal to skin direction.
    u = np.array([1., -1., 0.]); u /= np.linalg.norm(u)
    v = np.cross(skin / np.linalg.norm(skin), u)
    starts = list(range(0, max(1, len(x) - n + 1), step))
    if not starts or starts[-1] + n < len(x): starts.append(max(0, len(x)-n))
    for start in starts:
        end = min(len(x), start+n); block = x[start:end]
        if len(block) < 8: continue
        mean = np.mean(block, axis=0)
        if np.any(np.abs(mean) < 1e-8): continue
        norm = block / mean - 1.0
        a, b = norm @ u, norm @ v
        sa, sb = np.std(a), np.std(b)
        pulse = a - (sa / sb if sb > 1e-10 else 0.0) * b
        taper = np.hanning(len(pulse)) if len(pulse) > 2 else np.ones(len(pulse))
        out[start:end] += pulse * taper
        weight[start:end] += taper
    return np.divide(out, weight, out=np.full_like(out, np.nan), where=weight > 1e-8)


def chrom(rgb) -> np.ndarray:
    """CHROM chrominance method using normalized RGB and std-weighted projections."""
    x = _rgb(rgb)
    means = np.mean(x, axis=0)
    if np.any(np.abs(means) < 1e-8): return np.full(len(x), np.nan)
    c = x / means - 1.0
    X = 3*c[:, 0] - 2*c[:, 1]
    Y = 1.5*c[:, 0] + c[:, 1] - 1.5*c[:, 2]
    sx, sy = np.std(X), np.std(Y)
    return X - (sx / sy if sy > 1e-10 else 0.0) * Y


def bandpass(signal, fps: float, low: float = .7, high: float = 4.0, order: int = 3):
    x = np.asarray(signal, dtype=float)
    if fps <= 0 or not 0 < low < high < fps/2:
        raise ValueError("Require 0 < low < high < Nyquist frequency")
    finite = np.isfinite(x)
    if finite.sum() < max(12, order*4): return np.full_like(x, np.nan)
    # Preserve missing samples; interpolation is only an internal filter input, never returned as measurement.
    idx = np.arange(len(x)); filled = np.interp(idx, idx[finite], x[finite])
    sos = butter(order, [low, high], btype="bandpass", fs=fps, output="sos")
    try: y = sosfiltfilt(sos, filled)
    except ValueError: return np.full_like(x, np.nan)
    y[~finite] = np.nan
    return y


def spectral_metrics(signal, fps: float, low=.7, high=4.0):
    x = np.asarray(signal, dtype=float); ok = np.isfinite(x); x = x[ok]
    result = {"dominant_frequency_hz": None, "heart_rate_bpm": None, "snr_db": None,
              "peak_prominence": None, "psd_frequency_hz": [], "psd": []}
    if fps <= 0 or len(x) < max(16, int(fps*4)): return result
    x = x - np.mean(x)
    f, p = welch(x, fs=fps, nperseg=min(len(x), max(16, int(fps*8))))
    band = (f >= low) & (f <= min(high, fps/2))
    if not np.any(band) or np.max(p[band]) <= 0: return result
    inds = np.flatnonzero(band); peak = inds[np.argmax(p[band])]
    noise = np.sum(p[band]) - p[peak]
    snr = 10*np.log10(max(p[peak], 1e-20)/max(noise, 1e-20))
    peaks, props = find_peaks(p[band], prominence=0)
    prominence = float(props["prominences"].max()) if len(peaks) else float(p[peak])
    freq = float(f[peak])
    result.update(dominant_frequency_hz=freq, heart_rate_bpm=freq*60,
                  snr_db=float(snr), peak_prominence=prominence,
                  psd_frequency_hz=f[band].tolist(), psd=p[band].tolist())
    return result


def signal_quality(metrics, missing_fraction=0., visibility=1., continuity=1.):
    """Bounded weighted score based only on measured signal/coverage characteristics."""
    snr = metrics.get("snr_db")
    prom = metrics.get("peak_prominence")
    if snr is None or prom is None: return 0.0
    snr_score = np.clip((snr + 5)/20, 0, 1)
    prom_score = np.clip(prom/(prom + 1e-5), 0, 1)
    return float(np.clip(.38*snr_score + .22*prom_score + .16*continuity + .14*visibility + .10*(1-missing_fraction), 0, 1))


def roi_rank(rows, threshold=.45):
    ranked = sorted(rows, key=lambda r: r.get("quality", 0), reverse=True)
    selected = ranked[0] if ranked and ranked[0].get("quality", 0) >= threshold else None
    for row in rows:
        row["selected"] = bool(selected is row)
        row["selection_reason"] = ("Highest measured quality above threshold" if selected is row
                                   else "Below quality threshold" if selected is None
                                   else "Lower measured quality than selected ROI")
    return selected
