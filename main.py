"""polymarket-faa-attribution-02: forecast Polymarket Yes prices minute by
minute, 4 hours ahead, as five quantiles.

Method (fitted on public Polymarket 1-minute closes; no data from the final
window is used anywhere):
  For each market, the forecast is built from the EMPIRICAL distribution of
  that market's own minute-to-minute price changes, measured on its last
  24 hours (1,440 minutes) of prices:
    q_tau(h) = x + 0.8 * (Q_tau(h) - Q_50(h))
  where x is the last price (the checkpoint close), h = 1..240 minutes after
  the checkpoint, and Q_tau(h) is the tau-quantile (type-7 interpolated) of
  the h-minute changes observed in the window, evaluated at grid horizons
  [1, 2, 4, 8, 15, 30, 60, 120, 240] and linearly interpolated in h.

Why this wins (measured, see README.md):
  - Prediction-market prices are jump-dominated and flat most of the time:
    ~50% of 4-hour moves are exactly 0, the rest are heavy-tailed and
    asymmetric (a candidate ruled out collapses; upside is bounded by
    1 - x). A Gaussian random walk (the house baseline, sigma
    0.0025/sqrt-min) is the wrong shape: far too wide in the interquartile
    range at every minute, too symmetric, and it grows as sqrt(h) while
    the real conditional spread saturates.
  - Empirical quantiles capture the true shape: a tight interquartile
    range around the last price plus fat, asymmetric tails.
  - The 0.8 scale corrects for overlap: h-minute changes measured on
    overlapping windows overstate the spread a little (the window also
    contains the market's own early, jumpier regime); 0.8 was the
    robust middle of the flat optimum (0.75-0.85) across practice,
    self-cut NVIDIA cases and the FAA markets' own history.
  - The median is the last price, flat. Every drift/momentum/mean-reversion
    adjustment tested made the score WORSE (the apparent 4h "reversion"
    correlation is a floor/ceiling artifact of bounded prices, not a
    tradeable signal). The 24-hour window adapts to the current regime:
    after a crash the curve is fit on post-crash moves, not on the crash.
  - Fallback: a market whose window is too short for an empirical curve
    (< 2 grid horizons, i.e. fewer than ~21 prices) gets the house
    baseline forecast (sigma 0.0025); a market with no prices gets flat
    0.5 quantiles (keeps the case valid).

Backtest (honest, pooled pinball loss in percentage points; % = vs the
house baseline, lower is better):
  - 6 practice cases (NVIDIA CEO, exact answers): 0.302 vs 0.364 (83.2%)
  - 6 mature self-cut NVIDIA cases: 0.677 vs 0.721 (93.9%)
  - 11 early self-cut NVIDIA cases (listing chaos, stress test): 2.865
    vs 3.027 (94.7%)
  - 4 FAA cases cut from the family's own history: 0.311 vs 0.410 (75.9%)
  The forecast beats the baseline in every practice case (6/6) and every
  FAA case (4/4), and beats a house-model mimic (sigma/drift from hourly
  closes, the same inputs the house AI gets) on every evaluation set.
  The prize bar is the better of the house baseline and the house model
  forecast on the final; both were beaten in backtest, but the final is a
  different, news-driven event and 12 cases are few: no stronger claim
  than "beats the bar in every backtest we could run".

Runs offline, stdlib only, deterministic: a pure function of the case's
prices plus the constants above. Reads <id>.in files from MAKEFUN_INPUT_DIR
(default /input), writes <id>.out to MAKEFUN_OUTPUT_DIR (default /output).
Runtime ~0.05 s per case (limit 8 s).
"""

import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# The quantile levels the contest scores, in the order each row lists them.
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at those levels (same as the house baseline).
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
# House baseline spread, price units per square-root minute (fallback only).
BASE_SIGMA = 0.0025
# Horizons (minutes) at which the empirical change distribution is measured.
GRID_H = [1, 2, 4, 8, 15, 30, 60, 120, 240]
# How much of the last window the curve is fit on (minutes).
WINDOW = 1440
# Global scale on the empirical spread around its median.
SCALE = 0.8


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def _quantiles(changes):
    """Type-7 interpolated quantiles of a sorted sample at QUANTILES."""
    s = sorted(changes)
    n = len(s)
    out = []
    for t in QUANTILES:
        pos = t * (n - 1)
        lo = int(pos)
        hi = min(n - 1, lo + 1)
        out.append(s[lo] + (pos - lo) * (s[hi] - s[lo]))
    return out


def _emp_curve(prices):
    """Empirical change quantiles at grid horizons from the last WINDOW prices.
    Returns {h: [q05, q25, q50, q75, q95]} of the h-minute change distribution.
    A horizon is used only when at least 20 overlapping changes exist."""
    p = prices[-WINDOW:]
    n = len(p)
    curve = {}
    for h in GRID_H:
        if n - h < 20:
            continue
        curve[h] = _quantiles([p[i + h] - p[i] for i in range(n - h)])
    return curve


def _interp_curve(curve, h):
    """Linear interpolation of the empirical quantiles at horizon h (minutes);
    clamped to the nearest measured horizon outside the grid."""
    hs = sorted(curve.keys())
    if h <= hs[0]:
        return curve[hs[0]]
    if h >= hs[-1]:
        return curve[hs[-1]]
    for a, b in zip(hs, hs[1:]):
        if a <= h <= b:
            w = (h - a) / (b - a)
            return [curve[a][j] + w * (curve[b][j] - curve[a][j]) for j in range(5)]
    return curve[hs[-1]]


def _baseline_rows(x, horizon):
    """House baseline: q = clamp(x + z * 0.0025 * sqrt(h), 0, 1)."""
    return [[price6(x + z * BASE_SIGMA * math.sqrt(h)) for z in Z]
            for h in range(1, horizon + 1)]


def forecast_market(prices, horizon):
    """240 x 5 quantile rows for one market, from its price window."""
    if not prices:
        return [[0.5, 0.5, 0.5, 0.5, 0.5] for _ in range(horizon)]
    x = prices[-1]
    curve = _emp_curve(prices)
    if len(curve) < 2:
        return _baseline_rows(x, horizon)
    rows = []
    for h in range(1, horizon + 1):
        q = _interp_curve(curve, h)
        row = [price6(x + SCALE * (q[j] - q[2])) for j in range(5)]
        for j in range(1, 5):  # enforce non-decreasing rows
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def _sanitize_rows(rows):
    """Final defensive pass over a market's rows: every value in [0, 1],
    every row non-decreasing. The forecast path already guarantees both
    (price6 clamps, forecast_market enforces monotonicity), so this is a
    no-op on well-formed input; it only rescues a case whose input prices
    contain a non-finite or out-of-range value, which would otherwise
    produce an invalid row and lose the whole case."""
    out = []
    for row in rows:
        r = [0.5 if not math.isfinite(v) else min(1.0, max(0.0, v)) for v in row]
        for j in range(1, 5):
            if r[j] < r[j - 1]:
                r[j] = r[j - 1]
        out.append(r)
    return out


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    inputs = sorted(p for p in in_dir.glob("*.in"))
    for path in inputs:
        case_id = path.name[: -len(".in")]
        with open(path, "r", encoding="utf-8") as f:
            case = json.load(f)
        horizon = int(case.get("horizon_minutes", 240))
        forecasts = []
        for m in case.get("markets", []):
            prices = [float(v) for v in m.get("prices", [])]
            forecasts.append({"id": m["id"],
                              "q": _sanitize_rows(forecast_market(prices, horizon))})
        with open(out_dir / f"{case_id}.out", "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}) + "\n")


if __name__ == "__main__":
    main()
