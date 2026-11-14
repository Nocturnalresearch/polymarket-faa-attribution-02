# Polymarket FAA attribution forecast — make.fun `polymarket-faa-attribution-02`

Forecasts the Yes price of each of Polymarket's four FAA-outage attribution
markets (Iran / foreign actor / hack / AI sandbox escape), minute by minute
for 4 hours ahead of a checkpoint, as five quantiles (5%, 25%, 50%, 75%,
95%) per minute — 4,800 numbers per case. Entry for the make.fun bounty
of the same name ($250, no entry fee). This is the same task as
`polymarket-faa-attribution-01`, moved to earlier dates by the owner while
the news is live; the model is the same one, re-validated on fresh data.

## Method

Per market, the forecast is built from the **empirical distribution of that
market's own minute-to-minute price changes** over its last 24 hours
(1,440 minutes):

    q_tau(h) = x + 0.8 * (Q_tau(h) - Q_50(h))

where `x` is the last price at the checkpoint, `h = 1..240` minutes after
it, and `Q_tau(h)` is the tau-quantile (type-7 interpolated) of the
h-minute changes, measured at grid horizons `[1, 2, 4, 8, 15, 30, 60, 120,
240]` and linearly interpolated in `h`. The median is the last price, flat
— no drift, no momentum, no mean reversion (every such adjustment tested
made the score worse). A market with too short a window for an empirical
curve falls back to the house baseline (`q = clamp(x + z * 0.0025 *
sqrt(h), 0, 1)`); a market with no prices gets flat 0.5 quantiles. A
final sanitize pass over every row clamps values to [0, 1], replaces
non-finite values with 0.5 and enforces non-decreasing quantiles, so a
bad input price cannot invalidate a whole case.

Why it works: prediction-market prices are jump-dominated and flat most
of the time (~50% of 4-hour moves are exactly 0, the rest heavy-tailed and
asymmetric). The house baseline's Gaussian sqrt(h) shape is wrong at every
horizon — too wide in the interquartile range, too symmetric. Empirical
quantiles capture the true shape: a tight IQR around the last price plus
fat, asymmetric tails. Scale 0.8 is the robust middle of the flat optimum
(0.75–0.85), not the practice optimum (0.65), because one jumpy final
case can erase the gains of several calm ones.

## Backtest (pooled pinball loss, percentage points; % = vs house baseline)

| set | this entry | house baseline | % |
|---|---|---|---|
| practice, 6 cases (NVIDIA CEO) | 0.302 | 0.364 | 83.2% |
| self-cut NVIDIA, mature, 6 cases | 0.677 | 0.721 | 93.9% |
| self-cut NVIDIA, early, 11 cases | 2.865 | 3.027 | 94.7% |
| FAA family history, 4 cases | 0.311 | 0.410 | 75.9% |

The entry beats the baseline in every practice case (6/6) and every FAA
case (4/4), and beats a house-model mimic (sigma/drift from hourly closes)
on every evaluation set. The prize bar is the better of the house baseline
and the real house model forecast on the final; both were beaten in
backtest, but the final is a different, news-driven event and 12 cases are
few — no stronger claim than "beats the bar in every backtest we could
run".

## Run

    python3 main.py

Reads `<id>.in` case files from `MAKEFUN_INPUT_DIR` (default `/input`),
writes `<id>.out` to `MAKEFUN_OUTPUT_DIR` (default `/output`). Python 3
stdlib only, no network, deterministic (identical bytes across runs),
~0.05 s per case (limit 8 s, 1 CPU, 1024 MB).
