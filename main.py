"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Counts-growth model
-------------------
A trophy's earn rate is a ratio: earners / players tracked.  The two sides
of that ratio move for different reasons in week one -- a flood of new
players dilutes every rate at once, while existing players keep earning
and push individual counts up.  Fitting the ratio directly mixes those
forces together, so this program fits them apart, in count space:

* the tracked pool N(t) is extrapolated with a log-linear fit of
  ln N against x = ln(1 + hours since the first read);
* each trophy's earner count E_i(t) gets the same kind of fit,
  ln(E_i + 0.5) against x, with its slope shrunk toward the mean slope
  of its tier (platinum, gold, silver, bronze), because trophies of a
  tier progress at broadly similar speeds and a single noisy series
  should not be trusted on its own;
* the forecast rate is 100 * E_i(T) / N(T) at the target, blended in
  logit space 75/25 with simple persistence (the last observed rate),
  which anchors the extrapolation when the history is short or flat.

Uncertainty is expressed on the logit scale, where the contest scores:
each trophy's spread combines a floor with its own realised volatility
(the standard deviation of its logit moves per square-root day, shrunk
toward the median trophy), scaled by the square root of the horizon.
The five reported quantiles are the blended median shifted by the
standard normal quantiles for the 5/25/50/75/95 levels.

The program is fully deterministic: no randomness, no wall clock, fixed
iteration order everywhere.  Standard library only, no network.

Run locally with:
    MAKEFUN_INPUT_DIR=<dir with *.in> MAKEFUN_OUTPUT_DIR=<out dir> python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

EXPECTED_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_SCORES = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Model constants (see module docstring).
TIER_SHRINK = 0.4        # weight of the tier mean in a trophy's slope
PERSIST_WEIGHT = 0.25    # weight of the last rate in the logit blend
DEFAULT_VOL = 0.55       # volatility assumed when too few moves are seen
VOL_SHRINK = 0.5         # weight of the median vol in each trophy's vol
SPREAD_FLOOR_VAR = 0.12  # variance floor inside the spread square root
SPREAD_MIN = 0.35
SPREAD_MAX = 2.2


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    cleaned = text.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    moment = datetime.fromisoformat(cleaned)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def clamp(value, low, high):
    return low if value < low else high if value > high else value


def logit_of_percent(rate_pct):
    """Logit of a percentage rate, with the rate clamped off 0 and 100."""
    p = clamp(rate_pct, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def percent_from_logit(value):
    if value >= 0.0:
        return 100.0 / (1.0 + math.exp(-value))
    exp_v = math.exp(value)
    return 100.0 * exp_v / (1.0 + exp_v)


def round4(value):
    """Round to 4 decimal places, half up, deterministically."""
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    result = whole / 10000.0
    return int(result) if float(result).is_integer() else result


def ols(xs, ys):
    """Ordinary least squares y = a + b*x; returns (a, b).

    With fewer than two distinct x values there is no slope to speak of,
    so the fit degenerates to the mean of y with slope zero.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mean_y = sum(ys) / n
    if n < 2 or min(xs) == max(xs):
        return mean_y, 0.0
    mean_x = sum(xs) / n
    sxx = 0.0
    sxy = 0.0
    for x, y in zip(xs, ys):
        dx = x - mean_x
        sxx += dx * dx
        sxy += dx * (y - mean_y)
    if sxx <= 0.0:
        return mean_y, 0.0
    slope = sxy / sxx
    return mean_y - slope * mean_x, slope


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return None
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def read_stamp(entry):
    """The timestamp that positions a history read on the case timeline."""
    return parse_time(entry.get("slot") or entry["at"])


def scaled_logit_moves(series):
    """Logit moves between adjacent reads, each scaled by 1/sqrt(days).

    `series` is a list of (hours_since_first, rate) for one trophy, in
    time order.  A pair contributes only when the rate actually changed
    and time actually advanced: Exophase republishes an unchanged page
    for hours at a stretch, and those flat republishes say nothing about
    how jumpy the trophy is.  Scaling by the square root of the gap puts
    a 10-minute move and a 6-hour move on the same per-day footing.
    """
    moves = []
    for (h_prev, r_prev), (h_cur, r_cur) in zip(series, series[1:]):
        if r_cur == r_prev:
            continue
        gap_days = (h_cur - h_prev) / 24.0
        if gap_days <= 0.0:
            continue
        step = logit_of_percent(r_cur) - logit_of_percent(r_prev)
        moves.append(step / math.sqrt(gap_days))
    return moves


def trophy_volatility(series):
    """Standard deviation of a trophy's scaled logit moves.

    Fewer than two usable moves cannot pin a spread down, so the
    contest-wide default takes over for that trophy.
    """
    moves = scaled_logit_moves(series)
    if len(moves) < 2:
        return DEFAULT_VOL
    mean = sum(moves) / len(moves)
    var = sum((m - mean) ** 2 for m in moves) / len(moves)
    return math.sqrt(var)


def spread_for(vol, horizon_days):
    """Logit-scale half-width for one trophy at the given horizon.

    A variance floor keeps even a perfectly flat history honestly
    uncertain (the next Exophase refresh can still move every rate),
    and the clamp keeps one wild series from blowing the band up.
    """
    raw = math.sqrt(SPREAD_FLOOR_VAR + vol * vol * horizon_days)
    return clamp(raw, SPREAD_MIN, SPREAD_MAX)


def rate_from_log_counts(log_count, log_pool):
    """100 * exp(log_count - log_pool), clamped to the model's range.

    The log difference is clamped first so a steep fitted slope over a
    long horizon cannot overflow exp() before the rate clamp applies;
    the bounds are exactly the [0.02, 99.95] rate clamp in log space.
    """
    log_ratio = clamp(log_count - log_pool, math.log(0.0002), math.log(0.9995))
    return clamp(100.0 * math.exp(log_ratio), 0.02, 99.95)


def fallback_rate(tier, tier_fallback, global_fallback):
    """Median last rate for a trophy with no usable series of its own.

    Prefers its own tier's median, then the case-wide median, then a
    neutral 1%.  The checks are explicit None tests rather than truthy
    shortcuts: a tier whose median last rate is genuinely 0.0 must be
    honoured, not skipped over as if the tier had no data at all.
    """
    rate = tier_fallback.get(tier)
    if rate is not None:
        return rate
    if global_fallback is not None:
        return global_fallback
    return 1.0


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = float(case["horizon_minutes"]) / 1440.0

    # --- fallback medians per tier, from each trophy's last known rate ---
    last_rates = []
    for idx in range(len(trophies)):
        found = None
        for entry in reversed(history):
            rates = entry.get("rates") or []
            if idx < len(rates) and rates[idx] is not None:
                found = float(rates[idx])
                break
        last_rates.append(found)
    tier_last = {}
    for trophy, rate in zip(trophies, last_rates):
        if rate is not None:
            tier_last.setdefault(trophy["tier"], []).append(rate)
    tier_fallback = {tier: median(vals) for tier, vals in tier_last.items()}
    all_last = [r for r in last_rates if r is not None]
    global_fallback = median(all_last) if all_last else None

    if not history:
        rows = []
        for trophy in trophies:
            med_rate = fallback_rate(trophy["tier"], tier_fallback, global_fallback)
            centre = logit_of_percent(med_rate)
            rows.append({"id": trophy["id"],
                         "q": quantiles(centre, spread_for(DEFAULT_VOL, horizon_days))})
        return {"trophies": rows}

    first_moment = read_stamp(history[0])
    target_moment = parse_time(case["target"])
    hours = []
    for entry in history:
        delta = (read_stamp(entry) - first_moment).total_seconds() / 3600.0
        hours.append(delta)
    xs_all = [math.log1p(h) for h in hours]
    x_target = math.log1p((target_moment - first_moment).total_seconds() / 3600.0)

    # --- pool growth: one fit shared by every trophy in the case ---
    pool_y = [math.log(max(1.0, float(entry["players_tracked"]))) for entry in history]
    a_pool, b_pool = ols(xs_all, pool_y)
    log_pool_target = a_pool + b_pool * x_target

    # --- per-trophy count series, slopes and volatilities ---
    raw_slopes = []
    intercepts = []
    series_by_idx = []
    vols = []
    for idx in range(len(trophies)):
        series = []
        for pos, entry in enumerate(history):
            rates = entry.get("rates") or []
            if idx < len(rates) and rates[idx] is not None:
                series.append((hours[pos], float(rates[idx]),
                               float(entry["players_tracked"])))
        series_by_idx.append(series)
        if series:
            xs = [math.log1p(h) for h, _r, _n in series]
            ys = [math.log(_r / 100.0 * _n + 0.5) for _h, _r, _n in series]
            a_i, b_i = ols(xs, ys)
        else:
            a_i, b_i = 0.0, 0.0
        intercepts.append(a_i)
        raw_slopes.append(b_i)
        plain = [(h, r) for h, r, _n in series]
        vols.append(trophy_volatility(plain))

    # --- shrink slopes toward their tier mean (platinum: global mean) ---
    global_slope = sum(raw_slopes) / len(raw_slopes) if raw_slopes else 0.0
    tier_slopes = {}
    for trophy, slope in zip(trophies, raw_slopes):
        tier_slopes.setdefault(trophy["tier"], []).append(slope)
    tier_mean_slope = {tier: sum(v) / len(v) for tier, v in tier_slopes.items()}
    slopes = []
    for trophy, slope in zip(trophies, raw_slopes):
        if trophy["tier"] == "platinum" or len(tier_slopes[trophy["tier"]]) <= 1:
            centre = global_slope
        else:
            centre = tier_mean_slope[trophy["tier"]]
        slopes.append((1.0 - TIER_SHRINK) * slope + TIER_SHRINK * centre)

    # --- shrink volatilities toward the median trophy ---
    vol_centre = median(vols)
    if vol_centre is None:
        vol_centre = DEFAULT_VOL
    vols = [(1.0 - VOL_SHRINK) * v + VOL_SHRINK * vol_centre for v in vols]

    rows = []
    for idx, trophy in enumerate(trophies):
        spread = spread_for(vols[idx], horizon_days)
        if not series_by_idx[idx]:
            med_rate = fallback_rate(trophy["tier"], tier_fallback, global_fallback)
            centre = logit_of_percent(med_rate)
            rows.append({"id": trophy["id"], "q": quantiles(centre, spread)})
            continue
        log_count_target = intercepts[idx] + slopes[idx] * x_target
        rate_model = rate_from_log_counts(log_count_target, log_pool_target)
        centre = ((1.0 - PERSIST_WEIGHT) * logit_of_percent(rate_model)
                  + PERSIST_WEIGHT * logit_of_percent(last_rates[idx]))
        rows.append({"id": trophy["id"], "q": quantiles(centre, spread)})
    return {"trophies": rows}


def quantiles(centre_logit, spread):
    values = []
    previous = 0.0
    for z in Z_SCORES:
        q = clamp(percent_from_logit(centre_logit + z * spread), 0.0, 100.0)
        q = round4(q)
        if q < previous:
            q = previous
        values.append(q)
        previous = q
    return values


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("levels") not in (None, EXPECTED_LEVELS):
            raise ValueError(f"{path.name}: unexpected levels {case.get('levels')}")
        result = forecast_case(case)
        payload = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
