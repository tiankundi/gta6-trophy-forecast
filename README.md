# GTA VI Trophy Rarity Forecast — counts-growth model of week-one earn rates

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tiankundi/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/tiankundi/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, the earn rate Exophase will print at the target — the percent of tracked players who have earned that trophy — as five quantiles (5/25/50/75/95) scored by pinball loss on the logit scale.

## The model

An earn rate is a ratio, earners ÷ players tracked, and in week one its two sides move for different reasons: new players flooding in dilute every rate at once, while existing players keep progressing and push individual earner counts up. Fitting the ratio directly mixes those forces, so `main.py` fits them apart, in count space:

- **Pool growth.** `ln N` (players tracked) is fitted by ordinary least squares against `x = ln(1 + hours since the first read)` over the whole history, and read off at the target.
- **Earner growth.** Each trophy's count `E_i = rate_i / 100 × N` gets the same fit on `ln(E_i + 0.5)`; its slope is shrunk 60/40 toward the mean slope of its tier (platinum shrinks toward the global mean), because same-tier trophies progress at similar speeds and one noisy series should not be trusted alone.
- **Recombine and anchor.** The model rate is `100 × E_i(T) / N(T)`, clamped to [0.02, 99.95], then blended in logit space 75/25 with persistence (the last observed rate), which anchors the extrapolation when the history is short or flat.
- **Honest spread.** Each trophy's volatility is the standard deviation of its logit moves per square-root day over adjacent reads where the rate changed (default 0.55 when fewer than two moves exist), shrunk 50% toward the median trophy. The spread is `clamp(sqrt(0.12 + vol² × horizon_days), 0.35, 2.2)`, and the five quantiles are the blended median shifted by the normal quantiles ±1.645 / ±0.674 / 0 on the logit scale — the scale the pinball loss is scored on, so rare trophies count as much as common ones.
- **Fallbacks.** A trophy with no usable reads takes its tier's median last rate (then the case-wide median, then 1.0) as its median with the default spread; an empty history applies that to every trophy.

The program is fully deterministic (no randomness, no wall clock, fixed iteration order), uses only the Python 3.13 standard library, and needs no network.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` case file in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one `<id>.out` forecast per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`).

---

<sub>Entered on [make.fun](https://make.fun/?ref=dj5sp1xh&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
