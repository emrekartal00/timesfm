"""
Run many TimesFM and LightGBM configurations over the same rolling backtest and
rank them, so the choice between them is measured rather than argued.

    python sweep.py --excel cards.xlsx --target net_change --use-growth
    python sweep.py --excel cards.xlsx --preset full --origins 6
    python sweep.py --excel cards.xlsx --only gbm --horizon 14

Every configuration is scored on identical origins with identical data prep, so
the numbers are comparable. Results go to sweep_results.csv.

Reading the output:
  MAE       average error of the point forecast. Lower is better.
  pinball   scores the whole p10/p50/p90 interval, not just the point. Lower
            is better, and it is the metric to trust for a spiky series.
  coverage  share of actual values that fell inside p10..p90. Should be ~80%.
            Far below means overconfident: the interval is too narrow and will
            under-warn you in a bad month. Far above means it is too vague.

A configuration that wins on MAE but covers 50% is not the better model; it is
a sharper guess with worse honesty about its own uncertainty.

See FORECASTING.md for the full guide.
"""

import argparse
import itertools
import time

import numpy as np
import pandas as pd

import cc_lib as L

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--excel", required=True)
ap.add_argument("--sheet", default=0)
ap.add_argument("--date-col", default=None)
ap.add_argument("--balance-col", default=None)
ap.add_argument("--currency", default=None)
ap.add_argument("--growth-col", default=None)
ap.add_argument("--use-growth", action="store_true")
ap.add_argument("--grid", default="auto", choices=["auto", "business", "calendar"])
ap.add_argument("--target", default="net_change",
                choices=["purchases", "net_change", "balance"])
ap.add_argument("--horizon", type=int, default=30)
ap.add_argument("--origins", type=int, default=5,
                help="rolling origins per configuration (more = slower, surer)")
ap.add_argument("--preset", default="quick", choices=["quick", "full"],
                help="quick = a handful of configs; full = the whole grid")
ap.add_argument("--only", default="both", choices=["timesfm", "gbm", "both"])
ap.add_argument("--sort", default="pinball", choices=["mae", "pinball", "coverage"])
ap.add_argument("--out", default="sweep_results.csv")
ap.add_argument("--checkpoint", default=None)
ap.add_argument("--device", default=None)
args = ap.parse_args()


# ------------------------------------------------------------------- grids ---
# Each entry becomes one configuration. Keep the quick grids small: every
# configuration is refit at every origin, so cost is configs x origins.
TIMESFM_GRID = {
    "quick": dict(strategy=["auto", "raw"], covariates=[True, False],
                  znorm=[False], context=[None]),
    "full": dict(strategy=["auto", "multichannel", "raw", "signed-log"],
                 covariates=[True, False], znorm=[False, True],
                 context=[None, 365, 180]),
}
GBM_GRID = {
    "quick": dict(num_leaves=[15, 31], learning_rate=[0.05], rounds=[300],
                  min_data_in_leaf=[20]),
    "full": dict(num_leaves=[7, 15, 31, 63], learning_rate=[0.03, 0.05, 0.1],
                 rounds=[200, 300, 600], min_data_in_leaf=[10, 20, 40]),
}


def expand(grid):
  keys = list(grid)
  for combo in itertools.product(*(grid[k] for k in keys)):
    yield dict(zip(keys, combo))


def label(model, cfg):
  short = {"strategy": "strat", "covariates": "cov", "znorm": "znorm",
           "context": "ctx", "num_leaves": "leaves", "learning_rate": "lr",
           "rounds": "rounds", "min_data_in_leaf": "minleaf"}
  bits = [f"{short.get(k, k)}={v}" for k, v in cfg.items()]
  return f"{model}[{','.join(bits)}]"


# -------------------------------------------------------------------- run ---
print(f"loading {args.excel} ...")
data = L.load(args.excel, sheet=args.sheet, date_col=args.date_col,
              balance_col=args.balance_col, currency=args.currency,
              growth_col=args.growth_col, use_growth=args.use_growth, grid=args.grid)
for note in data.notes:
  print(f"  {note}")
print(f"  grid={data.grid_mode}  steps={len(data.index):,}  target={args.target}")

jobs = []
if args.only in ("timesfm", "both"):
  extra = {}
  if args.checkpoint:
    extra["checkpoint"] = args.checkpoint
  if args.device:
    extra["device"] = args.device
  for cfg in expand(TIMESFM_GRID[args.preset]):
    jobs.append(("timesfm", cfg, L.TimesFMParams(**cfg, **extra)))
if args.only in ("gbm", "both"):
  for cfg in expand(GBM_GRID[args.preset]):
    jobs.append(("gbm", cfg, L.GBMParams(**cfg)))

print(f"\n{len(jobs)} configurations x {args.origins} origins "
      f"x horizon {args.horizon}\n")

rows = []
for i, (model, cfg, params) in enumerate(jobs, 1):
  name = label(model, cfg)
  print(f"[{i:>3}/{len(jobs)}] {name:<58}", end="", flush=True)
  t0 = time.time()
  try:
    r = L.rolling_eval(model, data, args.target, args.horizon, args.origins, params)
    if r.empty:
      print("no origins fit in the history")
      continue
    m = r.mean(numeric_only=True)
    rows.append({"model": model, "config": name, "mae": m.mae,
                 "pinball": m.pinball, "coverage": m.coverage,
                 "mae_spike": m.mae_spike, "mae_other": m.mae_other,
                 "seconds": time.time() - t0, **cfg})
    print(f"MAE {m.mae:>9,.0f}  pin {m.pinball:>7,.0f}  "
          f"cov {m.coverage:>4.0%}  {time.time() - t0:>5.1f}s")
  except Exception as exc:
    print(f"failed: {type(exc).__name__}: {exc}")

if not rows:
  raise SystemExit("nothing ran successfully")

df = pd.DataFrame(rows).sort_values(args.sort,
                                    ascending=(args.sort != "coverage"))
if args.sort == "coverage":     # closest to 0.80 wins, not the largest
  df = df.reindex((df.coverage - 0.80).abs().sort_values().index)

print(f"\n{'=' * 78}\nRANKED BY {args.sort.upper()}\n{'=' * 78}")
print(f"{'config':<58}{'MAE':>10}{'pinball':>9}{'cov':>6}")
print("-" * 83)
for _, r in df.head(20).iterrows():
  print(f"{r.config:<58}{r.mae:>10,.0f}{r.pinball:>9,.0f}{r.coverage:>6.0%}")

df.to_csv(args.out, index=False)
print(f"\nwritten: {args.out}  ({len(df)} configurations)")

print(f"\n{'=' * 78}\nBEST OF EACH MODEL\n{'=' * 78}")
for model in df.model.unique():
  sub = df[df.model == model]
  for metric in ("mae", "pinball"):
    best = sub.loc[sub[metric].idxmin()]
    print(f"{model:<9} best {metric:<8} {best.config}")
    print(f"{'':>9}      MAE {best.mae:,.0f}  pinball {best.pinball:,.0f}  "
          f"coverage {best.coverage:.0%}")

if len(df.model.unique()) > 1:
  print("\nA fair comparison uses each model's own best configuration, not its")
  print("default. Take the winner above, then rerun credit_card_forecast.py")
  print("with those arguments to produce the actual forecast.")
