"""
Forecast credit-card usage from an Excel sheet, and report the seasonality in
the history.

    python credit_card_forecast.py --excel cards.xlsx
    python credit_card_forecast.py --excel cards.xlsx --target growth
    python credit_card_forecast.py --excel cards.xlsx --model gbm --rolling 6 --plot

All the data handling and both models live in cc_lib.py, which sweep.py also
uses, so the two scripts can never drift apart and report incomparable numbers.

See FORECASTING.md for the full guide.
"""

import argparse

import numpy as np
import pandas as pd

import cc_lib as L
import settings as S

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
UNIT = S.CURRENCY_UNIT


def money(value, width=16):
  """A currency figure with its unit attached, so no number is ambiguous."""
  return f"{value:>{width},.0f} {UNIT}" if UNIT else f"{value:>{width},.0f}"

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
src = ap.add_argument_group("input")
src.add_argument("--excel", required=True, help="path to the .xlsx file")
src.add_argument("--sheet", default=S.SHEET,
                 help="sheet name, or index counting from 0 (default: from settings.py)")
src.add_argument("--date-col", default=S.COLUMN_DATE, help="override the date column")
src.add_argument("--balance-col", default=S.COLUMN_BALANCE, help="override the balance column")
src.add_argument("--currency-col", default=S.COLUMN_CURRENCY, help="override the currency column")
src.add_argument("--currency", default=None, help="which currency to model")
src.add_argument("--growth-col", default=S.COLUMN_GROWTH, help="name of the growth column")
src.add_argument("--use-growth", action="store_true",
                 help="READ the daily change from the sheet's growth column "
                      "instead of recomputing it from the balance. This is NOT "
                      "how you ask for a growth forecast -- that is "
                      "--target growth. You rarely need this; the script says "
                      "so when it would change nothing")
src.add_argument("--grid", default=S.DEFAULT_GRID, choices=["auto", "business", "calendar"],
                 help="auto (default) keeps weekends only if they carry activity")

what = ap.add_argument_group("what to forecast")
what.add_argument("--target", default=S.DEFAULT_TARGET,
                  choices=["purchases", "net_change", "growth", "balance"])
what.add_argument("--horizon", type=int, default=S.DEFAULT_HORIZON, help="steps ahead")

mdl = ap.add_argument_group("models")
mdl.add_argument("--model", default=S.DEFAULT_MODEL,
                 choices=["timesfm", "gbm", "both"])
mdl.add_argument("--strategy", default=S.TIMESFM_STRATEGY,
                 choices=["auto", "multichannel", "signed-log", "raw"],
                 help="TimesFM only: how to handle a signed target")
mdl.add_argument("--no-covariates", action="store_true", help="TimesFM only")
mdl.add_argument("--znorm", action="store_true", help="TimesFM only")
mdl.add_argument("--context", type=int, default=S.TIMESFM_CONTEXT,
                 help="TimesFM only: cap the history fed to the model")
mdl.add_argument("--checkpoint", default=None,
                 help="folder holding the TimesFM weights, or a HuggingFace "
                      "repo id. Default comes from TIMESFM_WEIGHTS in settings.py")
mdl.add_argument("--device", default=S.TIMESFM_DEVICE, help="cuda / mps / cpu")
mdl.add_argument("--offline", action="store_true", default=S.TIMESFM_OFFLINE,
                 help="never contact HuggingFace. Only needed when the weights "
                      "come from the cache rather than a local folder")
mdl.add_argument("--deflator", default=None, metavar="FILE",
                 help="price index file (date + index columns) to divide out "
                      "inflation with. Overrides DEFLATOR_FILE in settings.py")
mdl.add_argument("--rescale", type=int, default=None, metavar="DAYS",
                 help="adjust for inflation: divide each day by how large a "
                      "typical day was at the time, measured over DAYS. Try 90. "
                      "0 forces it off. Default: RESCALE_WINDOW in settings.py")
mdl.add_argument("--num-leaves", type=int, default=S.GBM_NUM_LEAVES, help="GBM only")
mdl.add_argument("--learning-rate", type=float, default=S.GBM_LEARNING_RATE, help="GBM only")
mdl.add_argument("--rounds", type=int, default=S.GBM_ROUNDS, help="GBM only")

out = ap.add_argument_group("evaluation and output")
out.add_argument("--rolling", type=int, default=0, metavar="N",
                 help="score over N rolling origins instead of one holdout")
out.add_argument("--no-backtest", action="store_true")
out.add_argument("--plot", action="store_true", help="write forecast.png")
out.add_argument("--out", default="forecast.csv")
args = ap.parse_args()

# argparse hands back a string, and pandas treats a string as a sheet NAME.
# "--sheet 1" must mean the second sheet, not a sheet called "1".
if isinstance(args.sheet, str) and args.sheet.strip().lstrip("-").isdigit():
  args.sheet = int(args.sheet)

MODELS = ["timesfm", "gbm"] if args.model == "both" else [args.model]
tf_params = L.TimesFMParams(
    strategy=args.strategy, covariates=not args.no_covariates, znorm=args.znorm,
    context=args.context, device=args.device, offline=args.offline,
    rescale=args.rescale, deflator=args.deflator,
    **({"checkpoint": args.checkpoint} if args.checkpoint else {}))
gbm_params = L.GBMParams(num_leaves=args.num_leaves,
                         learning_rate=args.learning_rate, rounds=args.rounds,
                         rescale=args.rescale, deflator=args.deflator)
PARAMS = {"timesfm": tf_params, "gbm": gbm_params}


def rule(title):
  print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ------------------------------------------------------------------- loading --
rule("1. LOADING AND GRID")
data = L.load(args.excel, sheet=args.sheet, date_col=args.date_col,
              balance_col=args.balance_col, currency_col=args.currency_col,
              currency=args.currency, growth_col=args.growth_col,
              use_growth=args.use_growth, grid=args.grid)
for note in data.notes:
  print(f"  {note}")
print(f"\ngrid: {data.grid_mode} -> {len(data.index):,} steps "
      f"({data.index[0]:%Y-%m-%d} to {data.index[-1]:%Y-%m-%d})")
print(f"  mean change per step : {money(data.net_change.mean())}")
print(f"  mean daily purchases : {money(data.purchases.mean())}")
print(f"  steps with a payment : {(data.payments > 0).sum():>13,} steps "
      f"(largest {money(data.payments.max(), 0)})")

series = data.target(args.target)
print(f"\ntarget: {args.target}" + (f"   unit: {UNIT}" if UNIT else ""))


# -------------------------------------------------------------- seasonality --
rule("2. SEASONALITY IN THE HISTORY")
print("TimesFM only exploits seasonality, it never reports it. This section is")
print("ordinary statistics, so you can see the patterns the models are using.\n")

frame = pd.DataFrame({"value": series})
overall = frame["value"].mean()
print("By day of week (index = share of the average step):")
for d, grp in frame.groupby(frame.index.dayofweek)["value"]:
  idx = grp.mean() / overall if overall else float("nan")
  print(f"  {DAYS[d]}  {grp.mean():>12,.0f}  {idx:>5.2f}  {'#' * int(max(0, min(40, idx * 20)))}")

print("\nBy month:")
for m, grp in frame.groupby(frame.index.month)["value"]:
  print(f"  {m:>2}   {grp.mean():>12,.0f}  {grp.mean() / overall:>5.2f}")

print("\nStrongest days of the month:")
dom = (frame.groupby(frame.index.day)["value"].mean() / overall).sort_values(ascending=False)
for d, idx in dom.head(5).items():
  print(f"  day {d:>2}  {idx:>5.2f}")

try:
  from statsmodels.tsa.seasonal import STL
  from statsmodels.tsa.stattools import acf
  values = series.astype(float).to_numpy()
  lags = acf(values, nlags=min(40, len(values) // 3), fft=True)
  print("\nAutocorrelation peaks (lag -> correlation):")
  for p in sorted(range(1, len(lags)), key=lambda i: -abs(lags[i]))[:5]:
    print(f"  lag {p:>3}  {lags[p]:+.3f}")
  period = 5 if data.grid_mode == "business" else 7
  stl = STL(pd.Series(values, index=series.index), period=period, robust=True).fit()
  var = np.var(stl.resid + stl.seasonal)
  strength = max(0.0, 1 - np.var(stl.resid) / var) if var > 0 else 0.0
  print(f"\nWeekly seasonal strength: {strength:.2f} "
        f"({'strong' if strength > 0.6 else 'moderate' if strength > 0.3 else 'weak'})")
except Exception as exc:
  print(f"\n(ACF/STL unavailable: {type(exc).__name__}: {exc})")


# ---------------------------------------------------------------- backtests --
def show(df_scores, label):
  print(f"\n{label}")
  head = f" ({UNIT})" if UNIT else ""
  print(f"{'model':<16}{'MAE' + head:>18}{'pinball' + head:>18}{'80% cov':>9}"
        f"{'MAE spike' + head:>18}{'MAE other' + head:>18}")
  print("-" * 83)
  for name, row in df_scores.items():
    print(f"{name:<16}{row['mae']:>18,.0f}{row['pinball']:>18,.0f}"
          f"{row['coverage']:>8.0%}{row['mae_spike']:>18,.0f}"
          f"{row['mae_other']:>18,.0f}")


if not args.no_backtest and len(series) > args.horizon * 3:
  rule(f"3. BACKTEST — LAST {args.horizon} STEPS HELD OUT")
  if (series.iloc[-args.horizon:] < 0).any():
    print("sMAPE is meaningless on a signed series crossing zero; not reported.")
  upto = len(series) - args.horizon
  actual = series.to_numpy()[upto:].astype(float)
  scores = {}
  for name in MODELS:
    try:
      point, quant = L.forecast(name, data, args.target, upto, args.horizon, PARAMS[name])
      scores[name] = L.score(actual, point, quant)
    except Exception as exc:
      print(f"{name} failed: {type(exc).__name__}: {exc}")
  period = 5 if data.grid_mode == "business" else 7
  naive = np.resize(series.to_numpy()[:upto][-period:], args.horizon)
  scores["seasonal naive"] = L.score(actual, naive, np.repeat(naive[:, None], 3, axis=1))
  show(scores, "single holdout")
  print("\nOne holdout is one sample. Use --rolling 6 before trusting a ranking.")

if args.rolling:
  rule(f"4. ROLLING BACKTEST — {args.rolling} ORIGINS")
  print("Each origin holds out the next block and refits from scratch.\n")
  agg = {}
  for name in MODELS:
    try:
      r = L.rolling_eval(name, data, args.target, args.horizon, args.rolling, PARAMS[name])
      agg[name] = r
      print(f"{name}: " + "  ".join(f"{v:,.0f}" for v in r.mae))
    except Exception as exc:
      print(f"{name} failed: {type(exc).__name__}: {exc}")
  if agg:
    show({k: v.mean(numeric_only=True) for k, v in agg.items()},
         f"mean over {args.rolling} origins")
    if len(agg) > 1:
      wins = {k: 0 for k in agg}
      n = min(len(v) for v in agg.values())
      for i in range(n):
        wins[min(agg, key=lambda k: agg[k].mae.iloc[i])] += 1
      print("\nwins: " + ", ".join(f"{k} {v}" for k, v in wins.items()))
    print("\nCoverage should sit near 80%. Well below means the interval is too")
    print("narrow and will under-warn you in a bad month.")


# ----------------------------------------------------------------- forecast --
rule("5. FORECAST")
future = L.future_index(data.index[-1], args.horizon, data.grid_mode)
result = pd.DataFrame({"date": future, "day": [DAYS[d] for d in future.dayofweek]})
for name in MODELS:
  try:
    point, quant = L.forecast(name, data, args.target, len(series), args.horizon, PARAMS[name])
  except Exception as exc:
    print(f"{name} failed: {type(exc).__name__}: {exc}")
    continue
  sfx = "" if len(MODELS) == 1 else f"_{name}"
  result[f"forecast{sfx}"] = point
  result[f"p10{sfx}"] = quant[:, 0]
  result[f"p90{sfx}"] = quant[:, 2]

fcols = [c for c in result.columns if c.startswith("forecast")]
fmt = {c: "{:,.0f}".format for c in result.columns if c not in ("date", "day")}
print(result.head(14).to_string(index=False, formatters=fmt))
if len(result) > 14:
  print(f"... {len(result) - 14} more rows")

unit = "business days" if data.grid_mode == "business" else "days"
print(f"\nTotal over {args.horizon} {unit} "
      f"({future[0]:%Y-%m-%d} to {future[-1]:%Y-%m-%d})"
      + (f", in {UNIT}:" if UNIT else ":"))
for c in fcols:
  print(f"  {c:<22}{money(result[c].sum())}")
print("\nBy week:")
print(result.set_index("date")[fcols].resample("W").sum().to_string(
    formatters={c: "{:,.0f}".format for c in fcols}))
print("\nBy month:")
print(result.set_index("date")[fcols].resample("MS").sum().to_string(
    formatters={c: "{:,.0f}".format for c in fcols}))

result.to_csv(args.out, index=False)
print(f"\nwritten: {args.out}")

if args.plot:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  colors = {"forecast": "#c0392b", "forecast_timesfm": "#c0392b", "forecast_gbm": "#2d8659"}
  fig, ax = plt.subplots(figsize=(13, 5))
  recent = series.iloc[-min(len(series), args.horizon * 4):]
  ax.plot(recent.index, recent.to_numpy(), label="history", color="#3b6ea5", lw=1.1)
  for c in fcols:
    sfx = c.replace("forecast", "")
    ax.plot(future, result[c], label=c, color=colors.get(c, "#7f8c8d"), lw=1.6)
    ax.fill_between(future, result[f"p10{sfx}"], result[f"p90{sfx}"],
                    color=colors.get(c, "#7f8c8d"), alpha=0.15)
  ax.set_title(f"{args.target} — {args.horizon}-step forecast")
  ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
  fig.savefig("forecast.png", dpi=140)
  print("written: forecast.png")
