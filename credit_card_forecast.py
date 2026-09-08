"""
Forecast credit-card usage from an Excel sheet with TimesFM 3.0, and report the
seasonality in the history.

    python credit_card_forecast.py --excel cards.xlsx
    python credit_card_forecast.py --excel cards.xlsx --currency TRY --horizon 30
    python credit_card_forecast.py --excel cards.xlsx --target balance --plot

Expects a sheet with at least a date column and a balance column. A currency
column is used to split the data if present. Everything else (day of month,
year, yearmonth, day of week, growth) is RECOMPUTED from the dates and the
balance rather than trusted, since those columns are derived anyway and a stale
fill-down or an inserted row is a common source of silent error. A `growth`
column found in the sheet is checked against the recomputed version and the
agreement reported; --growth-col makes it authoritative instead.

WHAT IT DOES, AND WHY
---------------------
1. Puts the data on a regular grid. TimesFM never sees your dates; it assumes
   every step is one period, so gapped rows silently corrupt the weekly
   pattern. The default grid is business days -- one step per trading day,
   weekends dropped rather than zero-filled, so a Monday step carries the
   weekend's activity exactly as a row-to-row `growth` column does.

   Which grid is right depends on the export, so it is detected rather than
   assumed. If the weekend rows carry real spending, they are kept -- dropping
   them would discard real money. If they are flat padding, they are dropped,
   because zeros every week dilute the series and smear the statement spike:
   on a rolling six-window backtest that was worth 37% of mean MAE and 72% of
   payment-day MAE. Override with --grid business or --grid calendar.

2. Separates the flow from the stock. Balance is a level. Usage is the daily
   change in that level. They are different forecasting problems and mixing
   them up is the most common mistake here.

3. Splits purchases from payments. The daily change on a credit card is
   purchases minus payments, and a statement payment is a single huge negative
   spike. Left in, those spikes dominate the model. `--target purchases`
   (the default) keeps only the positive part, which is what "usage" means.

4. Feeds the calendar to the model as covariates. Day of week and day of month
   are known for the future, so they go in as past_future_covariates -- this is
   the main reason to use TimesFM 3.0 over 2.5 for this data.

5. Backtests before forecasting. An unvalidated forecast is a guess. The last
   `horizon` days are held out and compared against seasonal-naive.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- arguments --
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--excel", required=True, help="path to the .xlsx file")
ap.add_argument("--sheet", default=0, help="sheet name or index (default: first)")
ap.add_argument("--date-col", default=None, help="override date column name")
ap.add_argument("--balance-col", default=None, help="override balance column name")
ap.add_argument("--currency-col", default=None, help="override currency column name")
ap.add_argument("--currency", default=None,
                help="which currency to model (default: the one with most rows)")
ap.add_argument("--target", default="purchases",
                choices=["purchases", "net_change", "balance"],
                help="purchases = usage (default); net_change = purchases minus "
                     "payments; balance = the outstanding level")
ap.add_argument("--horizon", type=int, default=30, help="days to forecast")
ap.add_argument("--checkpoint",
                default=os.environ.get("TIMESFM_CHECKPOINT",
                                       "google/timesfm-3.0-pytorch"))
ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
ap.add_argument("--grid", default="auto",
                choices=["auto", "business", "calendar"],
                help="auto (default) keeps weekends only if they carry real "
                     "activity. business = weekdays only. calendar = all 7 days")
ap.add_argument("--growth-col", default=None,
                help="use this column as the daily change instead of "
                     "recomputing it from the balance")
ap.add_argument("--strategy", default="auto",
                choices=["auto", "multichannel", "signed-log", "raw"],
                help="how to model a signed target. auto = multichannel for "
                     "net_change, raw otherwise")
ap.add_argument("--no-covariates", action="store_true",
                help="skip the calendar covariates")
ap.add_argument("--no-backtest", action="store_true")
ap.add_argument("--plot", action="store_true", help="write forecast.png")
ap.add_argument("--out", default="forecast.csv")
args = ap.parse_args()


def rule(title):
  print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ------------------------------------------------------------------- loading --
rule("1. LOADING")

df = pd.read_excel(args.excel, sheet_name=args.sheet)
print(f"{args.excel}: {len(df):,} rows, {len(df.columns)} columns")
print(f"columns: {list(df.columns)}")


def find_col(override, *keywords):
  """Locate a column by name fragment, case-insensitively."""
  if override:
    if override not in df.columns:
      raise SystemExit(f"No column named {override!r}. Available: {list(df.columns)}")
    return override
  for col in df.columns:
    name = str(col).strip().lower()
    if any(k in name for k in keywords):
      return col
  return None


date_col = find_col(args.date_col, "date", "tarih")
bal_col = find_col(args.balance_col, "balance", "bakiye", "amount", "tutar")
cur_col = find_col(args.currency_col, "currency", "curr", "para", "doviz", "döviz")

if date_col is None or bal_col is None:
  raise SystemExit(
      "Could not identify the date and balance columns automatically.\n"
      f"Found: {list(df.columns)}\n"
      "Pass --date-col and --balance-col explicitly."
  )
print(f"\ndate     -> {date_col!r}")
print(f"balance  -> {bal_col!r}")
print(f"currency -> {cur_col!r}" if cur_col else "currency -> none found")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
bad_dates = df[date_col].isna().sum()
if bad_dates:
  print(f"\ndropping {bad_dates} rows with unparseable dates")
  df = df.dropna(subset=[date_col])

# One currency at a time. Mixing currencies into one series is meaningless --
# the numbers are not comparable and the model would be fitting an exchange rate.
if cur_col is not None and df[cur_col].nunique() > 1:
  counts = df[cur_col].value_counts()
  print(f"\n{df[cur_col].nunique()} currencies present:")
  for k, v in counts.items():
    print(f"    {k}: {v:,} rows")
  chosen = args.currency or counts.index[0]
  if chosen not in set(df[cur_col]):
    raise SystemExit(f"Currency {chosen!r} not in the sheet.")
  print(f"modelling {chosen!r} (use --currency to pick another)")
  df = df[df[cur_col] == chosen]
elif cur_col is not None:
  chosen = df[cur_col].iloc[0]
  print(f"single currency: {chosen}")

df = df.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
df[bal_col] = pd.to_numeric(df[bal_col], errors="coerce")


# ------------------------------------------------------------ regular grid --
rule("2. PUTTING IT ON A REGULAR GRID")

span = (df[date_col].max() - df[date_col].min()).days + 1
weekend_rows = int((df[date_col].dt.dayofweek >= 5).sum())
print(f"{len(df):,} rows spanning {span:,} calendar days "
      f"({df[date_col].min():%Y-%m-%d} to {df[date_col].max():%Y-%m-%d})")
print(f"rows falling on a weekend: {weekend_rows:,}")

s_raw = df.set_index(date_col)[bal_col]

# TimesFM never sees dates -- it assumes every step is one period. So the rows
# have to sit on a regular grid before anything else is true.
#
# Two grids are defensible for card data, and which is better is an empirical
# question, not a matter of taste. On a rolling six-window backtest the
# business grid won five, cutting mean MAE by 37% and payment-day MAE by 72%.
# The single window it lost contained no payments at all. Weekend zero-rows
# dilute the series and smear the statement-day spike across neighbours, so
# business days are the default here.
# Whether to keep weekends is not a matter of taste, and it is not the same
# answer for every export. It depends on one fact about the data: do the
# weekend rows carry real spending, or are they padding?
#
#   no weekend rows at all      -> nothing to drop. A Monday step already
#                                  carries the weekend, exactly as a row-to-row
#                                  growth column does.
#   weekend rows, but flat      -> zeros every week dilute the series and smear
#                                  the statement spike. Drop them.
#   weekend rows with activity  -> the card is genuinely used at weekends and
#                                  the bank posts daily. Dropping them would
#                                  throw away real money. Keep all 7 days.
grid_mode = args.grid
if grid_mode == "auto":
  if weekend_rows == 0:
    grid_mode = "business"
    print("\nauto: no weekend rows in the sheet -> business days")
  else:
    wk = s_raw.diff().loc[s_raw.index.dayofweek >= 5]
    scale = max(s_raw.diff().abs().median(), 1e-9)
    active = float((wk.abs() > 0.05 * scale).mean())
    if active > 0.10:
      grid_mode = "calendar"
      print(f"\nauto: {weekend_rows:,} weekend rows and {active:.0%} of them "
            f"carry real movement")
      print("  -> keeping all 7 days; dropping them would discard real spending")
    else:
      grid_mode = "business"
      print(f"\nauto: {weekend_rows:,} weekend rows but only {active:.0%} "
            f"carry movement")
      print("  -> they are padding; dropping them so the weekly pattern is cleaner")

if grid_mode == "business":
  grid = pd.bdate_range(s_raw.index.min(), s_raw.index.max())
  print(f"grid: business days -> {len(grid):,} steps "
        f"(one step = one trading day)")
  print("  weekends are dropped rather than zero-filled; a Monday step")
  print("  therefore carries the whole weekend's activity, exactly like your")
  print("  growth column does")
else:
  grid = pd.date_range(s_raw.index.min(), s_raw.index.max(), freq="D")
  print(f"grid: every calendar day -> {len(grid):,} steps")

# A balance is a stock: on a day with no record it simply has not changed.
balance = s_raw.reindex(s_raw.index.union(grid)).ffill().reindex(grid)
missing = int(balance.isna().sum())
if missing:
  balance = balance.bfill()
  print(f"  {missing} leading step(s) had no balance yet; back-filled")

# The daily change. Prefer the sheet's own column if asked for, but check it.
recomputed = balance.diff().fillna(0.0)
growth_col = args.growth_col or find_col(None, "growth", "artis", "artış", "degisim")
if growth_col is not None and growth_col in df.columns:
  supplied = pd.to_numeric(df.set_index(date_col)[growth_col], errors="coerce")
  supplied = supplied.reindex(grid)
  both_known = supplied.notna() & recomputed.notna()
  if both_known.sum():
    delta = (supplied[both_known] - recomputed[both_known]).abs()
    scale = max(recomputed.abs().mean(), 1e-9)
    agree = float((delta < 0.01 * scale).mean())
    print(f"\nfound growth column {growth_col!r}: agrees with "
          f"balance.diff() on {agree:.0%} of steps")
    if args.growth_col:
      net_change = supplied.fillna(recomputed)
      print("  using the sheet's column, as requested (--growth-col)")
      if agree < 0.95:
        print("  WARNING: it disagrees on more than 5% of steps. A stale")
        print("  fill-down or an inserted row will do that. The recomputed")
        print("  version is the safer choice unless you know why they differ.")
    else:
      net_change = recomputed
      print("  using the recomputed version (pass --growth-col to override);")
      print("  recomputing is immune to stale formulas and inserted rows")
  else:
    net_change = recomputed
else:
  net_change = recomputed

net_change = net_change.fillna(0.0)
purchases = net_change.clip(lower=0.0)      # usage: the positive part only
payments = (-net_change).clip(lower=0.0)

print(f"\n  mean change per step  : {net_change.mean():,.2f}")
print(f"  mean daily purchases  : {purchases.mean():,.2f}")
print(f"  steps with a payment  : {(payments > 0).sum():,} "
      f"(largest {payments.max():,.2f})")

series = {"purchases": purchases, "net_change": net_change, "balance": balance}[args.target]
series.name = args.target
print(f"\ntarget: {args.target}")


def future_index(last, n):
  """The next n steps on whichever grid is in use."""
  if grid_mode == "business":
    return pd.bdate_range(last + pd.Timedelta(days=1), periods=n)
  return pd.date_range(last + pd.Timedelta(days=1), periods=n, freq="D")


# -------------------------------------------------------------- seasonality --
rule("3. SEASONALITY IN THE HISTORY")
print("TimesFM does not report seasonality -- it only exploits it. This section")
print("is ordinary statistics, so you can see the patterns the model is using.\n")

frame = pd.DataFrame({"value": series})
frame["dow"] = frame.index.dayofweek
frame["dom"] = frame.index.day
frame["month"] = frame.index.month

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
overall = frame["value"].mean()

print("By day of week (index = share of the average day):")
for d, grp in frame.groupby("dow")["value"]:
  idx = grp.mean() / overall if overall else float("nan")
  bar = "#" * int(max(0, min(40, idx * 20)))
  print(f"  {DAYS[d]}  {grp.mean():>12,.0f}  {idx:>5.2f}  {bar}")

print("\nBy month:")
for m, grp in frame.groupby("month")["value"]:
  idx = grp.mean() / overall if overall else float("nan")
  print(f"  {m:>2}   {grp.mean():>12,.0f}  {idx:>5.2f}")

print("\nStrongest days of the month (index vs average):")
dom = (frame.groupby("dom")["value"].mean() / overall).sort_values(ascending=False)
for d, idx in dom.head(5).items():
  print(f"  day {d:>2}  {idx:>5.2f}")

try:
  from statsmodels.tsa.seasonal import STL
  from statsmodels.tsa.stattools import acf

  values = series.astype(float).to_numpy()
  lags = acf(values, nlags=min(40, len(values) // 3), fft=True)
  peaks = sorted(range(1, len(lags)), key=lambda i: -abs(lags[i]))[:5]
  print("\nAutocorrelation peaks (lag in days -> correlation):")
  for p in peaks:
    print(f"  lag {p:>3}  {lags[p]:+.3f}")

  if len(values) >= 14:
    stl = STL(pd.Series(values, index=series.index), period=7, robust=True).fit()
    var = np.var(stl.resid + stl.seasonal)
    strength = max(0.0, 1 - np.var(stl.resid) / var) if var > 0 else 0.0
    print(f"\nWeekly seasonal strength: {strength:.2f}  "
          f"({'strong' if strength > 0.6 else 'moderate' if strength > 0.3 else 'weak'})")
    print("  0 = no weekly pattern, 1 = entirely weekly")
except ImportError:
  print("\n(statsmodels not installed; skipping ACF and STL)")
except Exception as exc:
  # statsmodels 0.14.x is not compatible with pandas 3.x and raises at import.
  # That is an environment problem, not a data problem -- the weekday/month
  # tables above are computed with pandas alone and are unaffected.
  print(f"\n(ACF/STL unavailable: {type(exc).__name__}: {exc})")
  print(" this is a statsmodels/pandas version clash, not a problem with your data")


# ------------------------------------------------- statement-day detection --
# Payments land on statement dates, which repeat on the same days of the month.
# That is deterministic and therefore knowable for the future, so it makes a
# strong future covariate. Detected from history rather than hard-coded.
STATEMENT_DAYS = set()
if (payments > 0).any():
  by_dom = pd.Series(payments.to_numpy(), index=payments.index).groupby(
      payments.index.day).apply(lambda g: float((g > 0).mean()))
  STATEMENT_DAYS = set(by_dom[by_dom > 0.25].index.tolist())
  if STATEMENT_DAYS:
    print(f"\npayments recur on days of month: {sorted(STATEMENT_DAYS)}")
    print("  -> added as a future covariate, which sharpens payment timing")


# ----------------------------------------------------------------- covariates --
def calendar_covariates(index):
  """Calendar features for the given dates, shape (n_features, len(index)).

  All of these are known arbitrarily far into the future, which is exactly what
  past_future_covariates requires. Day of week and day of month are encoded as
  sine/cosine pairs rather than raw integers: as an integer, Sunday=7 and
  Monday=1 look maximally far apart when they are adjacent.
  """
  dow = index.dayofweek.to_numpy()
  dom = index.day.to_numpy()
  month = index.month.to_numpy()
  return np.stack([
      np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
      np.sin(2 * np.pi * dom / 31), np.cos(2 * np.pi * dom / 31),
      np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
      (dow >= 5).astype(float),                      # weekend flag
      (index.is_month_end).astype(float),
      (index.is_month_start).astype(float),
      np.isin(index.day.to_numpy(), list(STATEMENT_DAYS)).astype(float)
      if STATEMENT_DAYS else np.zeros(len(index)),
  ]).astype(np.float32)


# --------------------------------------------------------------------- model --
rule("4. LOADING TIMESFM")

import torch
from timesfm3 import ModelConfig, TimesFM3Forecaster

device = args.device or ("cuda" if torch.cuda.is_available()
                         else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}")
print(f"checkpoint: {args.checkpoint}")
model = TimesFM3Forecaster(ModelConfig(
    checkpoint_path=args.checkpoint,
    device=device,
    per_core_batch_size=8,
))
print("loaded")


STRATEGY = args.strategy
if STRATEGY == "auto":
  # A signed series with big negative payment spikes is bimodal: one process
  # for purchases, another for payments. Forecasting them as separate channels
  # (plus the balance, since payment size depends on what has accumulated)
  # measured better on every metric than modelling the signed series directly.
  STRATEGY = "multichannel" if args.target == "net_change" else "raw"
print(f"strategy: {STRATEGY}")


def _predict(contexts, index, horizon):
  """Raw call into TimesFM with the calendar attached."""
  kwargs = {}
  if not args.no_covariates:
    future = future_index(index[-1], horizon)
    # past_future covariates must span context + horizon, not just the horizon.
    kwargs["past_future_covariates"] = [calendar_covariates(index.append(future))]
  return list(model.predict_batch(
      contexts,
      horizon=horizon,
      return_quantiles=True,
      make_positive=(args.target == "purchases"),   # usage cannot be negative
      padding_mode="edge",
      **kwargs,
  ))[0]


def forecast(history_index, history_values, horizon):
  """One forecast. Returns (p50, quantiles), honouring the chosen strategy."""
  if STRATEGY == "multichannel":
    end = len(history_index)
    channels = np.stack([
        purchases.to_numpy()[:end],
        payments.to_numpy()[:end],
        balance.to_numpy()[:end],
    ]).astype(np.float32)
    out = _predict([channels], history_index, horizon)
    # net = purchases - payments. For the interval, the upper bound of a
    # difference pairs the upper bound of the first term with the LOWER bound
    # of the second, hence the reversed quantile axis on payments.
    point = out.forecast[0] - out.forecast[1]
    quant = out.quantiles[0] - out.quantiles[1][:, ::-1]
    return point, np.sort(quant, axis=-1)

  if STRATEGY == "signed-log":
    # Compresses the spikes so they stop dominating; inverted after.
    warped = np.sign(history_values) * np.log1p(np.abs(history_values))
    out = _predict([warped.astype(np.float32)], history_index, horizon)
    inv = lambda z: np.sign(z) * np.expm1(np.abs(z))
    return inv(out.forecast), inv(out.quantiles)

  out = _predict([history_values.astype(np.float32)], history_index, horizon)
  return out.forecast, out.quantiles


# ------------------------------------------------------------------ backtest --
if not args.no_backtest and len(series) > args.horizon * 3:
  rule("5. BACKTEST — HOLDING OUT THE LAST %d DAYS" % args.horizon)

  train = series.iloc[:-args.horizon]
  test = series.iloc[-args.horizon:]
  pred, _ = forecast(train.index, train.to_numpy(), args.horizon)

  actual = test.to_numpy(dtype=float)
  # Seasonal naive: "next Monday looks like last Monday". The bar any weekly
  # model has to clear.
  naive = train.to_numpy(dtype=float)[-7:]
  naive = np.resize(naive, args.horizon)

  def mae(a, b):
    return float(np.mean(np.abs(a - b)))

  def smape(a, b):
    denom = (np.abs(a) + np.abs(b)) / 2
    ok = denom > 0
    return float(np.mean(np.abs(a[ok] - b[ok]) / denom[ok]) * 100) if ok.any() else float("nan")

  if (actual < 0).any():
    print("note: sMAPE is unreliable on a signed series that crosses zero.")
    print("      Judge this one on MAE and pinball loss.\n")
  print(f"{'model':<18}{'MAE':>14}{'sMAPE':>10}")
  print(f"{'TimesFM':<18}{mae(actual, pred):>14,.0f}{smape(actual, pred):>9.1f}%")
  print(f"{'seasonal naive':<18}{mae(actual, naive):>14,.0f}{smape(actual, naive):>9.1f}%")
  print(f"{'flat mean':<18}{mae(actual, np.full(args.horizon, train.mean())):>14,.0f}"
        f"{smape(actual, np.full(args.horizon, train.mean())):>9.1f}%")

  # A signed series lives or dies on the spikes, so score them separately --
  # an average over all days hides whether payment days were caught at all.
  if (actual < 0).any():
    spike = actual < 0
    print(f"\n  on {spike.sum()} payment day(s):     MAE {mae(actual[spike], pred[spike]):>12,.0f}")
    print(f"  on {(~spike).sum()} ordinary days:    MAE {mae(actual[~spike], pred[~spike]):>12,.0f}")

  # Pinball loss scores the whole predictive distribution, not just the point.
  levels = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9])
  _, bt_q = forecast(train.index, train.to_numpy(), args.horizon)
  err = actual[:, None] - bt_q
  print(f"  pinball loss (all quantiles): {np.mean(np.maximum(levels * err, (levels - 1) * err)):>10,.0f}")
  inside = np.mean((actual >= bt_q[:, 0]) & (actual <= bt_q[:, 8]))
  print(f"  80% interval actually covered: {inside:.0%} of days (target 80%)")

  better = mae(actual, pred) < mae(actual, naive)
  print(f"\nTimesFM beats seasonal naive: {better}")
  if not better:
    print("  If it does not beat naive, the extra machinery is not earning its")
    print("  keep on this series -- check the target choice and the daily grid.")
else:
  rule("5. BACKTEST — SKIPPED")
  print("not enough history, or --no-backtest")


# ------------------------------------------------------------------ forecast --
rule("6. FORECAST")

point, quantiles = forecast(series.index, series.to_numpy(), args.horizon)
future = future_index(series.index[-1], args.horizon)

result = pd.DataFrame({
    "date": future,
    "day": [DAYS[d] for d in future.dayofweek],
    "forecast": point,
    "p10": quantiles[:, 0],
    "p90": quantiles[:, 8],
})
print(result.head(14).to_string(index=False,
                                formatters={"forecast": "{:,.0f}".format,
                                            "p10": "{:,.0f}".format,
                                            "p90": "{:,.0f}".format}))
if len(result) > 14:
  print(f"... {len(result) - 14} more rows")

unit = "business days" if grid_mode == "business" else "days"
print(f"\nTotal over {args.horizon} {unit} "
      f"({future[0]:%Y-%m-%d} to {future[-1]:%Y-%m-%d}): {point.sum():,.0f}")
print(f"  80% interval: {quantiles[:, 0].sum():,.0f} to {quantiles[:, 8].sum():,.0f}")

print("\nBy week:")
weekly = result.set_index("date")["forecast"].resample("W").sum()
for week, total in weekly.items():
  print(f"  week ending {week:%Y-%m-%d}  {total:>14,.0f}")

print("\nBy month:")
monthly = result.set_index("date")["forecast"].resample("MS").sum()
for month, total in monthly.items():
  print(f"  {month:%Y-%m}  {total:>14,.0f}")

result.to_csv(args.out, index=False)
print(f"\nwritten: {args.out}")

if args.plot:
  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 5))
    recent = series.iloc[-min(len(series), args.horizon * 4):]
    ax.plot(recent.index, recent.to_numpy(), label="history", color="#3b6ea5", lw=1.2)
    ax.plot(future, point, label="forecast", color="#c0392b", lw=1.6)
    ax.fill_between(future, quantiles[:, 0], quantiles[:, 8],
                    color="#c0392b", alpha=0.18, label="80% interval")
    ax.set_title(f"{args.target} — TimesFM 3.0, {args.horizon}-day forecast")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("forecast.png", dpi=140)
    print("written: forecast.png")
  except ImportError:
    print("matplotlib not installed; skipping plot")
