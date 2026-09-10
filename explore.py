"""
Exploratory analysis: what is actually IN this series before anything forecasts
it. Trend, the four seasonal cycles, holidays, spikes and outliers.

    python explore.py --excel yourfile.xlsx --target growth
    python explore.py --excel yourfile.xlsx --target growth --plot

In Spyder:

    %run explore.py --excel yourfile.xlsx --target growth --plot

Nothing here forecasts. It describes. Read it before trusting any model, and
again whenever a forecast looks wrong -- most bad forecasts are bad data, and
this is where that shows up.

Everything is printed as a table you can read, and --plot writes explore.png.
"""

import argparse

import numpy as np
import pandas as pd

import cc_lib as L
import settings as S

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--excel", required=True)
ap.add_argument("--sheet", default=S.SHEET)
ap.add_argument("--date-col", default=S.COLUMN_DATE)
ap.add_argument("--balance-col", default=S.COLUMN_BALANCE)
ap.add_argument("--currency-col", default=S.COLUMN_CURRENCY)
ap.add_argument("--currency", default=None)
ap.add_argument("--growth-col", default=S.COLUMN_GROWTH)
ap.add_argument("--use-growth", action="store_true")
ap.add_argument("--grid", default=S.DEFAULT_GRID,
                choices=["auto", "business", "calendar"])
ap.add_argument("--target", default="growth",
                choices=["purchases", "net_change", "growth", "balance"])
ap.add_argument("--deflator", default=None, metavar="FILE",
                help="price index file (date + index columns) to divide out "
                     "inflation with, so every year is in today's money. This "
                     "is the better option when you have the figures; see "
                     "fetch_tufe.py. Overrides DEFLATOR_FILE in settings.py")
ap.add_argument("--rescale", type=int, default=None, metavar="DAYS",
                help="adjust for inflation before analysing: divide each day by "
                     "how large a typical day was at that time, measured over "
                     "DAYS. Try 90. Use 0 to force it off")
ap.add_argument("--skip-first", type=int, default=None, metavar="N",
                help="drop N steps from the start (default: settings.py)")
ap.add_argument("--plot", action="store_true", help="write explore.png")
ap.add_argument("--out", default=None, help="save the decomposition to a CSV")
args = ap.parse_args()
if isinstance(args.sheet, str) and args.sheet.strip().lstrip("-").isdigit():
  args.sheet = int(args.sheet)
if args.skip_first is not None:
  S.SKIP_FIRST_STEPS = args.skip_first


def rule(title):
  print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def bar(value, reference, width=34):
  """A crude bar so a column of numbers can be read at a glance."""
  if not np.isfinite(value) or reference == 0:
    return ""
  return "#" * int(max(0, min(width, round(value / reference * width / 2))))


# ------------------------------------------------------------------ loading --
rule("1. THE DATA")
data = L.load(args.excel, sheet=args.sheet, date_col=args.date_col,
              balance_col=args.balance_col, currency_col=args.currency_col,
              currency=args.currency, growth_col=args.growth_col,
              use_growth=args.use_growth, grid=args.grid)
for note in data.notes:
  print(f"  {note}")

y = data.target(args.target).astype(float)
idx = y.index

DEFLATOR = args.deflator or S.DEFLATOR_FILE
RESCALED = args.rescale if args.rescale is not None else S.RESCALE_WINDOW

if DEFLATOR:
  # A price index removes price rises and leaves real growth. Everything is
  # measured against the last date in the history, so the numbers below are in
  # today's lira and a 2019 day can be compared with a 2026 one directly.
  infl, rate, last_known = L.load_deflator(DEFLATOR, idx, 0, data.grid_mode)
  y = pd.Series(y.to_numpy() / infl.reindex(idx).to_numpy(), index=idx)
  print(f"\n  DEFLATED using {DEFLATOR}. Values below are in today's lira.")
  gap = (idx[-1] - pd.Timestamp(last_known)).days
  if gap > 45:
    print(f"  NOTE: that index ends {pd.Timestamp(last_known):%Y-%m}, "
          f"{gap // 30} months short of the data; the rest is filled in at "
          f"{rate * 100:.2f}%/month.")
  RESCALED = 0                      # a price index supersedes the proxy
elif RESCALED:
  # Divide each day by how big a typical day was at that time. Everything below
  # is then measured in "typical days of that era" rather than lira, so a 2019
  # Tuesday and a 2026 Tuesday can be compared directly.
  scale = L.trailing_scale(y.to_numpy(), RESCALED)
  y = pd.Series(y.to_numpy() / scale, index=idx)
  print(f"\n  INFLATION-ADJUSTED using a {RESCALED}-day window.")
  print("  Values below are multiples of a typical day at the time, not lira.")
print(f"\n{len(y):,} steps, {idx[0]:%Y-%m-%d} to {idx[-1]:%Y-%m-%d} "
      f"({(idx[-1] - idx[0]).days / 365.25:.1f} years)")
print(f"\n{'mean':>14}{y.mean():>18,.0f}")
print(f"{'median':>14}{y.median():>18,.0f}")
print(f"{'std dev':>14}{y.std():>18,.0f}")
print(f"{'min':>14}{y.min():>18,.0f}   on {y.idxmin():%Y-%m-%d}")
print(f"{'max':>14}{y.max():>18,.0f}   on {y.idxmax():%Y-%m-%d}")
print(f"{'negative days':>14}{(y < 0).sum():>18,}   ({(y < 0).mean():.0%} of steps)")

# The median is far more useful than the mean here: settlement days are huge and
# rare, and they drag a mean around without describing a typical day.
if abs(y.mean()) > 3 * abs(y.median()) and y.median() != 0:
  print("\n  The mean is far from the median, so a few very large days dominate.")
  print("  Read the median as 'a typical day' and treat the mean with care.")


# -------------------------------------------------------------------- trend --
rule("2. TREND — is the level drifting?")
yearly = y.groupby(idx.year).agg(["median", "mean", "count"])
print(f"{'year':<8}{'typical day':>16}{'mean day':>16}{'days':>8}{'vs first year':>16}")
print("-" * 64)
# Compare against the first year that has enough days to mean anything. A
# part-year at the start would otherwise become the yardstick for everything.
full = yearly[yearly["count"] >= 60]
base = full.index[0] if len(full) else yearly.index[0]
first = yearly.loc[base, "median"]
if not np.isfinite(first) or first == 0:
  first = float(y.median())
  print(f"  (year {base} has no usable typical day; comparing against the "
        f"whole-series median instead)")
for year, row in yearly.iterrows():
  ratio = row["median"] / first if first else float("nan")
  print(f"{year:<8}{row['median']:>16,.0f}{row['mean']:>16,.0f}"
        f"{int(row['count']):>8}{ratio:>15.2f}x")

drift = yearly["median"].iloc[-1] / first if first else float("nan")
print(f"\nA typical day is {drift:.1f}x its {base} size.")
if RESCALED:
  print("  (already inflation-adjusted, so this should sit near 1.0; if it does")
  print("   not, something other than inflation is moving the level)")
if np.isfinite(drift) and drift > 2:
  print("  That is a large drift. Old data describes a different-sized business,")
  print("  which is what RESCALE_WINDOW in settings.py exists to handle. Whether")
  print("  it helps is measurable -- sweep it rather than assuming.")

roll = y.rolling(90, min_periods=30).median()
print(f"\n90-day typical day, sampled through the history:")
for when in pd.date_range(idx[0], idx[-1], periods=min(8, len(y) // 90 or 2)):
  nearest = roll.index[roll.index.get_indexer([when], method="nearest")[0]]
  print(f"  {nearest:%Y-%m}  {roll.loc[nearest]:>16,.0f}")

try:
  from statsmodels.tsa.stattools import adfuller
  stat, pval = adfuller(y.dropna().to_numpy(), autolag="AIC")[:2]
  print(f"\nStationarity test (ADF): p = {pval:.4f}")
  stationary = pval < 0.05
  drifting = np.isfinite(drift) and drift > 1.5
  if stationary and drifting:
    # These disagree often on this kind of data. The test is easily fooled by
    # large repeating settlement spikes, which look like strong mean reversion.
    # The year-by-year table above is the more reliable read.
    print("  The test says stable, but the year-by-year table says the level has")
    print(f"  grown {drift:.1f}x. Trust the table: this test is easily fooled by")
    print("  big repeating settlement days, which look like a return to the mean.")
  elif stationary:
    print("  The series is stationary: it wanders around a stable level.")
  else:
    print("  The series is NOT stationary: the level itself moves over time.")
    print("  Models cope, but old data is then less representative of today.")
except Exception as exc:
  print(f"\n(stationarity test unavailable: {type(exc).__name__})")


# ------------------------------------------------------- seasonal decompose --
rule("3. THE SEASONAL CYCLES — which ones are real?")
print("A cycle's STRENGTH is the share of the wobble it explains, after the")
print("trend is removed. 0 means absent, 1 means the series is nothing else.\n")

period_defs = [("weekly", 7), ("monthly", 30), ("quarterly", 91), ("yearly", 365)]
usable = [(n, p) for n, p in period_defs if len(y) >= p * 2 + 1]
strengths = {}
components = {}
try:
  from statsmodels.tsa.seasonal import MSTL
  fit = MSTL(pd.Series(y.to_numpy(), index=idx),
             periods=[p for _, p in usable]).fit()
  seasonal = fit.seasonal
  if isinstance(seasonal, pd.Series):
    seasonal = seasonal.to_frame()
  resid_var = float(np.var(fit.resid))
  print(f"{'cycle':<12}{'period':>9}{'strength':>11}{'reading':>14}")
  print("-" * 48)
  for (name, period), col in zip(usable, seasonal.columns):
    comp = seasonal[col]
    components[name] = comp
    total = float(np.var(comp + fit.resid))
    st = max(0.0, 1 - resid_var / total) if total > 0 else 0.0
    strengths[name] = st
    verdict = ("strong" if st > 0.6 else "moderate" if st > 0.3
               else "weak" if st > 0.1 else "negligible")
    print(f"{name:<12}{period:>9}{st:>11.2f}{verdict:>14}")
  print(f"\nWhat is left unexplained (noise): {resid_var / float(np.var(y)):.0%} "
        f"of the total variation.")
except Exception as exc:
  print(f"(MSTL unavailable: {type(exc).__name__}: {exc})")
  print("The profiles below are computed with plain averages and still hold.")


# ------------------------------------------------------------------ weekly --
rule("4. WEEKLY CYCLE — which days are busy?")
overall = y.median()
wk = y.groupby(idx.dayofweek).agg(["median", "mean", "count"])
print(f"{'day':<6}{'typical':>16}{'index':>8}{'days':>8}   relative size")
print("-" * 70)
for d, row in wk.iterrows():
  ratio = row["median"] / overall if overall else float("nan")
  print(f"{DAYS[d]:<6}{row['median']:>16,.0f}{ratio:>8.2f}{int(row['count']):>8}   "
        f"{bar(row['median'], overall)}")
print("\nIndex 1.00 is an average day. 1.40 means 40% busier than typical.")


# ----------------------------------------------------------------- monthly --
rule("5. MONTHLY CYCLE — which days of the month matter?")
dm = y.groupby(idx.day).agg(["median", "mean", "count"])
dm["index"] = dm["median"] / overall if overall else np.nan
top = dm.sort_values("median", ascending=False).head(6)
low = dm.sort_values("median").head(6)
print("Busiest days of the month:")
for d, row in top.iterrows():
  print(f"  day {d:>2}   {row['median']:>16,.0f}   index {row['index']:>5.2f}")
print("\nWeakest days of the month (settlements land here):")
for d, row in low.iterrows():
  flag = "  <- a configured payment day" if d in S.PAYMENT_DAYS_OF_MONTH else ""
  print(f"  day {d:>2}   {row['median']:>16,.0f}   index {row['index']:>5.2f}{flag}")

if (data.payments > 0).any():
  by_dom = pd.Series(data.payments.to_numpy(), index=idx).groupby(idx.day)
  share = by_dom.apply(lambda g: float((g > 0).mean()))
  common = share[share > 0.25].sort_values(ascending=False)
  print(f"\nDays of the month where a settlement usually happens:")
  for d, frac in common.items():
    print(f"  day {d:>2}   {frac:>5.0%} of months")
  print(f"\nsettings.py has PAYMENT_DAYS_OF_MONTH = {list(S.PAYMENT_DAYS_OF_MONTH)}")
  if not set(common.index) & set(S.PAYMENT_DAYS_OF_MONTH):
    print("  WARNING: none of those match what the data shows. Fix the setting.")


# ------------------------------------------------------------------ yearly --
rule("6. YEARLY CYCLE — which months are busy?")
if idx[-1].year - idx[0].year >= 1:
  mo = y.groupby(idx.month).agg(["median", "count"])
  print(f"{'month':<7}{'typical':>16}{'index':>8}{'years seen':>12}   relative size")
  print("-" * 74)
  for m, row in mo.iterrows():
    ratio = row["median"] / overall if overall else float("nan")
    n_years = row["count"] / 30.4
    print(f"{MONTHS[m-1]:<7}{row['median']:>16,.0f}{ratio:>8.2f}{n_years:>12.1f}   "
          f"{bar(row['median'], overall)}")
  if idx[-1].year - idx[0].year < 3:
    print("\n  With only a couple of years, a monthly pattern is hard to separate")
    print("  from coincidence. Treat these as suggestive, not established.")
else:
  print("Less than a year of data: no yearly cycle can be measured.")


# ---------------------------------------------------------------- holidays --
rule("7. HOLIDAYS — which ones actually move the numbers?")
hmap = L.holiday_map(idx)
if hmap:
  as_date = np.array([d.date() for d in idx])
  in_range = {d: n for d, n in hmap.items() if idx[0].date() <= d <= idx[-1].date()}
  is_hol = np.array([d in in_range for d in as_date])
  is_eve = np.array([(d + pd.Timedelta(days=1)).date() in in_range for d in idx])
  ordinary = y[~(is_hol | is_eve)]
  base_med = ordinary.median()
  print(f"{len(in_range)} {S.HOLIDAY_COUNTRY} public holidays across the period, "
        f"{len(set(in_range.values()))} distinct ones.\n")

  try:
    from scipy.stats import mannwhitneyu
    have_test = True
  except Exception:
    have_test = False

  # A holiday matters if its days differ from ordinary days by enough to notice
  # AND often enough that it is not one odd year. Tested per holiday, because
  # Kurban Bayramı and Republic Day are not the same kind of event.
  rows = []
  for name in sorted(set(in_range.values())):
    days = {d for d, n in in_range.items() if n == name}
    on = y[[d in days for d in as_date]]
    eve = y[[(d + pd.Timedelta(days=1)).date() in days for d in idx]]
    if len(on) < 3:
      continue
    p_on = np.nan
    if have_test and len(on) >= 5 and len(ordinary) >= 5:
      try:
        p_on = float(mannwhitneyu(on, ordinary, alternative="two-sided")[1])
      except Exception:
        p_on = np.nan
    rows.append({
        "name": name, "days": len(on),
        "on": on.median() / base_med if base_med else np.nan,
        "eve": eve.median() / base_med if (base_med and len(eve)) else np.nan,
        "p": p_on})

  rows.sort(key=lambda r: -abs((r["on"] if np.isfinite(r["on"]) else 1) - 1))
  print(f"{'holiday':<44}{'days':>6}{'on day':>9}{'eve':>8}{'real?':>8}")
  print("-" * 76)
  for r in rows:
    if np.isfinite(r["p"]):
      verdict = "yes" if r["p"] < 0.05 else "unclear"
    else:
      verdict = "too few"
    # A negative multiple means the median day there was a settlement, not
    # spending: the holiday or its eve collides with a payment date. That is a
    # calendar coincidence, not a holiday effect.
    clash = (np.isfinite(r["on"]) and r["on"] < 0) or (np.isfinite(r["eve"]) and r["eve"] < 0)
    print(f"{r['name'][:43]:<44}{r['days']:>6}{r['on']:>8.2f}x"
          f"{r['eve']:>7.2f}x{verdict:>8}" + ("  *" if clash else ""))
  if any((np.isfinite(r["on"]) and r["on"] < 0) or
         (np.isfinite(r["eve"]) and r["eve"] < 0) for r in rows):
    print("\n  * a negative multiple means that day usually carries a")
    print("    settlement, so it collides with a payment date. That is a")
    print("    calendar coincidence, not the holiday moving spending.")

  print("\n'on day' and 'eve' are multiples of an ordinary day: 2.00x means")
  print("double. 'real?' is whether that difference is big and consistent")
  print("enough to be more than chance, not just a one-off year.")
  movers = [r for r in rows if np.isfinite(r["p"]) and r["p"] < 0.05]
  if movers:
    print(f"\n{len(movers)} holiday(s) genuinely move card usage:")
    for r in movers:
      direction = "raises" if r["on"] > 1 else "lowers"
      print(f"  {r['name'][:50]:<52}{direction} it to {r['on']:.2f}x")
    print("\nThese are already covered: USE_HOLIDAYS feeds the holiday, its eve,")
    print("the day after and bridge days to the model as covariates.")
  else:
    print("\nNo single holiday clears the bar on its own. With a handful of")
    print("observations each that is common, and the covariate can still help.")
else:
  print(f"No holidays found for {S.HOLIDAY_COUNTRY!r}.")
  print("If that is wrong, `pip install holidays` -- the covariate is doing nothing.")


# ------------------------------------------------------ memory and outliers --
rule("8. MEMORY AND OUTLIERS")
try:
  from statsmodels.tsa.stattools import acf
  lags = acf(y.to_numpy(), nlags=min(400, len(y) // 3), fft=True)
  peaks = sorted(range(1, len(lags)), key=lambda i: -abs(lags[i]))[:8]
  print("Strongest repeats (how much a day resembles the day N steps earlier):\n")
  for p in sorted(peaks):
    # Longest period first: 365 is also a multiple of 7, and calling it
    # "52x weekly" rather than "yearly" would be useless. Only small multiples
    # are labelled, since "19x weekly" says nothing a reader can act on.
    label = ""
    for period, name, tol in ((365, "yearly", 5), (91, "quarterly", 3),
                              (30, "monthly", 2), (7, "weekly", 1)):
      mult = round(p / period)
      if mult >= 1 and mult <= 4 and abs(p - mult * period) <= tol:
        label = f"  <- {name}" if mult == 1 else f"  <- {mult}x {name}"
        break
    print(f"  {p:>4} steps back   {lags[p]:+.3f}{label}")
  print("\nA repeat at a multiple of 7 is the weekly rhythm, at ~30 the monthly")
  print("settlement rhythm, at ~365 an annual one.")
except Exception as exc:
  print(f"(autocorrelation unavailable: {type(exc).__name__})")

spread = y.quantile(0.75) - y.quantile(0.25)
fence = 3 * spread
big = y[(y - y.median()).abs() > fence]
print(f"\n{len(big):,} unusually large days ({len(big) / len(y):.1%} of the series),")
print(f"meaning more than {fence:,.0f} away from a typical day.")
if len(big):
  print("\nThe largest, with dates:")
  for when, val in big.reindex(big.abs().sort_values(ascending=False).index).head(8).items():
    print(f"  {when:%Y-%m-%d} {DAYS[when.dayofweek]}  {val:>18,.0f}"
          f"  (day {when.day} of the month)")


# ------------------------------------------------------------------ outputs --
if args.out:
  frame = pd.DataFrame({"value": y})
  for name, comp in components.items():
    frame[f"seasonal_{name}"] = comp
  frame.to_csv(args.out)
  print(f"\nwritten: {args.out}")

if args.plot:
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  rows = 3 + len(components)
  fig, axes = plt.subplots(rows, 1, figsize=(14, 2.6 * rows), sharex=False)
  axes[0].plot(idx, y.to_numpy(), lw=0.6, color="#3b6ea5")
  axes[0].set_title(f"{args.target} — the whole series")
  axes[1].plot(idx, y.rolling(90, min_periods=30).median(), lw=1.6, color="#c0392b")
  axes[1].set_title("trend (90-day typical day)")
  for ax, (name, comp) in zip(axes[2:], components.items()):
    ax.plot(idx, comp, lw=0.7, color="#2d8659")
    ax.set_title(f"{name} cycle" +
                 (f"  (strength {strengths[name]:.2f})" if name in strengths else ""))
  axes[-1].bar(range(7), [y[idx.dayofweek == d].median() for d in range(7)],
               color="#7f8c8d")
  axes[-1].set_xticks(range(7)); axes[-1].set_xticklabels(DAYS)
  axes[-1].set_title("typical day by weekday")
  for ax in axes:
    ax.grid(alpha=0.3)
  fig.tight_layout()
  fig.savefig("explore.png", dpi=130)
  print("\nwritten: explore.png")

print("\ndone.")
