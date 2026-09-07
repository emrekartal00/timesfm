"""
================================================================================
TimesFM 3.0 — a complete, runnable guide
================================================================================

Run it:      python timesfm_guide.py
One part:    python timesfm_guide.py --only 5
Offline:     python timesfm_guide.py --offline      (weights must already be cached)

Every section below is executable and prints what it does. Read it top to bottom
once, then copy the section you need.

WHAT THE MODEL IS
  TimesFM 3.0 is a *zero-shot* forecaster. You do not train it. You hand it the
  recent history of a series ("context") and ask for the next N steps
  ("horizon"). It was pretrained on a large corpus of time series and generalises
  to new ones it has never seen.

  It has no idea what your data means. It sees numbers and their order. It does
  not know your timestamps, your calendar, or your units. Anything the model
  should know about the future (a planned promotion, a holiday) has to be passed
  explicitly as a covariate -- see PART 8.

LICENSE
  Code is Apache-2.0. The 3.0 *weights* are non-commercial / non-production only.
  The 2.5 weights remain Apache-2.0 if you need commercial use.
================================================================================
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--only", type=int, help="run just one part, by number")
parser.add_argument("--offline", action="store_true", help="forbid all Hub access")
args = parser.parse_args()

SELECTED = args.only


def part(n: int, title: str) -> bool:
  """Prints a header and says whether this part should run."""
  if SELECTED is not None and SELECTED != n:
    return False
  print(f"\n{'=' * 78}\nPART {n}. {title}\n{'=' * 78}")
  return True


# ==============================================================================
# PART 1. INSTALL AND IMPORTS
# ==============================================================================
# Install once, from a clone of this repo:
#
#     pip install -e ".[torch]"
#
# Needs Python >= 3.10. The [torch] extra is a no-op if you already have PyTorch.
# The package installs two importable names:
#
#     timesfm3   -> TimesFM 3.0   (this guide)
#     timesfm    -> TimesFM 2.5   (previous generation, different API)
#
# Be careful not to mix them up. The 3.0 API lives in `timesfm3`.

import numpy as np
import torch

from timesfm3 import ForecastOutput, ModelConfig, TimesFM3Evaluator, TimesFM3Forecaster

if part(1, "INSTALL AND IMPORTS"):
  print(f"torch      {torch.__version__}")
  print(f"numpy      {np.__version__}")
  print("timesfm3   imported OK")


# ==============================================================================
# PART 2. CHOOSING A DEVICE
# ==============================================================================
# TimesFM runs on CUDA, Apple Silicon (MPS), or CPU. If you pass device=None the
# library only ever picks "cuda" or "cpu" -- it will NOT auto-detect MPS, so on a
# Mac you must ask for "mps" explicitly or you silently get slow CPU inference.
# Hence this helper.


def pick_device() -> str:
  """Best available device. Note the explicit MPS check -- the library omits it."""
  if torch.cuda.is_available():
    return "cuda"
  if torch.backends.mps.is_available():
    return "mps"
  return "cpu"


DEVICE = pick_device()

if part(2, "CHOOSING A DEVICE"):
  print(f"cuda available : {torch.cuda.is_available()}")
  print(f"mps available  : {torch.backends.mps.is_available()}")
  print(f"selected       : {DEVICE}")


# ==============================================================================
# PART 3. LOADING THE MODEL
# ==============================================================================
# Everything is configured through ModelConfig. Only a handful of fields matter
# in normal use; the rest are architecture knobs that must match the checkpoint,
# so leave them alone unless you know why you are changing them.
#
#   checkpoint_path      HF repo id, or a local .safetensors/.pth, or a directory.
#   device               "cuda" / "mps" / "cpu".
#   per_core_batch_size  How many series go through one forward pass. Raise it to
#                        go faster, lower it if you run out of memory.
#   local_files_only     True = never contact the network (see PART 4).
#   cache_dir            Where weights live. Default ~/.cache/huggingface.
#   token / revision     For gated or pinned checkpoints. 3.0 is ungated.
#
# THE CHECKPOINT IS 1.3 GB. The first load downloads it; later loads read the
# cache in about 3 seconds. Load the model ONCE and reuse it -- never construct
# it inside a loop.
#
# --- Which class? ------------------------------------------------------------
# There are two, and the difference is easy to miss because the README uses the
# second one:
#
#   TimesFM3Forecaster   The plain API. Conservative defaults, everything off.
#                        Use this for your own forecasting.
#
#   TimesFM3Evaluator    A subclass built for reproducing benchmark numbers. It
#                        flips on return_quantiles, use_symmetric_averaging and
#                        make_positive by default, adds a univariate= flag, and
#                        chunks inputs with more than 32 variates automatically.
#                        Symmetric averaging DOUBLES inference cost.
#
# They share the same predict_batch, so this guide uses the Forecaster and passes
# flags explicitly -- what you see is what you get.

if part(3, "LOADING THE MODEL"):
  print("loading google/timesfm-3.0-pytorch ...")

config = ModelConfig(
    checkpoint_path="google/timesfm-3.0-pytorch",
    device=DEVICE,
    per_core_batch_size=8,
    local_files_only=args.offline,
)
model = TimesFM3Forecaster(config)

if part(3, "LOADING THE MODEL (cont.)"):
  print(f"loaded on {DEVICE}")
  print(f"max context the model accepts : {model.global_context} steps")
  print(f"quantile levels               : {config.quantiles}")
  print(f"index used as the point forecast: {config.median_quantile_index} (= p50)")


# ==============================================================================
# PART 4. RUNNING WITH NO NETWORK
# ==============================================================================
# Inference itself makes ZERO network calls -- verified by blocking the socket
# layer and confirming predict_batch still runs.
#
# The catch is at LOAD time: from_pretrained pings huggingface.co to check
# whether your cached copy is stale, even when the weights are already on disk.
# That happens before any of your data exists in the process, so it cannot leak
# your series -- it sends the repo id, a user-agent and your IP. But if you want
# genuinely zero traffic, suppress it. Two ways:
#
#   ModelConfig(..., local_files_only=True)     <- preferred, explicit, scoped
#   export HF_HUB_OFFLINE=1                     <- global; must be set BEFORE
#                                                  timesfm3 is imported
#
# Either way the weights must already be cached, so the very first run on a new
# machine has to be online.

if part(4, "RUNNING WITH NO NETWORK"):
  print(f"local_files_only currently: {config.local_files_only}")
  print("your series never leave this process during predict_batch")


# ==============================================================================
# PART 5. INPUT FORMAT — THE PART THAT ACTUALLY TRIPS PEOPLE UP
# ==============================================================================
# predict_batch takes a LIST of arrays. The list is the batch. What each element
# looks like decides whether you get univariate or multivariate behaviour:
#
#   1-D array, shape (context_len,)          -> one univariate series
#   2-D array, shape (n_variates, context_len) -> ONE multivariate series whose
#                                                 channels are forecast jointly
#
# So [a, b] with 1-D a,b means "two independent series". A single 2-D array
# means "one series with several correlated channels". This is the distinction
# people get wrong most often.
#
# RULES
#   * Series may have DIFFERENT lengths. Ragged batches are fine.
#   * Every element must have the SAME number of variates. Mixing a 1-variate
#     and a 3-variate context in one call raises ValueError.
#   * Data is float32. Lists and float64 arrays are converted for you.
#   * TIME RUNS LEFT TO RIGHT. The last element is the most recent observation.
#     Sort ascending by timestamp. Feeding it backwards produces confident
#     nonsense -- there is no error, so check this.
#
# LENGTH HANDLING (automatic)
#   Longer than 15360 -> truncated, keeping the MOST RECENT 15360 steps.
#   Shorter           -> left-padded with a mask, so padding is ignored.
#   You never need to pad or trim yourself.
#
# HOW MUCH CONTEXT? More is usually better, up to a point. Give it at least a few
# full seasonal cycles -- for hourly data with daily seasonality, several days.

if part(5, "INPUT FORMAT"):
  t = np.arange(300, dtype=np.float32)

  univariate = np.sin(2 * np.pi * t / 24) + 0.02 * t          # (300,)
  short_one = np.sin(2 * np.pi * np.arange(80) / 24)           # (80,)  ragged
  multivariate = np.stack([                                    # (3, 300)
      np.sin(2 * np.pi * t / 24),
      np.cos(2 * np.pi * t / 24),
      0.02 * t,
  ]).astype(np.float32)

  print(f"univariate   {univariate.shape}  -> one series")
  print(f"short_one    {short_one.shape}   -> shorter, batches fine with the above")
  print(f"multivariate {multivariate.shape} -> ONE series, 3 joint channels")

  # Two independent univariate series of different lengths, in one call.
  out = list(model.predict_batch([univariate, short_one], horizon=12))
  print(f"\nbatch of 2 univariate -> {len(out)} outputs, "
        f"shapes {out[0].forecast.shape} and {out[1].forecast.shape}")

  # One multivariate series: all 3 channels forecast together.
  out_mv = list(model.predict_batch([multivariate], horizon=12))
  print(f"1 multivariate (3 ch) -> forecast shape {out_mv[0].forecast.shape}  (variates, horizon)")


# ==============================================================================
# PART 6. GETTING YOUR DATA IN (PARSING)
# ==============================================================================
# TimesFM wants plain NumPy. It never sees your timestamps -- it assumes your
# samples are already evenly spaced. Your job before calling it:
#
#   1. sort ascending by time
#   2. resample onto a regular grid (hourly, daily, ...) -- gaps must become rows
#   3. hand over the values column
#
# Step 2 matters. If your rows are irregularly spaced, the model still treats
# them as evenly spaced and the forecast is meaningless. Resample first.
#
#   df = pd.read_csv("sales.csv", parse_dates=["date"])
#   df = df.sort_values("date")
#   s  = df.set_index("date")["units"].resample("D").sum()   # regular daily grid
#   context = s.to_numpy(dtype=np.float32)
#
# For many series at once (per store, per SKU) group first, then batch:
#
#   groups   = list(df.groupby("store_id"))
#   contexts = [g["units"].to_numpy(np.float32) for _, g in groups]
#   ids      = [str(k) for k, _ in groups]
#   outputs  = list(model.predict_batch(contexts, horizon=28, ts_ids=ids))
#
# --- MISSING VALUES ---------------------------------------------------------
# Leave NaNs in. They are handled for you, in this order:
#   * leading all-NaN steps are trimmed off the front
#   * remaining NaNs are filled by linear interpolation
#   * an all-NaN series becomes zeros (you get a flat forecast -- check for this)
# Do NOT fill with 0 yourself; a real 0 is a data point and will be modelled.

if part(6, "GETTING YOUR DATA IN"):
  with_gaps = np.sin(2 * np.pi * np.arange(200) / 24).astype(np.float32)
  with_gaps[:5] = np.nan       # leading NaNs -> trimmed
  with_gaps[100:110] = np.nan  # interior gap -> interpolated
  print(f"context has {np.isnan(with_gaps).sum()} NaNs of {len(with_gaps)} steps")

  out = list(model.predict_batch([with_gaps], horizon=8))[0]
  print(f"forecast ran anyway, no NaNs out: {not np.isnan(out.forecast).any()}")
  print(f"first values: {np.round(out.forecast[:4], 4)}")


# ==============================================================================
# PART 7. THE OUTPUT
# ==============================================================================
# predict_batch returns a GENERATOR, not a list. Nothing happens until you
# consume it -- wrap it in list() or iterate. Outputs come back in input order.
#
# Each item is a ForecastOutput with three fields:
#
#   .ts_id       whatever you passed in ts_ids, else None
#   .forecast    the point forecast (the p50 / median)
#   .quantiles   all 9 quantiles, or None unless return_quantiles=True
#
# SHAPES depend on what you put in:
#
#   1-D input ->  forecast (horizon,)              quantiles (horizon, 9)
#   2-D input ->  forecast (n_variates, horizon)   quantiles (n_variates, horizon, 9)
#
# QUANTILES are the nine deciles [0.1 ... 0.9]. Index 4 is the median, which is
# exactly what .forecast contains. So:
#
#   q[:, 0]  p10   |  q[:, 4]  p50 == forecast  |  q[:, 8]  p90
#
# p10..p90 is an 80% interval. There is no p05/p95 -- if you need a wider band
# you cannot get it from this head.
#
# HORIZON: ask for any length. Internally it is rounded up to a multiple of 64
# and then sliced back down, so horizon=10 costs the same as horizon=64. Accuracy
# degrades the further out you go; the model was trained to 1k horizon.

if part(7, "THE OUTPUT"):
  series = (np.sin(2 * np.pi * np.arange(400) / 24) * 50 + 500).astype(np.float32)

  result = list(model.predict_batch(
      [series],
      horizon=6,
      return_quantiles=True,   # off by default on TimesFM3Forecaster!
      ts_ids=["demo-series"],
  ))[0]

  print(f"ts_id     {result.ts_id}")
  print(f"forecast  {result.forecast.shape}")
  print(f"quantiles {result.quantiles.shape}  (horizon, 9 deciles)\n")

  print(f"{'step':>4} {'p10':>9} {'p50':>9} {'p90':>9}   80% interval")
  for i, (point, q) in enumerate(zip(result.forecast, result.quantiles), 1):
    print(f"{i:>4} {q[0]:>9.2f} {point:>9.2f} {q[8]:>9.2f}   [{q[0]:.1f}, {q[8]:.1f}]")

  same = np.allclose(result.forecast, result.quantiles[:, 4])
  print(f"\n.forecast is exactly quantiles[:, 4]: {same}")


# ==============================================================================
# PART 8. COVARIATES — TELLING THE MODEL ABOUT THE FUTURE
# ==============================================================================
# New in 3.0, and the main reason to use it over 2.5. Two kinds:
#
#   past_only_covariates      shape (n_cov, context_len)
#       Things you measured but cannot know in advance: yesterday's weather,
#       realised traffic, observed competitor price.
#
#   past_future_covariates    shape (n_cov, context_len + horizon)
#       Things you DO know in advance: day of week, holiday flags, a promotion
#       calendar, scheduled price changes. This is the powerful one -- it lets
#       you say "next Tuesday is a holiday" and have the forecast respond.
#
# Both are passed as a LIST parallel to contexts, one entry per series, and any
# entry may be None if that series has no covariates.
#
# THE SHAPE RULE that causes most errors: past_future_covariates must be
# context_len + horizon long. Not horizon. The whole span, past and future.
#
# padding_mode: horizon is rounded up to a multiple of 64 internally, so a
# future covariate of exactly `horizon` can come up short. "edge" repeats the
# last value to cover the gap; "none" (default) leaves it and slices to fit.
# Pass padding_mode="edge" if you hit a length complaint here.
#
# Covariates are optional. Plenty of good forecasts use none.

if part(8, "COVARIATES"):
  ctx_len, horizon = 200, 24
  base = np.arange(ctx_len + horizon)

  # Target: demand with a weekly cycle, boosted whenever a promotion runs.
  promo = (base % 7 == 5).astype(np.float32)          # known for past AND future
  demand = (100 + 10 * np.sin(2 * np.pi * base / 7) + 40 * promo).astype(np.float32)

  target = demand[:ctx_len]                            # (200,)
  weather = np.random.RandomState(0).randn(1, ctx_len).astype(np.float32)  # past only
  promo_cov = promo.reshape(1, -1)                     # (1, 224) = context+horizon

  print(f"target                 {target.shape}")
  print(f"past_only  (weather)   {weather.shape}          = (n_cov, context)")
  print(f"past_future (promo)    {promo_cov.shape}          = (n_cov, context+horizon)")

  out = list(model.predict_batch(
      contexts=[target],
      horizon=horizon,
      past_only_covariates=[weather],
      past_future_covariates=[promo_cov],
      return_quantiles=True,
      padding_mode="edge",
  ))[0]

  future_promo = promo[ctx_len:]
  print(f"\nforecast {out.forecast.shape}")
  print(f"mean on promo days   : {out.forecast[future_promo == 1].mean():.1f}")
  print(f"mean on normal days  : {out.forecast[future_promo == 0].mean():.1f}")
  print("(the gap shows the future covariate is being used)")


# ==============================================================================
# PART 9. THE FLAGS ON predict_batch
# ==============================================================================
# Defaults below are those of TimesFM3Forecaster. TimesFM3Evaluator differs --
# see PART 3.
#
#   return_quantiles=False
#       Off by default. If .quantiles comes back None, this is why.
#
#   use_znorm=False
#       Z-normalise each series before the model and undo it after. Try it if a
#       series has an extreme scale or offset and the forecast looks off.
#
#   make_positive=False
#       Clamp the forecast at 0, but only for series whose context is entirely
#       non-negative. Right for counts, sales, volumes -- anything that cannot
#       go below zero. It will not clamp a series that already has negatives.
#
#   sort_quantiles=True
#       Sorts quantiles so p10 <= p50 <= p90. Quantile crossing is possible
#       without it. Leave this on.
#
#   use_symmetric_averaging=False
#       Also forecasts the negated series and averages, cancelling directional
#       bias. Slightly better, and DOUBLE the compute. Benchmarks use it.
#
#   padding_mode="none"
#       "none" or "edge". Only affects future covariates -- see PART 8.
#
#   ts_ids=None
#       Labels echoed back on each output. Use them when batching many series so
#       you can tell which forecast is which.

if part(9, "THE FLAGS"):
  counts = np.maximum(0, np.sin(np.linspace(0, 30, 200)) * 5 + 3).astype(np.float32)

  plain = list(model.predict_batch([counts], horizon=12))[0]
  guarded = list(model.predict_batch(
      [counts], horizon=12, make_positive=True, use_symmetric_averaging=True,
  ))[0]

  print(f"default          -> min {plain.forecast.min():.3f}")
  print(f"make_positive    -> min {guarded.forecast.min():.3f}  (never negative)")
  print("symmetric averaging also on above: ~2x the compute")


# ==============================================================================
# PART 10. DOING IT AT SCALE
# ==============================================================================
#   * Load the model ONCE. Construction reads 1.3 GB; reuse the object.
#   * Batch. One call with 500 series is far faster than 500 calls.
#   * per_core_batch_size is the real throughput knob. Raise it until memory
#     complains, then back off. 8-32 is a reasonable range.
#   * predict_batch is a generator, so you can stream results into a file
#     without holding every forecast in memory.
#   * Keep batches roughly equal in length. Each batch is padded to its longest
#     member, so one 15k-step series among 10k-step ones wastes compute.
#   * Cost scales with context length. Do not pass 15k steps if 1k forecasts
#     just as well -- measure it.

if part(10, "DOING IT AT SCALE"):
  import time

  rng = np.random.RandomState(0)
  many = [
      (np.sin(np.linspace(0, 30, 200)) + rng.randn(200) * 0.1).astype(np.float32)
      for _ in range(24)
  ]
  ids = [f"series-{i:03d}" for i in range(24)]

  start = time.time()
  results = list(model.predict_batch(many, horizon=24, ts_ids=ids, return_quantiles=True))
  elapsed = time.time() - start

  print(f"{len(results)} series, horizon 24 -> {elapsed:.2f}s "
        f"({elapsed / len(results) * 1000:.0f} ms per series)")
  print(f"ids preserved: {results[0].ts_id} ... {results[-1].ts_id}")


# ==============================================================================
# PART 11. GETTING RESULTS BACK OUT
# ==============================================================================
# There is no built-in export. Build the frame yourself. Remember the model has
# no timestamps -- you regenerate the future index from your own data.
#
#   last = s.index[-1]                                   # your last timestamp
#   idx  = pd.date_range(last, periods=horizon + 1, freq="D")[1:]
#   out  = pd.DataFrame({
#       "date": idx,
#       "forecast": result.forecast,
#       "p10": result.quantiles[:, 0],
#       "p90": result.quantiles[:, 8],
#   })
#   out.to_csv("forecast.csv", index=False)
#
# Many series at once:
#
#   rows = []
#   for r in results:
#       for h, (point, q) in enumerate(zip(r.forecast, r.quantiles), start=1):
#           rows.append({"id": r.ts_id, "step": h, "forecast": point,
#                        "p10": q[0], "p90": q[8]})
#   pd.DataFrame(rows).to_csv("forecasts.csv", index=False)

if part(11, "GETTING RESULTS BACK OUT"):
  try:
    import pandas as pd

    s = (np.sin(np.linspace(0, 30, 200)) * 10 + 100).astype(np.float32)
    r = list(model.predict_batch([s], horizon=5, return_quantiles=True))[0]

    idx = pd.date_range("2026-01-01", periods=len(s), freq="D")
    future = pd.date_range(idx[-1], periods=6, freq="D")[1:]

    frame = pd.DataFrame({
        "date": future,
        "forecast": r.forecast,
        "p10": r.quantiles[:, 0],
        "p90": r.quantiles[:, 8],
    })
    print(frame.to_string(index=False))
  except ImportError:
    print("pandas not installed; skipping")


# ==============================================================================
# PART 12. THINGS THAT WILL BITE YOU
# ==============================================================================
# 1. .quantiles is None            -> pass return_quantiles=True.
# 2. Empty results                 -> predict_batch is a generator; call list().
# 3. Silently slow on a Mac        -> device=None never picks MPS. Pass "mps".
# 4. Reversed forecasts            -> context must be oldest-first. No error is
#                                     raised if you feed it backwards.
# 5. Irregular timestamps          -> resample to a fixed grid first. The model
#                                     assumes even spacing and cannot tell.
# 6. Covariate length errors       -> past_future needs context+horizon, and
#                                     padding_mode="edge" covers the rounding.
# 7. "All contexts must have the
#     same number of target variates" -> you mixed 1-D and 2-D in one batch.
# 8. Flat / zero forecast          -> your context was probably all-NaN, which
#                                     silently becomes zeros.
# 9. Reloading in a loop           -> construct the model once.
# 10. Different numbers vs the
#     paper                        -> benchmarks use TimesFM3Evaluator defaults
#                                     (symmetric averaging + make_positive).

if part(12, "THINGS THAT WILL BITE YOU"):
  print("see the comments above -- 10 common failure modes")
  print("\nMinimal correct call:\n")
  print('    model = TimesFM3Forecaster(ModelConfig(')
  print('        checkpoint_path="google/timesfm-3.0-pytorch",')
  print('        device="mps",          # or "cuda" / "cpu"')
  print('    ))')
  print('    out = list(model.predict_batch(')
  print('        [context_array],       # oldest -> newest, float32')
  print('        horizon=24,')
  print('        return_quantiles=True,')
  print('    ))[0]')
  print('    out.forecast       # (24,)')
  print('    out.quantiles      # (24, 9), deciles p10..p90')

print("\ndone.")
