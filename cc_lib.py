"""Shared machinery for credit-card forecasting.

Kept separate from any command-line interface so both `credit_card_forecast.py`
(one run, readable output) and `sweep.py` (many runs, ranked table) use exactly
the same data preparation and models. If they diverged, their numbers would not
be comparable.

Nothing here prints unless you pass verbose=True.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pandas as pd

import settings as S

QUANTILE_LEVELS = np.array([0.1, 0.5, 0.9])


# =============================================================== data loading ==
@dataclasses.dataclass
class Series:
  """One currency's history, on a regular grid, with the pieces split out."""
  index: pd.DatetimeIndex
  balance: pd.Series
  net_change: pd.Series
  purchases: pd.Series
  payments: pd.Series
  statement_days: set
  grid_mode: str
  currency: str | None
  notes: list

  def target(self, name):
    # "growth" is the name spreadsheets use for the daily change; "net_change"
    # is the same thing. Both are accepted so nobody has to learn a second word
    # for a column they already have.
    return {"purchases": self.purchases,
            "net_change": self.net_change,
            "growth": self.net_change,
            "balance": self.balance}[name]


def _find_col(df, override, *keywords):
  if override:
    if override not in df.columns:
      raise ValueError(f"No column named {override!r}. Have: {list(df.columns)}")
    return override
  for col in df.columns:
    if any(k in str(col).strip().lower() for k in keywords):
      return col
  return None


def load(excel, sheet=None, date_col=None, balance_col=None, currency_col=None,
         currency=None, growth_col=None, use_growth=False, grid=None):
  # Defaults come from settings.py so calling load() directly behaves the same
  # way the scripts do, rather than silently reading a different sheet.
  """Read a sheet and return a Series ready to forecast."""
  notes = []
  sheet = S.SHEET if sheet is None else sheet
  grid = S.DEFAULT_GRID if grid is None else grid
  date_col = date_col or S.COLUMN_DATE
  balance_col = balance_col or S.COLUMN_BALANCE
  currency_col = currency_col or S.COLUMN_CURRENCY
  growth_col = growth_col or S.COLUMN_GROWTH
  df = pd.read_excel(excel, sheet_name=sheet)

  date_col = _find_col(df, date_col, "date", "tarih")
  bal_col = _find_col(df, balance_col, "balance", "bakiye", "amount", "tutar")
  cur_col = _find_col(df, currency_col, "currency", "curr", "para", "doviz", "döviz")
  if date_col is None or bal_col is None:
    raise ValueError(f"Could not find date/balance columns in {list(df.columns)}. "
                     "Pass date_col= and balance_col= explicitly.")
  notes.append(f"date={date_col!r} balance={bal_col!r} currency={cur_col!r}")

  df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
  dropped = int(df[date_col].isna().sum())
  if dropped:
    notes.append(f"dropped {dropped} rows with unparseable dates")
    df = df.dropna(subset=[date_col])

  chosen = None
  if cur_col is not None and df[cur_col].nunique() > 1:
    counts = df[cur_col].value_counts()
    chosen = currency or counts.index[0]
    if chosen not in set(df[cur_col]):
      raise ValueError(f"Currency {chosen!r} not in the sheet. Have: {list(counts.index)}")
    notes.append(f"{df[cur_col].nunique()} currencies; modelling {chosen!r} "
                 f"({counts[chosen]:,} rows). Currencies are never mixed -- "
                 f"the numbers are not comparable.")
    df = df[df[cur_col] == chosen]
  elif cur_col is not None:
    chosen = df[cur_col].iloc[0]

  df = df.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
  df[bal_col] = pd.to_numeric(df[bal_col], errors="coerce")
  raw = df.set_index(date_col)[bal_col]

  # ---- which grid? TimesFM assumes every step is one period, so the rows must
  # sit on a regular one. Whether weekends belong depends on the export, so it
  # is decided from the data rather than assumed.
  weekend_rows = int((raw.index.dayofweek >= 5).sum())
  mode = grid
  if mode == "auto":
    if weekend_rows == 0:
      mode = "business"
      notes.append("no weekend rows -> business-day grid")
    else:
      wk = raw.diff().loc[raw.index.dayofweek >= 5]
      scale = max(raw.diff().abs().median(), 1e-9)
      active = float((wk.abs() > 0.05 * scale).mean())
      mode = "calendar" if active > 0.10 else "business"
      notes.append(f"{weekend_rows:,} weekend rows, {active:.0%} carry movement "
                   f"-> {mode} grid" +
                   ("" if mode == "calendar" else " (they are padding)"))

  grid_index = (pd.bdate_range(raw.index.min(), raw.index.max()) if mode == "business"
                else pd.date_range(raw.index.min(), raw.index.max(), freq="D"))

  balance = raw.reindex(raw.index.union(grid_index)).ffill().reindex(grid_index)
  if balance.isna().any():
    balance = balance.bfill()

  recomputed = balance.diff().fillna(0.0)
  gcol = _find_col(df, growth_col, "growth", "artis", "artış", "degisim")
  net = recomputed
  if gcol is not None and gcol in df.columns:
    supplied = pd.to_numeric(df.set_index(date_col)[gcol], errors="coerce").reindex(grid_index)
    known = supplied.notna() & recomputed.notna()
    if known.sum():
      scale = max(recomputed.abs().mean(), 1e-9)
      agree = float(((supplied[known] - recomputed[known]).abs() < 0.01 * scale).mean())
      if use_growth:
        net = supplied.fillna(recomputed)
        notes.append(f"using sheet column {gcol!r} (agrees with balance.diff() "
                     f"on {agree:.0%} of steps)")
        if agree < 0.95:
          notes.append("WARNING: >5% disagreement. A stale fill-down or an "
                       "inserted row does that; recomputing is safer.")
      elif agree >= 0.9999:
        notes.append(f"found {gcol!r}: identical to balance.diff() on every "
                     f"step, so --use-growth would change nothing. Skip it.")
      else:
        notes.append(f"found {gcol!r}, agrees on {agree:.0%} of steps; using the "
                     f"recomputed version (--use-growth to prefer the sheet's)")

  net = net.fillna(0.0)
  purchases = net.clip(lower=0.0)
  payments = (-net).clip(lower=0.0)

  # Payments recur on fixed days of the month, which is knowable in advance and
  # therefore a strong future covariate.
  stmt = set()
  if (payments > 0).any():
    by_dom = pd.Series(payments.to_numpy(), index=grid_index).groupby(
        grid_index.day).apply(lambda g: float((g > 0).mean()))
    stmt = set(by_dom[by_dom > 0.25].index.tolist())
    if stmt:
      notes.append(f"payments recur on days of month {sorted(stmt)}")

  # Say plainly whether the holiday and schedule covariates are live. Both fail
  # quietly if the package is missing or the country code is wrong, and a silent
  # failure here cost 42% of accuracy in testing.
  if S.USE_HOLIDAYS and S.HOLIDAY_COUNTRY:
    found = holiday_dates(grid_index)
    inside = sum(1 for d in found if grid_index[0].date() <= d <= grid_index[-1].date())
    if inside:
      notes.append(f"holidays: {inside} {S.HOLIDAY_COUNTRY} public holidays in range")
    else:
      notes.append(f"holidays: NONE found for {S.HOLIDAY_COUNTRY!r} -- is the "
                   f"`holidays` package installed? covariate is doing nothing")
  if S.USE_SCHEDULED_PAYMENT_DAYS and S.PAYMENT_DAYS_OF_MONTH:
    notes.append(f"scheduled payment days: {list(S.PAYMENT_DAYS_OF_MONTH)} "
                 f"(plus the working day either side when one falls on a "
                 f"weekend or holiday)")

  return Series(grid_index, balance, net, purchases, payments, stmt, mode, chosen, notes)


def future_index(last, n, grid_mode):
  if grid_mode == "business":
    return pd.bdate_range(last + pd.Timedelta(days=1), periods=n)
  return pd.date_range(last + pd.Timedelta(days=1), periods=n, freq="D")


# ==================================================================== metrics ==
def pinball(actual, quant, levels=QUANTILE_LEVELS):
  err = actual[:, None] - quant
  return float(np.mean(np.maximum(levels * err, (levels - 1) * err)))


def score(actual, point, quant):
  spike = actual < 0
  return {
      "mae": float(np.mean(np.abs(actual - point))),
      "pinball": pinball(actual, quant),
      "coverage": float(np.mean((actual >= quant[:, 0]) & (actual <= quant[:, 2]))),
      "mae_spike": float(np.mean(np.abs(actual - point)[spike])) if spike.any() else float("nan"),
      "mae_other": float(np.mean(np.abs(actual - point)[~spike])) if (~spike).any() else float("nan"),
  }


# ===================================================================== models ==
_TIMESFM_CACHE = {}


def _timesfm_model(checkpoint, device, batch_size, offline=False):
  """Loaded once per process -- the checkpoint is 1.3 GB."""
  import torch
  from timesfm3 import ModelConfig, TimesFM3Forecaster
  if device is None:
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
  key = (checkpoint, device, batch_size, offline)
  if key not in _TIMESFM_CACHE:
    # local_files_only stops the Hub revision check that otherwise runs even
    # when the weights are already cached. Harmless with internet, and the
    # difference between working and hanging without it.
    _TIMESFM_CACHE[key] = TimesFM3Forecaster(ModelConfig(
        checkpoint_path=checkpoint, device=device,
        per_core_batch_size=batch_size, local_files_only=offline))
  return _TIMESFM_CACHE[key], device



# ------------------------------------------------- holidays and payment dates --
_HOLIDAY_CACHE = {}


def holiday_dates(index):
  """Public holidays covering the index and a year beyond it, for the horizon.

  Uses the `holidays` package, so Turkey's moving religious holidays (Eid al-Fitr
  and Eid al-Adha shift about eleven days earlier each year) are handled -- a
  hard-coded list would silently go wrong after one year.
  """
  if not S.USE_HOLIDAYS or not S.HOLIDAY_COUNTRY:
    return set()
  years = tuple(range(index[0].year, index[-1].year + 2))
  key = (S.HOLIDAY_COUNTRY, years)
  if key not in _HOLIDAY_CACHE:
    try:
      import holidays as _h
      _HOLIDAY_CACHE[key] = set(
          _h.country_holidays(S.HOLIDAY_COUNTRY, years=list(years)).keys())
    except Exception:
      _HOLIDAY_CACHE[key] = set()      # package missing or country unsupported
  return _HOLIDAY_CACHE[key]


def _shift(day, holis, direction):
  """Walk off weekends and holidays in `direction` until a working day."""
  for _ in range(10):
    if day.weekday() < 5 and day.date() not in holis:
      return day
    day = day + pd.Timedelta(days=direction)
  return day


def scheduled_payment_dates(index, holis):
  """The nominal payment days, and where they land once weekends and holidays
  push them.

  Commercial card payments fall on fixed days of the month -- the 4th, 14th and
  24th by default -- but a date landing on a Sunday or a public holiday moves to
  a working day, which is why the 14th shows up as the 13th or the 15th. Both
  directions are returned because the convention differs by bank, and giving the
  model both lets it work out which one your data follows.
  """
  nominal, earlier, later = set(), set(), set()
  if not S.USE_SCHEDULED_PAYMENT_DAYS or not S.PAYMENT_DAYS_OF_MONTH:
    return nominal, earlier, later
  months = pd.period_range(index[0], index[-1] + pd.Timedelta(days=400), freq="M")
  for month in months:
    for dom in S.PAYMENT_DAYS_OF_MONTH:
      try:
        day = pd.Timestamp(year=month.year, month=month.month, day=int(dom))
      except ValueError:
        continue                        # e.g. the 31st of a 30-day month
      nominal.add(day.date())
      earlier.add(_shift(day, holis, -1).date())
      later.add(_shift(day, holis, +1).date())
  return nominal, earlier, later


def calendar_covariates(index, statement_days):
  """Features knowable arbitrarily far ahead, so they may span the horizon.

  Day of week and day of month are sine/cosine pairs rather than raw integers:
  as integers, Sunday=7 and Monday=1 look maximally far apart when adjacent.
  """
  dow, dom, month = index.dayofweek.to_numpy(), index.day.to_numpy(), index.month.to_numpy()
  feats = []
  if S.USE_DAY_OF_WEEK:
    feats += [np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)]
  if S.USE_DAY_OF_MONTH:
    feats += [np.sin(2 * np.pi * dom / 31), np.cos(2 * np.pi * dom / 31)]
  if S.USE_MONTH_OF_YEAR:
    feats += [np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)]
  if S.USE_WEEKEND_FLAG:
    feats.append((dow >= 5).astype(float))
  if S.USE_MONTH_EDGES:
    feats += [index.is_month_end.astype(float), index.is_month_start.astype(float)]
  if S.USE_STATEMENT_DAYS:
    feats.append(np.isin(dom, list(statement_days)).astype(float)
                 if statement_days else np.zeros(len(index)))

  holis = holiday_dates(index)
  as_dates = np.array([d.date() for d in index])
  if S.USE_HOLIDAYS and holis:
    is_hol = np.array([d in holis for d in as_dates], dtype=float)
    # The eve of a holiday is often the busiest day of all, and the working day
    # stranded between a holiday and a weekend is often the quietest.
    prev_day = np.array([(d - pd.Timedelta(days=1)).date() for d in index])
    next_day = np.array([(d + pd.Timedelta(days=1)).date() for d in index])
    is_eve = np.array([d in holis for d in next_day], dtype=float)
    after = np.array([d in holis for d in prev_day], dtype=float)
    bridge = ((is_hol == 0) & (index.dayofweek < 5) &
              ((is_eve == 1) | (after == 1))).astype(float)
    feats += [is_hol, is_eve, after, bridge]

  if S.USE_SCHEDULED_PAYMENT_DAYS and S.PAYMENT_DAYS_OF_MONTH:
    nominal, earlier, later = scheduled_payment_dates(index, holis)
    feats.append(np.array([d in nominal for d in as_dates], dtype=float))
    feats.append(np.array([d in earlier for d in as_dates], dtype=float))
    feats.append(np.array([d in later for d in as_dates], dtype=float))
    # Signed distance to the closest scheduled day, so the model can see a
    # payment approaching rather than only recognising the day itself.
    targets = np.array(sorted(S.PAYMENT_DAYS_OF_MONTH), dtype=float)
    dist = np.min(np.abs(dom[:, None] - targets[None, :]), axis=1)
    feats.append(np.clip(dist, 0, 15) / 15.0)
  if not feats:
    # Every switch in settings.py PART 1 is off. Hand back a single flat row so
    # the model still runs; it simply learns nothing from the calendar.
    feats = [np.zeros(len(index))]
  return np.stack(feats).astype(np.float32)


@dataclasses.dataclass
class TimesFMParams:
  strategy: str = S.TIMESFM_STRATEGY   # auto | multichannel | signed-log | raw
  covariates: bool = True
  znorm: bool = S.TIMESFM_ZNORM
  make_positive: bool = False
  batch_size: int = 8
  device: str | None = S.TIMESFM_DEVICE
  offline: bool = S.TIMESFM_OFFLINE
  # Where the weights come from, in order of precedence: --checkpoint on the
  # command line, then TIMESFM_WEIGHTS in settings.py, then the TIMESFM_CHECKPOINT
  # environment variable, then the public HuggingFace repo.
  checkpoint: str = (S.TIMESFM_WEIGHTS
                     or os.environ.get("TIMESFM_CHECKPOINT")
                     or "google/timesfm-3.0-pytorch")
  context: int | None = S.TIMESFM_CONTEXT   # cap on history fed to the model


def timesfm_forecast(data, target, upto, horizon, p: TimesFMParams):
  """Forecast data.target(target)[:upto] forward `horizon` steps."""
  model, _ = _timesfm_model(p.checkpoint, p.device, p.batch_size, p.offline)
  index = data.index[:upto]
  values = data.target(target).to_numpy()[:upto]

  strategy = p.strategy
  if strategy == "auto":
    strategy = "multichannel" if target == "net_change" else "raw"

  start = 0 if p.context is None else max(0, upto - p.context)
  index, values = index[start:], values[start:]

  kwargs = {}
  if p.covariates:
    span = index.append(future_index(index[-1], horizon, data.grid_mode))
    kwargs["past_future_covariates"] = [calendar_covariates(span, data.statement_days)]

  def call(contexts):
    return list(model.predict_batch(
        contexts, horizon=horizon, return_quantiles=True,
        make_positive=p.make_positive or target == "purchases",
        use_znorm=p.znorm, padding_mode="edge", **kwargs))[0]

  if strategy == "multichannel":
    ch = np.stack([data.purchases.to_numpy()[start:upto],
                   data.payments.to_numpy()[start:upto],
                   data.balance.to_numpy()[start:upto]]).astype(np.float32)
    out = call([ch])
    # net = purchases - payments; the upper bound of a difference pairs the
    # upper of the first term with the LOWER of the second, hence the flip.
    point = out.forecast[0] - out.forecast[1]
    quant = np.sort(out.quantiles[0] - out.quantiles[1][:, ::-1], axis=-1)
    return point, quant[:, [0, 4, 8]]

  if strategy == "signed-log":
    warped = np.sign(values) * np.log1p(np.abs(values))
    out = call([warped.astype(np.float32)])
    inv = lambda z: np.sign(z) * np.expm1(np.abs(z))
    return inv(out.forecast), inv(out.quantiles)[:, [0, 4, 8]]

  out = call([values.astype(np.float32)])
  return out.forecast, out.quantiles[:, [0, 4, 8]]


@dataclasses.dataclass
class GBMParams:
  lags: tuple = S.GBM_LAGS
  windows: tuple = S.GBM_WINDOWS
  num_leaves: int = S.GBM_NUM_LEAVES
  learning_rate: float = S.GBM_LEARNING_RATE
  rounds: int = S.GBM_ROUNDS
  min_data_in_leaf: int = S.GBM_MIN_DATA_IN_LEAF
  feature_fraction: float = S.GBM_FEATURE_FRACTION


def _origin_features(values, t, p: GBMParams):
  """Everything knowable at time t. Uses no data after t."""
  past = values[: t + 1]
  feats = [past[-l] if len(past) >= l else np.nan for l in p.lags]
  for w in p.windows:
    win = past[-w:] if len(past) >= w else past
    feats += ([np.nan] * 4 if len(win) == 0
              else [win.mean(), win.std(), win.min(), win.max()])
  neg = np.where(past < 0)[0]
  feats.append(float(t - neg[-1]) if len(neg) else np.nan)
  feats.append(float(past[neg[-1]]) if len(neg) else np.nan)
  feats.append(float(past[-28:].sum()))
  return feats


def _target_calendar(date, h, statement_days, holis=frozenset(),
                     sched=(frozenset(), frozenset(), frozenset())):
  """The same facts the TimesFM covariates carry. Both models must get the same
  information or the comparison between them means nothing."""
  nominal, earlier, later = sched
  d = date.date()
  return [float(h), float(date.dayofweek), float(date.day), float(date.month),
          float(date.dayofweek >= 5), float(date.is_month_end),
          float(date.is_month_start),
          float(date.day in statement_days) if statement_days else 0.0,
          float(d in holis),
          float((date + pd.Timedelta(days=1)).date() in holis),
          float((date - pd.Timedelta(days=1)).date() in holis),
          float(d in nominal), float(d in earlier), float(d in later),
          min(abs(date.day - np.array(sorted(S.PAYMENT_DAYS_OF_MONTH), dtype=float)).min()
              if S.PAYMENT_DAYS_OF_MONTH else 15.0, 15.0) / 15.0]


def gbm_forecast(data, target, upto, horizon, p: GBMParams):
  """Direct multi-horizon LightGBM: one model over all steps, h as a feature.

  Lags and rolling statistics come from the forecast origin so they cannot see
  the future; calendar features come from the target date, which is knowable in
  advance -- the same information TimesFM gets through past_future covariates.
  """
  import lightgbm as lgb

  index = data.index[:upto]
  values = data.target(target).to_numpy()[:upto].astype(float)
  n = len(values)
  start = max(p.lags) + max(p.windows)
  if n - start < horizon + 30:
    raise RuntimeError(f"need more history: have {n}, want >{start + horizon + 30}")

  holis = holiday_dates(index)
  sched = scheduled_payment_dates(index, holis)

  rows, targets = [], []
  for t in range(start, n - 1):
    origin = _origin_features(values, t, p)
    for h in range(1, horizon + 1):
      if t + h >= n:
        break
      rows.append(origin + _target_calendar(index[t + h], h, data.statement_days,
                                            holis, sched))
      targets.append(values[t + h])
  X = np.asarray(rows, dtype=float)
  y = np.asarray(targets, dtype=float)

  future = future_index(index[-1], horizon, data.grid_mode)
  last = _origin_features(values, n - 1, p)
  Xf = np.asarray([last + _target_calendar(future[h - 1], h, data.statement_days,
                                           holis, sched)
                   for h in range(1, horizon + 1)], dtype=float)

  base = dict(objective="quantile", verbosity=-1, num_leaves=p.num_leaves,
              learning_rate=p.learning_rate, min_data_in_leaf=p.min_data_in_leaf,
              feature_fraction=p.feature_fraction, bagging_fraction=0.9, bagging_freq=1)
  preds = [lgb.train({**base, "alpha": a}, lgb.Dataset(X, label=y),
                     num_boost_round=p.rounds).predict(Xf)
           for a in (0.1, 0.5, 0.9)]
  quant = np.sort(np.stack(preds, axis=1), axis=1)
  return quant[:, 1], quant


def _raw_forecast(model, data, target, upto, horizon, params):
  if model == "timesfm":
    return timesfm_forecast(data, target, upto, horizon, params or TimesFMParams())
  if model == "gbm":
    return gbm_forecast(data, target, upto, horizon, params or GBMParams())
  raise ValueError(f"unknown model {model!r}")


def _conformal_width(model, data, target, upto, horizon, params, origins):
  """How much the p10-p90 interval has to widen to actually cover 80%.

  Split-conformal calibration (Romano et al., conformalized quantile
  regression). A quantile model's own interval is a claim; this measures how
  often that claim held on data the model had not seen, and returns the padding
  needed to make it true.

  For each calibration origin the model is refit on everything before it and
  scored on the block that follows. The conformity score for one step is how far
  outside the interval the actual value fell -- negative when it fell inside.
  The padding is the TARGET_COVERAGE percentile of those scores.
  """
  scores = []
  for k in range(1, origins + 1):
    cut = upto - k * horizon
    if cut < horizon * 3:
      break
    try:
      _, quant = _raw_forecast(model, data, target, cut, horizon, params)
    except Exception:
      continue
    actual = data.target(target).to_numpy()[cut:cut + horizon].astype(float)
    lo, hi = quant[:, 0], quant[:, 2]
    scores.extend(np.maximum(lo - actual, actual - hi))
  if not scores:
    return 0.0
  # Never tighten: this corrects overconfidence, and a model whose interval is
  # already honest should be left alone.
  return float(max(0.0, np.quantile(scores, S.TARGET_COVERAGE)))


_CONFORMAL_CACHE = {}


def forecast(model, data, target, upto, horizon, params=None, calibrate=None):
  """Single entry point. model is 'timesfm' or 'gbm'.

  With calibration on, the p10/p90 bounds are widened by an amount measured on
  held-out data, so the stated 80% interval covers about 80% in practice.
  """
  point, quant = _raw_forecast(model, data, target, upto, horizon, params)
  if calibrate is None:
    calibrate = S.CALIBRATE_INTERVALS
  if not calibrate:
    return point, quant

  # The level is part of the key, so changing TARGET_COVERAGE takes effect
  # instead of reusing padding computed for a different level.
  key = (model, target, upto, horizon, id(data), repr(params), S.TARGET_COVERAGE)
  if key not in _CONFORMAL_CACHE:
    _CONFORMAL_CACHE[key] = _conformal_width(
        model, data, target, upto, horizon, params, S.CALIBRATION_ORIGINS)
  pad = _CONFORMAL_CACHE[key]
  if pad > 0:
    quant = quant.copy()
    quant[:, 0] -= pad
    quant[:, 2] += pad
  return point, quant


# ================================================================ evaluation ==
def rolling_eval(model, data, target, horizon, origins=5, params=None):
  """Refit and score at `origins` successively earlier cut-offs.

  A single holdout is one sample and rankings flip on it; several origins is
  the difference between a number and a conclusion.
  """
  series = data.target(target)
  results = []
  for k in range(origins):
    upto = len(series) - (k + 1) * horizon
    if upto < horizon * 2:
      break
    actual = series.to_numpy()[upto:upto + horizon].astype(float)
    point, quant = forecast(model, data, target, upto, horizon, params)
    results.append({"origin": -(k + 1) * horizon, **score(actual, point, quant)})
  return pd.DataFrame(results)
