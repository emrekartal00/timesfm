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
    return {"purchases": self.purchases,
            "net_change": self.net_change,
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


def load(excel, sheet=0, date_col=None, balance_col=None, currency_col=None,
         currency=None, growth_col=None, use_growth=False, grid="auto"):
  """Read a sheet and return a Series ready to forecast."""
  notes = []
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
      else:
        notes.append(f"found {gcol!r}, agrees on {agree:.0%} of steps; using the "
                     f"recomputed version (use_growth=True to override)")

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


def _timesfm_model(checkpoint, device, batch_size):
  """Loaded once per process -- the checkpoint is 1.3 GB."""
  import torch
  from timesfm3 import ModelConfig, TimesFM3Forecaster
  if device is None:
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
  key = (checkpoint, device, batch_size)
  if key not in _TIMESFM_CACHE:
    _TIMESFM_CACHE[key] = TimesFM3Forecaster(ModelConfig(
        checkpoint_path=checkpoint, device=device, per_core_batch_size=batch_size))
  return _TIMESFM_CACHE[key], device


def calendar_covariates(index, statement_days):
  """Features knowable arbitrarily far ahead, so they may span the horizon.

  Day of week and day of month are sine/cosine pairs rather than raw integers:
  as integers, Sunday=7 and Monday=1 look maximally far apart when adjacent.
  """
  dow, dom, month = index.dayofweek.to_numpy(), index.day.to_numpy(), index.month.to_numpy()
  return np.stack([
      np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
      np.sin(2 * np.pi * dom / 31), np.cos(2 * np.pi * dom / 31),
      np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
      (dow >= 5).astype(float),
      index.is_month_end.astype(float),
      index.is_month_start.astype(float),
      np.isin(dom, list(statement_days)).astype(float) if statement_days else np.zeros(len(index)),
  ]).astype(np.float32)


@dataclasses.dataclass
class TimesFMParams:
  strategy: str = "auto"          # auto | multichannel | signed-log | raw
  covariates: bool = True
  znorm: bool = False
  make_positive: bool = False
  batch_size: int = 8
  device: str | None = None
  checkpoint: str = os.environ.get("TIMESFM_CHECKPOINT", "google/timesfm-3.0-pytorch")
  context: int | None = None      # cap on how much history to feed


def timesfm_forecast(data, target, upto, horizon, p: TimesFMParams):
  """Forecast data.target(target)[:upto] forward `horizon` steps."""
  model, _ = _timesfm_model(p.checkpoint, p.device, p.batch_size)
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
  lags: tuple = (1, 2, 3, 4, 5, 6, 7, 10, 14, 21, 28, 35)
  windows: tuple = (7, 14, 28)
  num_leaves: int = 31
  learning_rate: float = 0.05
  rounds: int = 300
  min_data_in_leaf: int = 20
  feature_fraction: float = 0.9


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


def _target_calendar(date, h, statement_days):
  return [float(h), float(date.dayofweek), float(date.day), float(date.month),
          float(date.dayofweek >= 5), float(date.is_month_end),
          float(date.is_month_start),
          float(date.day in statement_days) if statement_days else 0.0]


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

  rows, targets = [], []
  for t in range(start, n - 1):
    origin = _origin_features(values, t, p)
    for h in range(1, horizon + 1):
      if t + h >= n:
        break
      rows.append(origin + _target_calendar(index[t + h], h, data.statement_days))
      targets.append(values[t + h])
  X = np.asarray(rows, dtype=float)
  y = np.asarray(targets, dtype=float)

  future = future_index(index[-1], horizon, data.grid_mode)
  last = _origin_features(values, n - 1, p)
  Xf = np.asarray([last + _target_calendar(future[h - 1], h, data.statement_days)
                   for h in range(1, horizon + 1)], dtype=float)

  base = dict(objective="quantile", verbosity=-1, num_leaves=p.num_leaves,
              learning_rate=p.learning_rate, min_data_in_leaf=p.min_data_in_leaf,
              feature_fraction=p.feature_fraction, bagging_fraction=0.9, bagging_freq=1)
  preds = [lgb.train({**base, "alpha": a}, lgb.Dataset(X, label=y),
                     num_boost_round=p.rounds).predict(Xf)
           for a in (0.1, 0.5, 0.9)]
  quant = np.sort(np.stack(preds, axis=1), axis=1)
  return quant[:, 1], quant


def forecast(model, data, target, upto, horizon, params=None):
  """Single entry point. model is 'timesfm' or 'gbm'."""
  if model == "timesfm":
    return timesfm_forecast(data, target, upto, horizon, params or TimesFMParams())
  if model == "gbm":
    return gbm_forecast(data, target, upto, horizon, params or GBMParams())
  raise ValueError(f"unknown model {model!r}")


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
