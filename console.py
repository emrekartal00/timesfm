"""
==============================================================================
RUN EVERYTHING FROM AN IPYTHON CONSOLE — no terminal needed.
==============================================================================

Start by moving into the folder that holds these files, then import this:

    import os
    os.chdir(r"C:\\path\\to\\timesfm")     # your folder; keep the r before the quote
    import console

Then, in order:

    console.check()                        # are the packages installed?
    console.forecast("mydata.xlsx")        # forecast, and show the report
    console.compare("mydata.xlsx")         # which model is better on YOUR data
    console.sweep("mydata.xlsx")           # find the best settings

Each function prints a report and hands back a table you can keep looking at:

    f = console.forecast("mydata.xlsx")
    f.head(20)
    f.to_excel("result.xlsx")

Every option from settings.py works here too, by name. For example:

    console.forecast("mydata.xlsx", target="growth", horizon=60)

If something goes wrong the error is printed in full rather than hidden.
==============================================================================
"""

import sys
import traceback

REQUIRED = [("pandas", "reading your spreadsheet"),
            ("openpyxl", "opening .xlsx files"),
            ("numpy", "arithmetic"),
            ("torch", "running TimesFM"),
            ("timesfm3", "the TimesFM model itself")]
OPTIONAL = [("lightgbm", "the gbm comparison model"),
            ("sklearn", "needed by lightgbm"),
            ("holidays", "holiday covariates -- worth 42% accuracy in testing"),
            ("statsmodels", "the seasonality statistics"),
            ("matplotlib", "the plot")]


def check():
  """Report which packages are present. Fixes nothing, just tells you."""
  import importlib
  print("REQUIRED — nothing runs without these")
  missing = []
  for name, why in REQUIRED + [("", "")] + OPTIONAL:
    if not name:
      print("\nOPTIONAL — these switch themselves off if absent")
      continue
    try:
      importlib.import_module(name)
      print(f"  OK       {name:<14} {why}")
    except Exception:
      print(f"  MISSING  {name:<14} {why}")
      missing.append(name)
  if missing:
    print("\nTo install the missing ones, run this line in the console:")
    print("    %pip install -r requirements-forecasting.txt")
    print("\n'%pip' is an IPython command; it installs into the Python you are")
    print("using right now, which is what you want.")
  else:
    print("\nEverything is installed.")
  return missing


def install():
  """Install the requirements from inside the console, printing the real error.

  A plain subprocess call with check=True raises CalledProcessError and throws
  the explanation away, which is why an install failure often looks like a
  traceback about subprocess.py with no clue in it.
  """
  import subprocess
  cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements-forecasting.txt"]
  print("running:", " ".join(cmd), "\n")
  r = subprocess.run(cmd, capture_output=True, text=True)
  print(r.stdout[-4000:])
  if r.returncode != 0:
    print("\n--- FAILED, exit code", r.returncode, "---")
    print(r.stderr[-4000:])
    print("\nRead the last few lines above. Common causes:")
    print("  * no internet, or a company proxy blocking pypi.org")
    print("  * no torch build for this Python version (needs 3.10 to 3.13)")
    print(f"    -- yours is {sys.version.split()[0]}")
  else:
    print("\nInstalled. Now run console.check() to confirm.")
  return r.returncode


def _load(excel, **kw):
  import cc_lib as L
  keys = ("sheet", "date_col", "balance_col", "currency_col", "currency",
          "growth_col", "use_growth", "grid")
  data = L.load(excel, **{k: v for k, v in kw.items() if k in keys})
  for note in data.notes:
    print(f"  {note}")
  print(f"  grid={data.grid_mode}  steps={len(data.index):,}  "
        f"{data.index[0]:%Y-%m-%d} to {data.index[-1]:%Y-%m-%d}")
  return data, L


def forecast(excel, target=None, horizon=None, model="timesfm", **kw):
  """Forecast, print a summary, and return the result as a table.

      f = console.forecast("mydata.xlsx", target="growth", horizon=30)

  model can be "timesfm" or "gbm".
  """
  try:
    import pandas as pd
    import settings as S
    data, L = _load(excel, **kw)
    target = target or S.DEFAULT_TARGET
    horizon = horizon or S.DEFAULT_HORIZON
    series = data.target(target)

    point, quant = L.forecast(model, data, target, len(series), horizon)
    future = L.future_index(data.index[-1], horizon, data.grid_mode)
    out = pd.DataFrame({"date": future, "forecast": point,
                        "p10": quant[:, 0], "p90": quant[:, 2]})

    unit = "business days" if data.grid_mode == "business" else "days"
    print(f"\n{model} forecast of {target}, {horizon} {unit} "
          f"({future[0]:%Y-%m-%d} to {future[-1]:%Y-%m-%d})")
    print(f"  total     {point.sum():>16,.0f}")
    print(f"  range     {quant[:, 0].sum():>16,.0f} to {quant[:, 2].sum():,.0f}")
    print(f"\nfirst rows:\n{out.head(10).to_string(index=False)}")
    print(f"\nReturned a table of {len(out)} rows. Save it with:")
    print('    f.to_excel("result.xlsx", index=False)')
    return out
  except Exception:
    traceback.print_exc()
    print("\nThe real error is the LAST line above, not the first.")
    return None


def compare(excel, target=None, horizon=None, origins=None, **kw):
  """Score TimesFM against the gradient-boosted model on your own data.

      console.compare("mydata.xlsx", target="growth", origins=6)

  Read MAE (lower is better) together with coverage, which should be near 80%.
  A model with the lowest MAE and 50% coverage is not better -- it is a sharper
  guess that is dishonest about its own uncertainty.
  """
  try:
    import pandas as pd
    import settings as S
    data, L = _load(excel, **kw)
    target = target or S.DEFAULT_TARGET
    horizon = horizon or S.DEFAULT_HORIZON
    origins = origins or S.DEFAULT_ROLLING_ORIGINS

    rows = []
    for name in ("timesfm", "gbm"):
      try:
        r = L.rolling_eval(name, data, target, horizon, origins)
        m = r.mean(numeric_only=True)
        rows.append({"model": name, "MAE": m.mae, "pinball": m.pinball,
                     "coverage": m.coverage, "MAE_payment_days": m.mae_spike,
                     "MAE_other_days": m.mae_other})
      except Exception as exc:
        print(f"  {name} failed: {type(exc).__name__}: {exc}")
    out = pd.DataFrame(rows)
    print(f"\nscored over {origins} rolling origins, horizon {horizon}\n")
    print(out.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    if len(out) == 2:
      best = out.loc[out.MAE.idxmin(), "model"]
      print(f"\nlower MAE: {best}")
      print("Check coverage too -- nearest to 0.80 is the more honest interval.")
    return out
  except Exception:
    traceback.print_exc()
    return None


def sweep(excel, target=None, horizon=None, origins=3, only="both", **kw):
  """Try many settings and rank them, so you can stop guessing.

      s = console.sweep("mydata.xlsx", target="growth", only="gbm")

  Copy the winning values into settings.py.
  """
  try:
    import itertools
    import pandas as pd
    import cc_lib as L
    import settings as S
    data, _ = _load(excel, **kw)
    target = target or S.DEFAULT_TARGET
    horizon = horizon or S.DEFAULT_HORIZON

    jobs = []
    if only in ("timesfm", "both"):
      for st in ("auto", "raw"):
        for cov in (True, False):
          jobs.append(("timesfm", f"timesfm strategy={st} covariates={cov}",
                       L.TimesFMParams(strategy=st, covariates=cov)))
    if only in ("gbm", "both"):
      for leaves in (7, 15, 31, 63):
        jobs.append(("gbm", f"gbm num_leaves={leaves}",
                     L.GBMParams(num_leaves=leaves)))

    rows = []
    print(f"\ntrying {len(jobs)} settings, {origins} origins each\n")
    for model, name, params in jobs:
      try:
        r = L.rolling_eval(model, data, target, horizon, origins, params)
        m = r.mean(numeric_only=True)
        rows.append({"setting": name, "MAE": m.mae, "pinball": m.pinball,
                     "coverage": m.coverage})
        print(f"  {name:<44} MAE {m.mae:>10,.0f}  cov {m.coverage:>4.0%}")
      except Exception as exc:
        print(f"  {name:<44} failed: {type(exc).__name__}")
    out = pd.DataFrame(rows).sort_values("pinball")
    print("\nranked best first:\n")
    print(out.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("\nPut the winner's values into settings.py.")
    return out
  except Exception:
    traceback.print_exc()
    return None
