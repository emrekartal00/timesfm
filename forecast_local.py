"""Runnable TimesFM 3.0 starter with device auto-detection.

  python forecast_local.py                 # demo series, downloads weights if needed
  python forecast_local.py --offline       # force no network (weights must be cached)
  python forecast_local.py --csv data.csv --column sales --horizon 24

See SETUP-LOCAL.md for install and privacy notes.
"""

import argparse
import os

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--csv", help="CSV file to forecast from")
parser.add_argument("--column", help="column to forecast (default: last numeric column)")
parser.add_argument("--horizon", type=int, default=24)
parser.add_argument("--device", help="cuda / mps / cpu (default: auto-detect)")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument(
    "--offline",
    action="store_true",
    help="block all Hub access; requires weights already in ~/.cache/huggingface",
)
args = parser.parse_args()

# Must be set before timesfm3 (and huggingface_hub) are imported.
if args.offline:
  os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
from timesfm3 import ModelConfig, TimesFM3Evaluator


def pick_device() -> str:
  if args.device:
    return args.device
  if torch.cuda.is_available():
    return "cuda"
  if torch.backends.mps.is_available():
    return "mps"
  return "cpu"


def load_series() -> np.ndarray:
  """The demo series if no CSV was given, otherwise one column of the CSV."""
  if not args.csv:
    t = np.arange(400, dtype=np.float32)
    return (np.sin(2 * np.pi * t / 24) + 0.01 * t).astype(np.float32)

  import pandas as pd

  df = pd.read_csv(args.csv)
  if args.column:
    col = args.column
  else:
    numeric = df.select_dtypes("number").columns
    if numeric.empty:
      raise SystemExit(f"no numeric column found in {args.csv}")
    col = numeric[-1]
    print(f"no --column given, using {col!r}")
  return df[col].to_numpy(dtype=np.float32)


device = pick_device()
print(f"device: {device}" + (" (offline)" if args.offline else ""))

forecaster = TimesFM3Evaluator(
    ModelConfig(
        checkpoint_path="google/timesfm-3.0-pytorch",
        per_core_batch_size=args.batch_size,
        device=device,
    )
)

series = load_series()
print(f"context length: {len(series)}, horizon: {args.horizon}")

output = list(
    forecaster.predict_batch(
        [series],
        horizon=args.horizon,
        return_quantiles=True,
        use_symmetric_averaging=False,
    )
)[0]

# quantiles are the deciles 0.1 .. 0.9; columns 0 and 8 give an 80% interval.
print(f"\n{'step':>5}  {'forecast':>12}  {'p10':>12}  {'p90':>12}")
for i, (point, qs) in enumerate(zip(output.forecast, output.quantiles), start=1):
  print(f"{i:>5}  {point:>12.4f}  {qs[0]:>12.4f}  {qs[8]:>12.4f}")
