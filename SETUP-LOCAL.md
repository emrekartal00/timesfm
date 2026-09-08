# TimesFM 3.0 — local setup

Working notes for running TimesFM 3.0 outside of Colab. Everything here was run
and verified, not copied from the upstream README.

**Where to go next**
- `timesfm_guide.py` — annotated, runnable walkthrough of the whole API.
- `forecast_local.py` — small starter that forecasts a CSV column.
- `OFFLINE-WEIGHTS.md` — machines where HuggingFace is blocked.

## Install

```bash
git clone https://github.com/emrekartal00/timesfm.git
cd timesfm
python -m pip install -e ".[torch]"
```

Python >= 3.10. Use `python -m pip`, not bare `pip`: it guarantees the package
lands in the interpreter you will actually run. On a machine that already has
PyTorch, `[torch]` is a no-op and only `safetensors` is added.

Check it worked:

```bash
python -c "import timesfm3; print('OK')"
```

### "No module named timesfm3"

Two causes, in order of likelihood.

**1. You cloned but did not install.** The package lives under `src/`, so it is
not importable just because you are standing in the repo. Run the install above.

**2. Interpreter mismatch.** Most machines have several Pythons and the install
went to a different one than you are running. On the Mac this was developed on,
four of six Pythons on `PATH` could not see `timesfm3`. Check both sides match:

```bash
python -c "import sys; print(sys.executable)"    # the one you run
python -m pip -V                                 # the one you installed into
```

If an IDE is involved, activate its interpreter or venv *first*, then install.

Windows `cmd.exe` needs the extra unquoted: `python -m pip install -e .[torch]`.
PowerShell takes it quoted as written.

## Weights

Checkpoint `google/timesfm-3.0-pytorch` — **1.3 GB**, ungated, downloaded on
first model load into `~/.cache/huggingface`. The first load takes as long as
your connection needs; later loads take about 3 seconds.

If the machine cannot reach HuggingFace, see `OFFLINE-WEIGHTS.md`.

> **Licence:** the 3.0 *weights* are `timesfm-non-commercial-license-v1.0` —
> non-commercial, non-production only. The source code is Apache-2.0, and the
> 2.5 weights remain Apache-2.0 if commercial use is ever needed.

## Running it

```python
from timesfm3 import TimesFM3Forecaster, ModelConfig

model = TimesFM3Forecaster(ModelConfig(
    checkpoint_path="google/timesfm-3.0-pytorch",
    device="cuda",             # "mps" on Apple Silicon, "cpu" otherwise
    per_core_batch_size=8,
))

out = list(model.predict_batch(
    [series],                  # 1-D array, oldest -> newest, float32
    horizon=24,
    return_quantiles=True,
))[0]

out.forecast     # (24,)     point forecast = the p50
out.quantiles    # (24, 9)   deciles p10 .. p90
```

Contexts may be ragged 1-D arrays (independent univariate series) or
`(n_variates, context_len)` 2-D arrays (one multivariate series forecast
jointly). 3.0 also takes `past_only_covariates` and `past_future_covariates`
natively. Context up to 15360 steps.

## Four things that are easy to get wrong

These are not in the upstream README and each cost time here.

**`device=None` never selects MPS.** The library auto-detects only `cuda`, then
falls back to `cpu`. On Apple Silicon you must pass `device="mps"` explicitly or
you silently get slow CPU inference. Both bundled scripts handle this for you.

**The README uses the benchmark class.** There are two:

| | |
|---|---|
| `TimesFM3Forecaster` | The plain API. Conservative defaults. Use this. |
| `TimesFM3Evaluator` | Benchmark subclass. Silently enables `use_symmetric_averaging` (**2x the compute**) and `make_positive`, adds `univariate=`, chunks >32 variates. |

**`return_quantiles` defaults to `False`** on `TimesFM3Forecaster`. This is the
usual reason `.quantiles` comes back `None`.

**`predict_batch` returns a generator.** Nothing runs until you consume it.
Wrap it in `list()`.

## Privacy: is inference local?

**Yes — inference makes zero network calls.** Verified by blocking the socket
layer (`socket.connect`, `socket.create_connection`, `socket.getaddrinfo` all
raising) and confirming `predict_batch` still ran to completion.

The caveat is at **load** time: `from_pretrained` contacts huggingface.co to
check whether the cached checkpoint is stale, even when the weights are already
on disk. That happens in the constructor, before any of your data exists in the
process, so it cannot carry your series — it sends the repo id, a user-agent and
your IP.

To suppress even that:

```python
ModelConfig(..., local_files_only=True)     # preferred: explicit and scoped
```

or `export HF_HUB_OFFLINE=1` before `timesfm3` is imported. Either way the
weights must already be cached.

Nothing under `src/timesfm3/` imports `requests` or `urllib`, and there is no
telemetry, analytics or upload path. The only Hub touchpoint is
`PyTorchModelHubMixin.from_pretrained`.

## Verified environment

| | |
|---|---|
| Machine | Apple Silicon (arm64), macOS |
| Python | 3.12 |
| torch | 2.10, MPS available |
| timesfm | 3.0.1, editable install |
| Cold load | ~20 min (1.3 GB download) |
| Warm load | 2.9 s |
| Forecast | 0.3 s — 2 series, horizon 24, with quantiles |
| Throughput | 26 ms per series — 24 series, horizon 24, batch 8 |
