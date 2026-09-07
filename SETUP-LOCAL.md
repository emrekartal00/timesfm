# Local TimesFM 3.0 setup (verified)

Notes from getting TimesFM 3.0 running locally for forecasting. Everything below
was actually run and verified, not copied from docs.

## Install

```bash
git clone https://github.com/emrekartal00/timesfm.git
cd timesfm
python -m venv .venv && source .venv/bin/activate   # or use an existing env
pip install -e ".[torch]"
```

Requires Python >= 3.10. On a machine that already has PyTorch, the `[torch]`
extra is a no-op — only `safetensors` gets pulled in.

## Weights

Checkpoint `google/timesfm-3.0-pytorch` — **1.3 GB**, ungated, downloads on first
model load into `~/.cache/huggingface`. The first load therefore takes as long as
your connection needs; subsequent loads are ~3s.

> **License:** the 3.0 *weights* are `timesfm-non-commercial-license-v1.0` —
> non-commercial, non-production only. The source code is Apache-2.0, and the
> 2.5 weights remain Apache-2.0 if commercial use is ever needed.

## Running it

```python
from timesfm3 import TimesFM3Evaluator, ModelConfig

f = TimesFM3Evaluator(ModelConfig(
    checkpoint_path="google/timesfm-3.0-pytorch",
    per_core_batch_size=8,
    device="cuda",   # "mps" on Apple Silicon, "cpu" otherwise
))

out = list(f.predict_batch([series], horizon=24, return_quantiles=True,
                           use_symmetric_averaging=False))
out[0].forecast     # (horizon,)
out[0].quantiles    # (horizon, 9)  -> deciles 0.1 .. 0.9
```

See `forecast_local.py` for a runnable starter with device auto-detection.

Contexts may be ragged 1-D arrays (univariate) or `(n_variates, context_len)`
2-D arrays (multivariate). TimesFM 3.0 also takes `past_only_covariates` and
`past_future_covariates` natively. Context length up to 16k.

## Privacy: is inference local?

**Yes — inference makes zero network calls.** Verified by sealing the process's
network (`socket.connect`, `socket.create_connection`, `socket.getaddrinfo` all
raising) and confirming `predict_batch` still ran.

The one caveat is at **load** time: `from_pretrained` contacts huggingface.co to
check whether the cached checkpoint is stale, even when the weights are already
local. That call happens in the constructor, before any data enters the process,
so it cannot carry your series — it sends the repo id, a user-agent and your IP.

To guarantee no network at all once the weights are cached:

```bash
export HF_HUB_OFFLINE=1
```

It must be set **before** `timesfm3` is imported. With it set, load + inference
complete with every outbound connection blocked.

Code audit backing this up: nothing under `src/timesfm3/` imports `requests` or
`urllib`, and there is no telemetry, analytics or upload path. The only Hub
touchpoint is `PyTorchModelHubMixin.from_pretrained`.

## Verified environment

| | |
|---|---|
| Machine | Apple Silicon (arm64), macOS |
| Python | 3.12 |
| torch | 2.10, MPS available |
| timesfm | 3.0.1 (editable install) |
| Cold load | ~20 min (1.3 GB download) |
| Warm load | 2.9 s |
| Forecast | 0.3 s — 2 series, horizon 24, with quantiles |
