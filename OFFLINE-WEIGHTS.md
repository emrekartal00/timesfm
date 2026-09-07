# TimesFM 3.0 weights — offline transfer

The 1.3 GB checkpoint split into 3 parts, for machines where HuggingFace
downloads are blocked. Copy this whole folder to the target machine.

## Rebuild

Python only, no shell required — works on Windows, macOS and Linux:

    python rejoin_weights.py

Run it from inside this folder, or point it anywhere:

    python rejoin_weights.py --dir C:\path\to\transfer

It verifies every part against SHA256SUMS, joins them, checks the rebuilt file
against original.sha256, then deletes the parts (pass --keep to retain them).
Needs only the standard library — no pip install, no numpy, no torch.

To create the parts in the first place:

    python rejoin_weights.py --split model.safetensors --parts 3

`rejoin.sh` (macOS/Linux) and `rejoin.bat` (Windows) do the same thing if you
prefer a shell.

## Use it

Point `checkpoint_path` at this directory instead of the HF repo id. No network
is touched — verified with the socket layer blocked.

    from timesfm3 import TimesFM3Forecaster, ModelConfig

    model = TimesFM3Forecaster(ModelConfig(
        checkpoint_path="/path/to/this/folder",   # NOT "google/timesfm-3.0-pytorch"
        device="cuda",                            # or "mps" / "cpu"
    ))

The directory must contain `config.json` and `model.safetensors` — both are
here. The `.part-*` files can be deleted after a successful rejoin.

For the bundled scripts, set the env var instead of editing them:

    export TIMESFM_CHECKPOINT=/path/to/this/folder     # mac/linux
    set TIMESFM_CHECKPOINT=C:\path\to\this\folder      # windows cmd

    python timesfm_guide.py
    python forecast_local.py --horizon 24

## Contents

    model.safetensors.part-aa   421 MB
    model.safetensors.part-ab   421 MB
    model.safetensors.part-ac   419 MB
    config.json                 1.2 KB
    SHA256SUMS                  per-part checksums
    original.sha256             checksum of the rebuilt whole file

Rebuilt file: 1,322,898,824 bytes,
sha256 a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da

## Licence

These weights are `timesfm-non-commercial-license-v1.0` — non-commercial,
non-production use only. Copying them to another machine is fine; the licence
travels with them.
