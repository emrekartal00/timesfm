# Running TimesFM 3.0 where HuggingFace is blocked

The checkpoint is a single 1.3 GB file. On a machine that cannot reach the Hub,
carry it in yourself and point the loader at a local folder. No HF account, no
cache structure to reproduce — verified with the socket layer blocked.

## What has to travel

A folder containing exactly two things:

    model.safetensors     1.3 GB
    config.json           1.2 KB     <- easy to forget, nothing loads without it

Everything else in `transfer/` is tooling to get those two files across intact.

## Splitting, for media that cannot take a 1.3 GB file

    python rejoin_weights.py --split model.safetensors --parts 3

Writes `model.safetensors.part-aa/ab/ac`, plus `SHA256SUMS` (per part) and
`original.sha256` (the whole file). Copy the parts, the two checksum files and
`config.json`.

## Rejoining on the target machine

Python only — no shell, no pip install, no numpy or torch. Standard library,
any Python >= 3.8, Windows/macOS/Linux:

    python rejoin_weights.py

Run it from inside the folder, or point it anywhere:

    python rejoin_weights.py --dir C:\path\to\transfer

It verifies each part against `SHA256SUMS`, joins them, checks the result
against `original.sha256`, then deletes the parts (`--keep` to retain them).
Streams in 8 MB chunks, so memory stays flat.

`rejoin.sh` and `rejoin.bat` do the same if you prefer a shell.

## If you already joined the parts by hand

    python rejoin_weights.py --verify

This finds the joined file whatever it is called, checks it, and renames it to
`model.safetensors`. Two things it protects you from:

**File size does not prove the join worked.** Parts concatenated in the wrong
order produce a file of exactly the right length. Only the checksum tells you.
The correct digest is:

    a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da

**The extension is just part of the filename.** There is no separate file-type
attribute on Windows, macOS or Linux, and nothing needs converting — the bytes
already are safetensors. A file called `model_safetensors` becomes a proper
`.safetensors` file by renaming it:

    import os
    os.rename("model_safetensors", "model.safetensors")
    print(os.listdir("."))      # shows TRUE names; Explorer hides extensions

Use Python rather than Explorer for this. Explorer hides known extensions by
default, which is how files end up misnamed in the first place, and it will
happily create `model.safetensors.txt` without telling you.

**If the checksum does not match, do not re-transfer yet.** As long as the
`.part-*` files are still there, the copy was almost certainly fine and only
the join went wrong. Delete the joined file and run `python rejoin_weights.py`,
which joins in the correct order and re-checks.

## Pointing the code at the weights

Give it the **folder**, not the file. The folder must hold both
`model.safetensors` and `config.json`.

```python
from timesfm3 import TimesFM3Forecaster, ModelConfig

model = TimesFM3Forecaster(ModelConfig(
    checkpoint_path=r"C:\path\to\transfer",   # the folder
    device="cuda",                            # or "mps" / "cpu"
    local_files_only=True,                    # never contact HuggingFace
))
```

Note the `r"..."` prefix on Windows paths. Without it `\t`, `\n` and friends
are read as escape characters and the path breaks in confusing ways. Forward
slashes work too.

`local_files_only=True` matters here: without it the loader tries to reach
huggingface.co to check for a newer revision, which hangs or errors on a
machine where the Hub is blocked.

The bundled scripts take the same location three ways:

    python forecast_local.py --checkpoint C:\path\to\transfer

    set TIMESFM_CHECKPOINT=C:\path\to\transfer        # cmd, this session
    $env:TIMESFM_CHECKPOINT="C:\path\to\transfer"     # PowerShell
    export TIMESFM_CHECKPOINT=/path/to/transfer       # mac/Linux

Be careful with the environment variable from an IDE — a run configuration
usually does not inherit what you set in a separate terminal. If it seems to be
ignored, use `--checkpoint` or set `checkpoint_path` in code.

Sanity check before running anything:

```python
import os
print(sorted(os.listdir(r"C:\path\to\transfer")))
# must contain both 'config.json' and 'model.safetensors'
```

## Licence

These weights are `timesfm-non-commercial-license-v1.0` — non-commercial,
non-production use only. Copying them to another machine you control is fine;
the licence travels with them. Do not publish them to a public repository.
