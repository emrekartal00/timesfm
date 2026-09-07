"""
Rejoin (or split) the TimesFM 3.0 checkpoint. Pure Python, standard library only.

Runs on Windows, macOS and Linux with any Python >= 3.8. No pip install, no
shell, no numpy, no torch needed -- this script only moves bytes around.

REJOIN (the usual case):

    python rejoin_weights.py

    Run it from inside the folder holding the model.safetensors.part-* files,
    or point it anywhere:

    python rejoin_weights.py --dir C:\\path\\to\\transfer

SPLIT (to create the parts in the first place):

    python rejoin_weights.py --split model.safetensors --parts 3

The rejoin verifies every part against SHA256SUMS before joining, then checks
the rebuilt file against original.sha256. A bad USB copy is caught here rather
than surfacing later as a confusing model-loading error.
"""

import argparse
import hashlib
import os
import sys

CHUNK = 8 * 1024 * 1024  # 8 MB, keeps memory flat regardless of file size
# Progress uses carriage returns, which turn into spam when piped to a file or
# an IDE console that does not handle them. Only animate on a real terminal.
TTY = sys.stdout.isatty()
TARGET = "model.safetensors"
PART_GLOB = TARGET + ".part-"


def human(n):
  """Bytes as a readable size."""
  for unit in ("B", "KB", "MB", "GB"):
    if n < 1024 or unit == "GB":
      return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
    n /= 1024.0


def sha256_of(path, label=None):
  """Streaming SHA-256, with a progress line so a 1.3 GB file doesn't look hung."""
  total = os.path.getsize(path)
  done = 0
  h = hashlib.sha256()
  with open(path, "rb") as fh:
    while True:
      block = fh.read(CHUNK)
      if not block:
        break
      h.update(block)
      done += len(block)
      if label and total and TTY:
        pct = done * 100 // total
        print(f"\r  {label}: {pct}%", end="", flush=True)
  if label:
    print(f"\r  {label}: done ({human(total)})")
  return h.hexdigest()


def find_parts(directory):
  """The part files, in the order they must be concatenated."""
  names = [n for n in os.listdir(directory) if n.startswith(PART_GLOB)]
  if not names:
    raise SystemExit(
        f"No files named {PART_GLOB}* found in:\n  {directory}\n"
        "Run this script from the transfer folder, or pass --dir."
    )
  # Suffixes are aa, ab, ac ... so plain lexicographic order is correct.
  return [os.path.join(directory, n) for n in sorted(names)]


def read_sums(directory):
  """Parse SHA256SUMS into {filename: digest}. Missing file is not fatal."""
  path = os.path.join(directory, "SHA256SUMS")
  if not os.path.exists(path):
    return {}
  sums = {}
  with open(path, "r", encoding="utf-8") as fh:
    for line in fh:
      line = line.strip()
      if not line:
        continue
      digest, _, name = line.partition(" ")
      sums[name.strip().lstrip("*")] = digest.strip()
  return sums


def do_join(directory, output, keep):
  parts = find_parts(directory)
  total = sum(os.path.getsize(p) for p in parts)
  print(f"Found {len(parts)} parts, {human(total)} total:")
  for p in parts:
    print(f"  {os.path.basename(p):<32} {human(os.path.getsize(p)):>10}")

  # 1. Verify each part before spending time on the join.
  sums = read_sums(directory)
  if sums:
    print("\nVerifying parts against SHA256SUMS ...")
    bad = []
    for p in parts:
      name = os.path.basename(p)
      if name not in sums:
        print(f"  {name}: not listed, skipping")
        continue
      if sha256_of(p, name) != sums[name]:
        bad.append(name)
        print(f"  {name}: MISMATCH")
    if bad:
      raise SystemExit(
          "\nThese parts are corrupted: " + ", ".join(bad) +
          "\nRecopy them from the source machine and run this again."
      )
    print("  all parts OK")
  else:
    print("\nNo SHA256SUMS file; skipping per-part verification.")

  # 2. Check there is room before writing.
  out_path = output or os.path.join(directory, TARGET)
  try:
    free = getattr(os, "statvfs", None)
    if free:
      st = os.statvfs(directory)
      if st.f_bavail * st.f_frsize < total:
        raise SystemExit(
            f"Not enough free space: need {human(total)} in {directory}"
        )
  except (AttributeError, OSError):
    pass  # Windows without statvfs; the write below will fail loudly anyway.

  # 3. Join, streaming so memory use stays flat.
  print(f"\nWriting {out_path} ...")
  written = 0
  with open(out_path, "wb") as out:
    for p in parts:
      with open(p, "rb") as fh:
        while True:
          block = fh.read(CHUNK)
          if not block:
            break
          out.write(block)
          written += len(block)
          if TTY:
            print(f"\r  {written * 100 // total}%", end="", flush=True)
  print(f"\r  done ({human(written)})")

  # 4. Verify the result.
  expected_path = os.path.join(directory, "original.sha256")
  if os.path.exists(expected_path):
    with open(expected_path, "r", encoding="utf-8") as fh:
      expected = fh.read().split()[0].strip()
    print("\nVerifying rebuilt file ...")
    actual = sha256_of(out_path, TARGET)
    if actual != expected:
      raise SystemExit(
          f"\nCHECKSUM MISMATCH\n  expected {expected}\n  actual   {actual}\n"
          "The rebuilt file is not correct. Recopy the parts and retry."
      )
    print(f"  OK  {actual}")
  else:
    print("\nNo original.sha256 found; cannot verify the rebuilt file.")

  if not keep:
    for p in parts:
      os.remove(p)
    print(f"\nRemoved {len(parts)} part files (pass --keep to retain them).")

  config = os.path.join(directory, "config.json")
  print("\nDone.")
  if not os.path.exists(config):
    print("WARNING: config.json is missing from this folder. The model will not")
    print("load without it -- copy it across from the source machine.")
  else:
    print("Point checkpoint_path at this folder:")
    print(f'    TIMESFM_CHECKPOINT={directory}')


def do_verify(directory, path):
  """Check an already-joined file against original.sha256, renaming if needed."""
  target = os.path.join(directory, TARGET)

  if path:
    candidate = path
  elif os.path.exists(target):
    candidate = target
  else:
    # Someone joined the parts by hand and named the result something else.
    # Find a plausible file: right size, not one of the parts.
    expected_size = None
    sums_path = os.path.join(directory, "original.sha256")
    guesses = [
        n for n in os.listdir(directory)
        if not n.startswith(PART_GLOB)
        and n not in ("SHA256SUMS", "original.sha256", "config.json")
        and os.path.isfile(os.path.join(directory, n))
        and os.path.getsize(os.path.join(directory, n)) > 100 * 1024 * 1024
    ]
    if len(guesses) == 1:
      candidate = os.path.join(directory, guesses[0])
      print(f"Found a large file that is probably the join: {guesses[0]}")
    elif not guesses:
      raise SystemExit(f"No joined file found in {directory}. Pass one explicitly.")
    else:
      raise SystemExit("Several candidates: " + ", ".join(guesses) +
                       "\nPass the right one explicitly.")

  if not os.path.exists(candidate):
    raise SystemExit(f"No such file: {candidate}")

  size = os.path.getsize(candidate)
  print(f"Checking {os.path.basename(candidate)} ({human(size)}, {size:,} bytes)")

  expected_path = os.path.join(directory, "original.sha256")
  if not os.path.exists(expected_path):
    raise SystemExit("No original.sha256 in this folder; cannot verify.")
  with open(expected_path, "r", encoding="utf-8") as fh:
    expected = fh.read().split()[0].strip()

  print("Hashing (a minute or two for 1.3 GB) ...")
  actual = sha256_of(candidate, os.path.basename(candidate))

  if actual != expected:
    print(f"\n  expected {expected}")
    print(f"  actual   {actual}")
    raise SystemExit(
        "\nMISMATCH -- the joined file is not correct.\n"
        "If you still have the .part-* files, the transfer is probably fine and\n"
        "only the join went wrong (usually the wrong order). Delete the joined\n"
        "file and run:  python rejoin_weights.py\n"
        "That joins them in the correct order and re-checks."
    )

  print(f"  OK  {actual}")
  print("\nThe file is byte-for-byte identical to the original.")

  # Make sure it is named what the loader expects.
  if os.path.abspath(candidate) != os.path.abspath(target):
    os.rename(candidate, target)
    print(f"Renamed {os.path.basename(candidate)} -> {TARGET}")

  if not os.path.exists(os.path.join(directory, "config.json")):
    print("\nWARNING: config.json is missing -- the model will not load without it.")
  else:
    print("\nReady. Point checkpoint_path at this folder:")
    print(f"    TIMESFM_CHECKPOINT={directory}")


def do_split(source, n_parts):
  """Split a file into n roughly equal parts, writing SHA256SUMS alongside."""
  directory = os.path.dirname(os.path.abspath(source)) or "."
  total = os.path.getsize(source)
  size = -(-total // n_parts)  # ceiling division
  print(f"Splitting {source} ({human(total)}) into {n_parts} parts ...")

  written, index, made = 0, 0, []
  with open(source, "rb") as src:
    while written < total:
      suffix = chr(ord("a") + index // 26) + chr(ord("a") + index % 26)
      name = os.path.join(directory, f"{PART_GLOB}{suffix}")
      remaining = size
      with open(name, "wb") as out:
        while remaining > 0:
          block = src.read(min(CHUNK, remaining))
          if not block:
            break
          out.write(block)
          remaining -= len(block)
          written += len(block)
      made.append(name)
      print(f"  {os.path.basename(name)}  {human(os.path.getsize(name))}")
      index += 1

  with open(os.path.join(directory, "SHA256SUMS"), "w", encoding="utf-8") as fh:
    for p in made:
      fh.write(f"{sha256_of(p)}  {os.path.basename(p)}\n")
  with open(os.path.join(directory, "original.sha256"), "w", encoding="utf-8") as fh:
    fh.write(sha256_of(source, os.path.basename(source)) + "\n")
  print(f"\nWrote {len(made)} parts, SHA256SUMS and original.sha256.")
  print("Copy those, plus config.json, to the target machine.")


def main():
  ap = argparse.ArgumentParser(
      description="Rejoin or split the TimesFM 3.0 checkpoint.",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=__doc__,
  )
  ap.add_argument("--dir", default=None,
                  help="folder holding the parts (default: this script's folder)")
  ap.add_argument("--output", default=None, help="output path for the joined file")
  ap.add_argument("--keep", action="store_true",
                  help="keep the .part-* files after a successful join")
  ap.add_argument("--verify", nargs="?", const=True, metavar="FILE",
                  help="check an already-joined file and rename it correctly")
  ap.add_argument("--split", metavar="FILE", help="split FILE instead of joining")
  ap.add_argument("--parts", type=int, default=3, help="number of parts for --split")
  a = ap.parse_args()

  if a.split:
    do_split(a.split, a.parts)
    return

  directory = a.dir or os.path.dirname(os.path.abspath(__file__))
  if not os.path.isdir(directory):
    raise SystemExit(f"Not a folder: {directory}")

  if a.verify:
    do_verify(directory, None if a.verify is True else a.verify)
    return

  do_join(directory, a.output, a.keep)


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    sys.exit("\nInterrupted. Nothing was verified; rerun to finish.")
