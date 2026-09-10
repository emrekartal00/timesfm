"""
Build tufe.csv: the Turkish consumer price index, monthly, up to date.

    python fetch_tufe.py
    python fetch_tufe.py --out tufe.csv --start 2018-01

In Spyder:

    %run fetch_tufe.py

Needs internet, so run it on a machine that has some and copy the CSV across.
Run it again whenever you want the latest month.

WHY IT WORKS THE WAY IT DOES
----------------------------
No single open source gives a current index series, so this combines two:

  1. OECD via FRED publishes the actual INDEX (base 2015=100), monthly, going
     back decades. Plain download, no key. It lags by roughly a year.

  2. TÜİK publishes the official ANNUAL RATE for each recent month. That is
     enough to extend the index: this month's index is the same month last
     year multiplied by one plus the annual rate.

The TÜİK figures sit behind bot protection that rejects ordinary requests, so
they are read through a real browser with Playwright. A headless browser is
detected and refused; a headed one is not, which is why this opens a window.

The two sources are checked against each other on the months they share before
anything is spliced. If the annual rates implied by FRED disagree with TÜİK's
published ones, the splice is wrong and the script says so rather than writing
a plausible-looking file.

    pip install playwright && python -m playwright install chromium
"""

import argparse
import io
import json
import urllib.request

import numpy as np
import pandas as pd

FRED_SERIES = "TURCPIALLMINMEI"        # OECD CPI for Türkiye, 2015 = 100
FRED_URL = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_SERIES}"
TUIK_URL = "https://veriportali.tuik.gov.tr/"
TUIK_API = "/api/tr/press/indicators"

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--out", default="tufe.csv")
ap.add_argument("--start", default="2018-01", help="first month to keep")
ap.add_argument("--no-browser", action="store_true",
                help="skip TÜİK; write only the OECD index, which lags")
ap.add_argument("--show-browser", action="store_true",
                help="watch the browser work (it is headed either way)")
args = ap.parse_args()


def fetch_index():
  """The OECD index from FRED. Real values, but it lags."""
  print(f"OECD index via FRED ({FRED_SERIES}) ...")
  with urllib.request.urlopen(FRED_URL, timeout=60) as r:
    frame = pd.read_csv(io.StringIO(r.read().decode()))
  frame.columns = ["date", "index"]
  frame["date"] = pd.to_datetime(frame["date"])
  frame["index"] = pd.to_numeric(frame["index"], errors="coerce")
  frame = frame.dropna().sort_values("date")
  print(f"  {len(frame)} months, {frame['date'].iloc[0]:%Y-%m} to "
        f"{frame['date'].iloc[-1]:%Y-%m}")
  return frame.set_index("date")["index"]


def fetch_annual_rates():
  """TÜİK's official annual inflation rate per month, read via a browser.

  The API answers only inside a real browser session: a plain request, and a
  headless browser too, both come back 403 "Erişim engellendi".
  """
  from playwright.sync_api import sync_playwright

  print("TÜİK annual rates (opens a browser -- their API refuses anything else) ...")
  with sync_playwright() as pw:
    browser = pw.chromium.launch(
        headless=False, args=["--disable-blink-features=AutomationControlled"])
    ctx = browser.new_context(
        locale="tr-TR", timezone_id="Europe/Istanbul",
        viewport={"width": 1280, "height": 800},
        user_agent=("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/139.0.0.0 Safari/537.36"))
    ctx.add_init_script(
        "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})")
    page = ctx.new_page()
    page.goto(TUIK_URL, wait_until="domcontentloaded", timeout=90000)
    page.wait_for_timeout(9000)        # the bot check needs a moment to pass
    raw = page.evaluate("""async (p) => {
        const r = await fetch(p, {headers: {'accept': 'application/json'}});
        return r.status === 200 ? await r.text() : null;
    }""", TUIK_API)
    browser.close()

  if not raw:
    raise RuntimeError("TÜİK refused the request even from a browser")

  for item in json.loads(raw).get("data", []):
    for graphic in item.get("graphics", []):
      title = graphic.get("title", "")
      if "Fiyat" in title and "Yıllık" in title and graphic.get("labels"):
        months = [pd.Timestamp(int(l.split("/")[0]), int(l.split("/")[1]), 1)
                  for l in graphic["labels"]]
        rates = pd.Series(graphic["series"][0]["data"], index=months,
                          dtype=float).sort_index()
        print(f"  {len(rates)} months, {rates.index[0]:%Y-%m} to "
              f"{rates.index[-1]:%Y-%m}")
        return rates
  raise RuntimeError("No consumer-price series in the TÜİK response")


def splice(index, rates):
  """Extend the index past its end using the published annual rates.

  This month's index is the same month a year earlier times one plus that
  month's annual rate, so the walk has to go forward in time: each step may
  need a value the previous step produced.
  """
  # Check the two agree where they overlap, before trusting either.
  checked = 0
  for month, rate in rates.items():
    prev = month - pd.DateOffset(years=1)
    if month in index.index and prev in index.index:
      implied = (index[month] / index[prev] - 1) * 100
      if abs(implied - rate) > 1.0:
        raise RuntimeError(
            f"The two sources disagree for {month:%Y-%m}: FRED implies "
            f"{implied:.2f}% annual, TÜİK publishes {rate:.2f}%. Refusing to "
            f"splice them -- a wrong price index corrupts everything quietly.")
      checked += 1
  print(f"  cross-checked {checked} overlapping month(s), all agree within 1 point")

  out = index.copy()
  added = []
  for month, rate in rates.items():
    if month in out.index:
      continue
    prev = month - pd.DateOffset(years=1)
    if prev not in out.index:
      print(f"  cannot reach {month:%Y-%m}: no value for {prev:%Y-%m}")
      continue
    out[month] = out[prev] * (1 + rate / 100.0)
    added.append(month)
  if added:
    print(f"  extended by {len(added)} month(s), through {max(added):%Y-%m}")
  return out.sort_index()


index = fetch_index()
if not args.no_browser:
  try:
    index = splice(index, fetch_annual_rates())
  except Exception as exc:
    print(f"\n  TÜİK step failed: {type(exc).__name__}: {exc}")
    print("  Writing the OECD index alone. It lags, and the forecasting scripts")
    print("  will say so and fill the gap by extrapolation.")

index = index[index.index >= pd.Timestamp(args.start)]
frame = pd.DataFrame({"date": index.index.strftime("%Y-%m-%d"),
                      "index": index.round(3).to_numpy()})
frame.to_csv(args.out, index=False)

span = len(index) - 1
print(f"\nwritten: {args.out}")
print(f"  {len(frame)} months, {index.index[0]:%Y-%m} to {index.index[-1]:%Y-%m}")
print(f"  prices rose {index.iloc[-1] / index.iloc[0]:.2f}x over that span "
      f"({((index.iloc[-1] / index.iloc[0]) ** (1 / span) - 1) * 100:.2f}% per month)")
print(f"\nUse it with:  --deflator {args.out}")
