# Forecasting your own data

Guide to the forecasting files in this fork. Everything here was measured
on real runs, not assumed.

| file | what it is |
|---|---|
| `credit_card_forecast.py` | **Start here.** One run: seasonality report, backtest, forecast, CSV, plot. |
| `sweep.py` | Runs many configurations of both models over the same backtest and ranks them. |
| `explore.py` | Describes the data: trend, the four seasonal cycles, holidays, outliers. No forecasting. |
| `settings.py` | **Every knob, in plain English with examples.** Edit this, not the code. |
| `cc_lib.py` | The shared library both scripts import. Only edit for new behaviour. |
| `timesfm_guide.py` | Unrelated to card data: an annotated tour of the raw TimesFM API. |

The two scripts share `cc_lib.py` deliberately. If they each had their own copy
of the data preparation, their numbers would quietly stop being comparable.

---

## 1. Install

In Spyder or any IPython console:

```python
%pip install -e .[torch]
%pip install -r requirements-forecasting.txt
```

In a terminal:

```bash
python -m pip install -e ".[torch]"                      # TimesFM itself
python -m pip install -r requirements-forecasting.txt    # everything else
```

`%pip` installs into the console you are actually running, which is what you
want. In a terminal use `python -m pip`, not plain `pip`: it guarantees the packages land in the
same Python you run the scripts with, which is the usual reason an install looks
fine and the script still says `No module named ...`.

That file gives **minimum** versions rather than exact ones, on purpose. Exact
pins make pip try to compile a package from source when it has no ready-built
one for your Python, which on Windows fails with:

```
preparing metadata (pyproject.toml) did not run successfully
failed to activate VS environment: no vswhere.exe
```

That message means "no prebuilt package, and no C++ compiler to build one". The
fix is never to install Visual Studio — it is to let pip choose a version that
ships ready-built. If you hit it anyway, your Python is likely too new for one
of the packages; 3.11 or 3.12 is the safest choice today.

## 2. Quick start

In Spyder or any IPython console, use `%run`. It passes the arguments through
and sets the import path for you:

```python
%run credit_card_forecast.py --excel yourfile.xlsx --target growth --rolling 6
%run sweep.py --excel yourfile.xlsx --origins 6
```

In a terminal the same commands drop the `%`:

```bash
python credit_card_forecast.py --excel yourfile.xlsx --target growth --rolling 6
python sweep.py --excel yourfile.xlsx --origins 6
```

Everything below is written with the terminal form. To use it in Spyder, put
`%run` in front and drop `python`.

Give the full path if the spreadsheet is somewhere else, keeping the `r` before
the quote on Windows:

```python
%run credit_card_forecast.py --excel r"C:\Users\you\Desktop\cards.xlsx" --target growth
```

## 3. Looking at the data first — `explore.py`

```bash
python explore.py --excel yourfile.xlsx --target growth --plot
```

Nothing here forecasts; it describes. Eight sections: the data itself, whether
the level is drifting, the strength of each seasonal cycle, then the weekly,
monthly and yearly profiles, holidays, and outliers. Run it before trusting any
forecast, and again whenever one looks wrong — most bad forecasts are bad data.

**Seasonal strength** is the share of the wobble a cycle explains once the trend
is removed: 0 means absent, 1 means the series is nothing else. All four cycles
are separated at once with MSTL, so a weekly rhythm is not confused for a
monthly one.

The section to check first is the monthly one. It lists the days of the month
where settlements actually land, alongside what `PAYMENT_DAYS_OF_MONTH` claims,
and says so when they disagree:

```
Days of the month where a settlement usually happens:
  day 24     73% of months
  day  4     70% of months
  day 14     68% of months
settings.py has PAYMENT_DAYS_OF_MONTH = [4, 14, 24]
```

`--deflator tufe.csv` reports everything in today's lira instead of the lira of
the day, which changes what the trend section is telling you. On a test series
the nominal figures showed a typical day growing to 8.5x its 2019 size; deflated
by the real index it came out at 0.9x — the same data describing growth or a
real-terms decline depending only on whether inflation was removed. `--rescale`
does a cruder version of this without needing an index file.

`--plot` writes `explore.png`; `--out parts.csv` saves the decomposed cycles.

## 4. What the script does to your data

### `--target growth` and `--use-growth` are not the same thing

They sound alike and catch people out:

* **`--target growth`** — *what* to forecast. The daily change rather than the
  balance level.
* **`--use-growth`** — *where that daily change comes from*. Your sheet's column
  instead of being recomputed from the balance.

**You almost certainly only need the first.** The script recomputes the daily
change from the balance, compares it with your growth column, and prints how
closely they match. A growth column is normally just balance minus yesterday's
balance, so they match exactly, and the flag then changes nothing — the script
says so outright:

```
found 'growth': identical to balance.diff() on every step, so
--use-growth would change nothing. Skip it.
```

It only matters when the two disagree, which means a stale formula or an
inserted row — and in that case the recomputed version is the one to trust.

**It only ever reads the date and balance columns.** Day of month, year,
yearmonth, day of week and growth are all derived from those, so they are
recomputed rather than trusted — a stale fill-down or an inserted row is a
common and silent source of error. Your `growth` column is checked against the
recomputed version and the agreement rate printed; `--use-growth` makes yours
authoritative anyway.

**It puts the rows on a regular grid.** TimesFM never sees dates; it assumes
every step is one period. Whether weekends belong is decided from your data:

| your sheet | what happens |
|---|---|
| no weekend rows | business-day grid; a Monday step carries the weekend, exactly as a row-to-row growth column does |
| weekend rows, flat | dropped as padding — weekly zeros dilute the series and smear the statement spike |
| weekend rows with real spending | all 7 days kept; dropping them would discard real money |

Override with `--grid business` or `--grid calendar`. Watch the line that starts
`278 weekend rows, 99% carry movement ->` to see which you got.

**It separates the flow from the stock.** Balance is a level; usage is the daily
change. On a card that change is purchases *minus* payments, and a statement
payment is one huge negative spike, so the two are modelled apart.

**It finds your statement days.** Days of the month where payments recur are
detected and passed to both models as a future covariate, since a date is
knowable in advance.

---

## 5. `credit_card_forecast.py` arguments

### Input

| argument | default | meaning |
|---|---|---|
| `--excel PATH` | *required* | the .xlsx file |
| `--sheet N` | `settings.SHEET` | sheet name, or index from 0. The second sheet is `1` |
| `--date-col NAME` | `settings.COLUMN_DATE` | override date column detection |
| `--balance-col NAME` | `settings.COLUMN_BALANCE` | override balance column detection |
| `--currency-col NAME` | `settings.COLUMN_CURRENCY` | override currency column detection |
| `--currency TRY` | most rows | which currency to model |
| `--growth-col NAME` | `settings.COLUMN_GROWTH` | name of your growth column |
| `--use-growth` | off | read the daily change from your sheet's column instead of recomputing it. **You probably do not need this** — see below |
| `--grid MODE` | `auto` | `auto`, `business`, `calendar` |

Column names are set in `settings.py` PART 0, so normally you pass none of
these. If a name is left blank there, detection falls back to matching English
and Turkish words (`date`/`tarih`, `balance`/`bakiye`, `currency`/`döviz`,
`growth`/`artış`) — but that only works when the name contains the whole word,
so a column called `DAT` has to be named explicitly. Either way the first
printed line shows what was used:

```
date='DAT' balance='BALANCE' currency='CUR'
```

**Currencies are never mixed.** Mixing them would just fit an exchange rate. If
your sheet has several, one is chosen and reported; use `--currency` to pick.

### What to forecast

| argument | default | meaning |
|---|---|---|
| `--target purchases` | `purchases` | usage only: the positive part of the change |
| `--target growth` | | purchases **and** payments, signed. Same thing your sheet's growth column holds |
| `--target net_change` | | an alias for `growth`; both names work |
| `--target balance` | | the outstanding level itself |
| `--horizon N` | `30` | steps ahead. On a business grid this is **trading days**, so 20 ≈ 4 weeks. The output prints the real dates. |

### Models

| argument | default | applies to | meaning |
|---|---|---|---|
| `--model` | `both` | | `timesfm`, `gbm`, or `both` side by side |
| `--strategy` | `auto` | TimesFM | `auto`, `multichannel`, `signed-log`, `raw` |
| `--no-covariates` | off | TimesFM | drop calendar covariates. **Costs a lot** — 80% worse in testing |
| `--znorm` | off | TimesFM | z-normalise before the model, undo after |
| `--context N` | all | TimesFM | cap how much history is fed |
| `--checkpoint PATH` | `settings.TIMESFM_WEIGHTS` | TimesFM | folder holding the weights, for an offline machine |
| `--device` | `settings.TIMESFM_DEVICE` | TimesFM | `cuda`, `mps`, `cpu` |
| `--offline` | `settings.TIMESFM_OFFLINE` | TimesFM | never contact HuggingFace |
| `--num-leaves N` | `31` | GBM | tree size. Lower = smoother |
| `--learning-rate F` | `0.05` | GBM | lower needs more rounds |
| `--rounds N` | `300` | GBM | boosting iterations |

`--strategy` only matters for a signed target. `auto` picks `multichannel` for
`growth`, which forecasts purchases, payments and balance as three joint
channels and recombines them. Balance is in there because payment size depends
on what has accumulated, which is what lets the model anticipate the spike.

### Evaluation and output

| argument | default | meaning |
|---|---|---|
| `--rolling N` | `0` | score over N rolling origins. **Use 6.** One holdout is one sample and rankings flip on it |
| `--no-backtest` | off | skip the single holdout |
| `--plot` | off | write `forecast.png` |
| `--out PATH` | `forecast.csv` | where the forecast goes |

---

## 6. Changing things without touching code — `settings.py`

`settings.py` is a plain list of named values, each with a sentence explaining
what it does and an example. Edit a value, save, rerun. Nothing else needs
touching, and if you break it the script errors out without changing anything.
To undo everything: `git checkout settings.py`.

Two rules: `True` and `False` are capitalised exactly like that, and words need
quotes (`"business"`) while numbers do not (`31`).

It has five parts.

**Part 0 — your spreadsheet.** Which sheet to read and the four column names,
set once so you never type them. Everything else in the sheet is ignored
deliberately: `day`, `year`, `yearmonth` and `dayweek` are all derived from the
date and are recomputed rather than trusted, and `diff` is ignored because it is
growth with weekends forced to zero — and those forced zeros are exactly what
hurts the forecast.

**Part 1 — which calendar facts the models get.** Nineteen covariates behind
nine on/off switches, including public holidays and your scheduled payment
dates. This is the biggest lever in the pipeline. Switching off just
two of them made TimesFM 84% worse in testing:

| | error | payment-day error |
|---|---|---|
| all covariates on | **1,180** | **3,603** |
| day-of-month and statement-day off | 2,175 | 12,849 |

Holidays and scheduled payment dates were added later and are worth as much
again. On a sheet with Turkish holidays and payments due on the 4th, 14th and
24th:

| | TimesFM | payment-day | GBM | payment-day |
|---|---|---|---|---|
| holidays + payment dates on | **844** | **1,204** | **858** | **1,269** |
| both off | 1,452 | 4,609 | 1,080 | 3,180 |

That is 42% off TimesFM's error and 74% off its payment-day error.

**Public holidays** need the `holidays` package (`pip install holidays`) and are
set with `HOLIDAY_COUNTRY = "TR"`. Turkey's religious holidays move about eleven
days earlier each year, so they are computed rather than listed — a hard-coded
list would go quietly wrong after one year. Four facts are supplied: the holiday
itself, its eve, the day after, and "bridge" days.

**Scheduled payment dates** are set with `PAYMENT_DAYS_OF_MONTH = (4, 14, 24)`.
You do not list the shifted dates. A payment due on the 14th that lands on a
Sunday or a holiday moves to a working day, and the script marks the nearest
working day both before and after each due date, so the model learns which way
your bank moves them. It also measures the distance to the nearest due date, so
a payment can be seen approaching rather than only recognised on arrival.
Verified: the 14th on a Saturday yields the 13th or the 16th; the 24th on a
Sunday yields the 22nd or the 25th.

**Part 2 — calibrating GBM.** The four settings that matter, each with a
sensible range and which direction to move it. The section explains overfitting
in plain terms: the model memorises your history, looks brilliant on old data,
forecasts badly. The tell is a low coverage number with `--rolling 6`.

**Part 3 — what GBM looks back at.** Which previous days it sees (`GBM_LAGS`)
and over what stretches it summarises (`GBM_WINDOWS`), with examples for weekly
rather than daily data.

**Part 4 — defaults for your data**, so you stop retyping the same flags.

**Part 5 — TimesFM settings**: where the weights are, which processor to use,
strategy, and how much history to feed. On a machine that cannot reach the
internet, `TIMESFM_WEIGHTS` is the one to set — point it at the folder holding
`model.safetensors` and `config.json`, not at the file:

```python
TIMESFM_WEIGHTS = r"C:\Users\you\Desktop\transfer"
```

Keep the `r` before the quote on Windows. See `OFFLINE-WEIGHTS.md` for getting
the weights onto that machine in the first place.

**A folder path is already offline.** It is read straight off the disk and never
touches the network — verified by blocking the socket layer and loading anyway.
`TIMESFM_OFFLINE = True` is for the other case: when `TIMESFM_WEIGHTS` is `None`
so the weights come from the HuggingFace cache. Loading them then pings
huggingface.co to ask whether a newer copy exists, even though they are already
downloaded, and on a machine where that site is blocked the request hangs rather
than failing fast. Setting it skips the check.

Either way your data goes nowhere: that check runs before the spreadsheet is
read and carries only the model name.

Anything you set here can still be overridden for one run from the command
line, and each comment names the flag that does it.

### Calibrating GBM without guessing

Do not tune by hand. Run the sweep and read the table:

```bash
python sweep.py --excel yourfile.xlsx --only gbm --origins 6
```

It tries the combinations and ranks them. Copy the winner into `settings.py`.
That is the whole method. In testing it found `num_leaves=15` beat the default
`31` (error 1,055 against 1,117) — a smaller, simpler model generalising better,
which is the usual answer.

---

## 7. `sweep.py` arguments

Everything in the Input section above works here too, plus:

| argument | default | meaning |
|---|---|---|
| `--preset quick` | `quick` | 6 configurations; enough to see the shape |
| `--preset full` | | the whole grid — dozens of configurations, much slower |
| `--only` | `both` | restrict to `timesfm` or `gbm` |
| `--origins N` | `5` | rolling origins per configuration |
| `--sort` | `pinball` | rank by `mae`, `pinball`, or `coverage` (closest to 80%) |
| `--out PATH` | `sweep_results.csv` | full results, one row per configuration |

Cost is *configurations × origins*, and every configuration refits at every
origin. `--preset full --origins 6` on GBM is dozens of fits — start with
`quick`.

To change what is swept, edit `TIMESFM_GRID` and `GBM_GRID` at the top of
`sweep.py`. They are plain dictionaries of lists.

---

## 8. Reading the numbers

| metric | in plain terms | why this one |
|---|---|---|
| **MAE** | "on a typical day we are off by this many lira" (shown with the unit from `CURRENCY_UNIT`) | RMSE squares errors, so 45,000-lira payment days would drown out everything else. MAPE and sMAPE explode near zero, and this series sits near zero at weekends and goes negative on payments — sMAPE rated the *worst* model best when tested. |
| **pinball** | "how good are all three numbers together, not just the middle one" | MAE grades only the middle number, so a model with a useless range can win on it. **Trust this one** on a spiky series. |
| **coverage** | "when it claims 80%, is it really 80%?" | The honesty check. It is what caught a model claiming 80% and delivering 52%. |
| **MAE spike** | error on payment days only | These are large and rare; averaged in, they hide everything else. |
| **MAE other** | error on ordinary days only | What you actually live with day to day. |

### Where the 80% came from, and how to change it

Not from a judgement about your business. TimesFM only ever reports the nine
deciles p10 to p90, so p10–p90 is the widest range it has, and 80% is what the
architecture hands us.

Calibration removes that limit, because it measures real errors rather than
asking the model. `TARGET_COVERAGE` in `settings.py` sets the level:

| asked for | actually got | average width |
|---|---|---|
| 50% | 69% | 2,059 |
| 80% | 81% | 2,651 |
| 90% | 88% | 3,030 |
| 95% | 92% | 3,747 |

Choosing is a question about consequences, not statistics: what does it cost to
be surprised? 0.90 is reasonable for planning cash; 0.95 when being caught short
is expensive. The range gets wider, and that width is the honest price of the
extra certainty — a wider range is not a worse forecast, it is the same forecast
told honestly.

Asking for 50% returns 69%, not 50%. Calibration never *narrows* an interval, so
below the model's natural width you simply get its natural width.

**Coverage is the one people ignore.** A model with the lowest MAE and 50%
coverage is not better — it is a sharper guess that is dishonest about its own
uncertainty, and it will under-warn you in a bad month.

`CALIBRATE_INTERVALS` in `settings.py` fixes this, and is on by default. The
gradient-boosted model fits its training data tightly, so its stated interval
comes out too narrow — on real card data it claimed 80% and delivered 52%.
Calibration refits the model on older slices of your history, measures how far
outside its interval reality actually landed, and widens the final interval by
exactly that much. It never narrows one, so a model that was already honest is
untouched.

Measured over four origins:

| model | calibration | MAE | pinball | coverage |
|---|---|---|---|---|
| gbm | off | 858 | 281 | 69% |
| gbm | **on** | 858 | **277** | **81%** |
| timesfm | off | 844 | 280 | 85% |
| timesfm | on | 844 | 280 | 85% (unchanged — already honest) |

The point forecast never moves; only the range does. The cost is a few extra
refits per forecast.

**sMAPE is not reported** for signed targets. It is meaningless on a series that
crosses zero; it rated the *worst* model best in testing.

---

## 9. What the measurements showed

On synthetic data shaped like a card statement, over rolling origins:

| finding | numbers |
|---|---|
| Covariates matter more than the model choice | TimesFM 1,180 with, 2,122 without |
| GBM is the sharper point forecast | GBM 1,055 MAE vs TimesFM 1,180 |
| TimesFM is the honest interval | 86% coverage vs GBM's 62–71% against a nominal 80% |
| Weekend padding hurts when weekends are dead | mean MAE −37%, payment-day MAE −72% dropping them |
| Multichannel beats modelling the signed series | MAE 2,940 → 2,428, payment-day 45,616 → 36,435 |
| Seasonal naive is a floor, not a rival | 3,339 |

**TimesFM is switched off.** On the real data it came out at roughly twice the
error of the gradient-boosted model — MAE near 1,000,000,000 against
615,543,454 — so `DEFAULT_MODEL = "gbm"` and nothing calls it. The code stays,
and `--model timesfm` still works if that finding is worth re-checking, but the
1.3 GB of model weights are not needed for normal use.

`settings.py` holds the configuration that won the sweep on real data:
`GBM_NUM_LEAVES = 63`, `GBM_LEARNING_RATE = 0.05`, `GBM_ROUNDS = 1200`,
`GBM_MIN_DATA_IN_LEAF = 40`, `RESCALE_WINDOW = 0`. Re-sweep if the data
changes shape; a smaller sheet usually wants smaller trees.

Two caveats that matter more than the table. This is **synthetic data whose
structure I designed**, which flatters GBM: real card data is messier, so run
`sweep.py` on your own sheet and believe that instead. And **GBM trains on your
history while TimesFM does not** — on a short sheet GBM degrades sharply and
TimesFM barely notices.

MiniRocket was implemented and measured too: mean MAE 4,597, worse than seasonal
naive, winning no origin. It is a classification transform whose pooling
deliberately discards magnitude, which is the thing forecasting needs. It was
dropped from the CLI; the implementation is in git history at commit `a04b108`.

---

## 10. Improving a poor result

In the order that has actually paid off:

**1. Check calibration is on.** If the gradient-boosted model's coverage is in
the 40s or 50s, `CALIBRATE_INTERVALS` is not taking effect. This costs nothing
in point accuracy and is the single largest coverage fix available.

**2. Check the covariates are live.** The loading section must say how many
holidays it found. If it says NONE, `pip install holidays` — that was worth 42%
in testing.

**3. Adjust for drift, if a shorter context beat using everything.**
`--rescale 90` on either script, or `RESCALE_WINDOW = 90` in `settings.py`.

How it works, on one day of real data:

```
2019-06-04   actual 340,644,948   typical day then  62,609,090   ->  5.44
2026-06-02   actual 919,474,107   typical day then 660,387,407   ->  1.39
```

Each day is divided by how large a typical day was **at that time**, measured
over the trailing 90 days. A quiet 2026 day is a much bigger number than a busy
2019 day, and in lira the model cannot tell those apart; in "multiples of a
typical day" it can. The forecast is made on the levelled series and then
multiplied back by today's scale, so the answer comes out in lira.

The median is used rather than the mean, so settlement spikes do not inflate the
scale, and it is shifted one day so a value never helps set its own scale.

**Better, if you have the figures: use a real price index.** `--deflator
tufe.csv`, or `DEFLATOR_FILE` in `settings.py`.

`tufe.csv` is in the repo, current to 2026-08. To refresh it:

```bash
python fetch_tufe.py
```

It combines two sources because neither is enough alone: OECD via FRED gives
real index values but lags about a year, and TÜİK publishes the official annual
rate for recent months, which is enough to extend the index forward. It
cross-checks the two on the months they share and refuses to splice them if the
rates disagree, rather than writing a plausible-looking file.

TÜİK's API rejects ordinary requests, and a headless browser too, so the script
drives a real one through Playwright — which is why a window opens. Install it
once with `pip install playwright && python -m playwright install chromium`.

Any file with a date column and an index column works, so ENAG or a
sector-specific index can be dropped in instead — the file is yours.

```
date,index
2019-01-01,100.0
2019-02-01,101.0
```

The difference matters. `--rescale` divides by how big a typical day was, which
removes **all** growth — if the business genuinely doubled in real terms, that
is erased along with the price rises. A price index removes only the price
rises and leaves real growth for the model to learn. Measured over three
origins on a series where prices rose 10x:

| inflation handling | MAE | pinball | coverage |
|---|---|---|---|
| off | 575,870,157 | 201,791,729 | 78% |
| **price index** | **537,158,215** | **161,603,852** | 84% |
| trailing scale | 553,317,200 | 169,631,995 | 88% |

Past the end of your index file, prices are extended at the average monthly rate
of its final year, and that assumed rate is reported. Everything is measured
against the last date in your history, so forecasts come back in today's lira.
 That
pattern means the old data is hurting: a normal day years ago is not a normal
day now. `RESCALE_WINDOW = 90` divides every day by how big a typical day was at
that time, forecasts the levelled series, then restores today's scale — no
inflation figures needed. On a series drifting 8.7x it moved the
gradient-boosted model from MAE 584M to 552M, pinball 200M to 174M, and coverage
77% to 89%. It made TimesFM slightly worse, which fits: TimesFM normalises
internally, while the trees split on absolute values and drift genuinely hurts
them. Sweep it rather than assuming.

**4. Push the boosting rounds up.** If `rounds=600` wins nearly every pairing,
the grid stopped before the optimum. The full sweep now goes to 2000.

**5. Consider the target.** `--target purchases` is a cleaner problem than
`--target growth`: no sign changes and no payment spikes. If you only need the
spending side, it forecasts considerably better.

---

## 11. Troubleshooting

**`No module named timesfm3`** — you cloned but did not install, or your IDE
uses a different interpreter. Install from inside the console with
`%pip install -e .[torch]`, which uses the Python you are actually running. See
`SETUP-LOCAL.md`.

**`failed to activate VS environment: no vswhere.exe`** — pip found no
ready-built package for your Python and tried to compile one. Do not install
Visual Studio; see the Install section above.

**`CalledProcessError` from `subprocess.py` with no explanation** — a
`subprocess.run(..., check=True)` call discarded pip's message. Use `%pip` in
the console instead, or pass `capture_output=True, text=True` and print
`.stdout` and `.stderr`.

**Detected the wrong columns** — check the first printed section, then pass
`--date-col` / `--balance-col`.

**`need more history`** — GBM needs roughly `max(lag) + max(window) + horizon +
30` steps. Shorten `--horizon`, or use `--model timesfm`, which is zero-shot and
copes with far less.

**Forecast is flat or zero** — usually an all-NaN or constant context. Check the
`mean change per step` line.

**Hangs on an offline machine** — TimesFM is trying to reach HuggingFace. Set
`TIMESFM_WEIGHTS` in `settings.py` to your local weights folder; see
`OFFLINE-WEIGHTS.md`.

**Everything runs locally.** Your data never leaves the machine.

**Do not commit your data.** `.gitignore` already excludes `*.xlsx`,
`forecast.csv` and `sweep_results.csv`, because this fork is public.
