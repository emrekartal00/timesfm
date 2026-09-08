# Forecasting your own data

Guide to the forecasting files in this fork. Everything here was measured
on real runs, not assumed.

| file | what it is |
|---|---|
| `credit_card_forecast.py` | **Start here.** One run: seasonality report, backtest, forecast, CSV, plot. |
| `sweep.py` | Runs many configurations of both models over the same backtest and ranks them. |
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

## 3. What the script does to your data

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

## 4. `credit_card_forecast.py` arguments

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
| `--device` | auto | TimesFM | `cuda`, `mps`, `cpu` |
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

## 5. Changing things without touching code — `settings.py`

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

## 6. `sweep.py` arguments

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

## 7. Reading the numbers

| metric | meaning |
|---|---|
| **MAE** | average error of the single number. Lower is better. |
| **pinball** | scores the whole p10/p50/p90 range, not just the point. Lower is better. **Trust this one** on a spiky series. |
| **coverage** | share of actual values that landed inside p10–p90. Should be ≈80%. |
| **MAE spike** | error on payment days only. |
| **MAE other** | error on ordinary days only. |

**Coverage is the one people ignore.** A model with the lowest MAE and 50%
coverage is not better — it is a sharper guess that is dishonest about its own
uncertainty, and it will under-warn you in a bad month.

**sMAPE is not reported** for signed targets. It is meaningless on a series that
crosses zero; it rated the *worst* model best in testing.

---

## 8. What the measurements showed

On synthetic data shaped like a card statement, over rolling origins:

| finding | numbers |
|---|---|
| Covariates matter more than the model choice | TimesFM 1,180 with, 2,122 without |
| GBM is the sharper point forecast | GBM 1,055 MAE vs TimesFM 1,180 |
| TimesFM is the honest interval | 86% coverage vs GBM's 62–71% against a nominal 80% |
| Weekend padding hurts when weekends are dead | mean MAE −37%, payment-day MAE −72% dropping them |
| Multichannel beats modelling the signed series | MAE 2,940 → 2,428, payment-day 45,616 → 36,435 |
| Seasonal naive is a floor, not a rival | 3,339 |

**Use GBM for the number, TimesFM for the range** — or run both, which is the
default.

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

## 9. Troubleshooting

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
