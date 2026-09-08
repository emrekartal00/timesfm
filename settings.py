"""
==============================================================================
SETTINGS — this is the file to edit. You do not need to know how to code.
==============================================================================

Every setting below is one line shaped like:

    NAME = value          # a sentence explaining what it does

To change something: edit the value on the right of the "=", save the file, and
run the script again. That is the whole process. Nothing else needs touching.

Two rules that will save you time:

  1. True and False must be capitalised exactly like that.
  2. Numbers have no quotes. Words do: "business", not business.

If you break something, the script will print an error and change nothing. You
can always undo by running:  git checkout settings.py

Every setting here can also be overridden on the command line for a single run
without editing anything. The command-line flag is named in each comment.
==============================================================================
"""

# =============================================================================
# PART 1. WHICH CALENDAR FACTS THE MODELS GET
# =============================================================================
# Neither model knows what a date is. It sees a list of numbers. These settings
# hand it facts about each day, INCLUDING the days it is forecasting, because a
# calendar is knowable in advance. This is the single biggest lever in the whole
# pipeline: switching them all off made TimesFM 80% worse in testing
# (error 1,180 -> 2,122).
#
# Turn one off by changing True to False. Turn it back on the same way.
# Try switching one off, rerun, and see whether the error goes up or down.

USE_DAY_OF_WEEK = True
# Tells the model which weekday each day is. Switch this off if your spending
# has no weekly rhythm. Almost every card does, so keep it on unless testing.

USE_DAY_OF_MONTH = True
# Tells the model where in the month each day sits. This is what lets it learn
# "spending rises after the 25th" or "the payment lands mid-month".

USE_MONTH_OF_YEAR = True
# Tells the model which month it is, so it can learn things like a December
# spike. Needs at least two years of history to be worth anything -- with one
# year the model sees each month only once and cannot tell a pattern from an
# accident. Set this to False if your sheet is shorter than two years.

USE_WEEKEND_FLAG = True
# A simple yes/no for Saturday and Sunday. Overlaps with USE_DAY_OF_WEEK but
# makes the weekend easier for the model to spot.

USE_MONTH_EDGES = True
# Two yes/no flags for the first and last day of the month. Useful when
# salaries or direct debits land exactly on those days.

USE_STATEMENT_DAYS = True
# A yes/no flag for the days of the month your payments actually land on. The
# script finds these itself by looking at your history -- you do not enter them.
# It prints what it found, for example:
#     payments recur on days of month [14, 15, 16]
# This is the most valuable covariate for a signed forecast. Keep it on.

USE_HOLIDAYS = True
# Tells the models which days are public holidays, plus the day before one, the
# day after one, and "bridge" days (a lone working day trapped between a holiday
# and a weekend, which people often take off).
# Holiday eves are frequently the busiest spending days of the year, and the
# holidays themselves among the quietest, so this is worth having.
# Set to False to switch all four holiday facts off.

HOLIDAY_COUNTRY = "TR"
# Which country's public holidays to use. Quotes required. "TR" is Turkey.
# This handles the religious holidays correctly, which matters: Ramazan and
# Kurban Bayramı move about eleven days earlier every year, so a hard-coded
# list would quietly go wrong after one year.
# Other examples: "DE" Germany, "GB" United Kingdom, "US" United States.
# Set to None (no quotes) to switch holidays off entirely.

USE_SCHEDULED_PAYMENT_DAYS = True
# Tells the models about your regular payment dates, described just below.
# This is separate from USE_STATEMENT_DAYS: that one is guessed from your
# history, this one is the schedule you already know. Having both is fine and
# usually better, because each catches what the other misses.

PAYMENT_DAYS_OF_MONTH = (4, 14, 24)
# The days of the month your commercial card payments are due. Keep the commas
# and brackets exactly as they are.
# You do NOT need to list the shifted dates. A payment due on the 14th that
# lands on a Sunday or a public holiday moves to a working day, which is why you
# see the 13th or the 15th, and the script works those out for you -- for every
# due date it also marks the nearest working day before and the nearest working
# day after, so the model can learn which way your bank moves them. It also
# measures how far each day is from the nearest due date, so it can see a
# payment coming rather than only recognising the day it lands.
# If your card is personal rather than commercial, or the dates differ:
#     PAYMENT_DAYS_OF_MONTH = (15,)          one date a month, note the comma
#     PAYMENT_DAYS_OF_MONTH = (1, 15)        two dates a month
# Set to () to switch this off.


# =============================================================================
# PART 2. CALIBRATING THE GRADIENT-BOOSTED MODEL (GBM)
# =============================================================================
# GBM builds hundreds of small decision trees, each correcting the mistakes of
# the ones before it. These four settings control that. The usual failure is
# OVERFITTING: the model memorises your history instead of learning from it,
# looks brilliant on old data, and forecasts badly.
#
# HOW TO TELL YOU HAVE OVERFIT: run with --rolling 6. If the error is much worse
# than you expected while the model looks confident (coverage well under 80%),
# it has memorised. Make the model SIMPLER: lower GBM_NUM_LEAVES, raise
# GBM_MIN_DATA_IN_LEAF.
#
# The fastest way to calibrate is not to guess. Run this and read the table:
#     python sweep.py --excel yourfile.xlsx --only gbm --origins 6
# It tries many combinations and ranks them. Then copy the winner in here.

GBM_NUM_LEAVES = 31
# How complicated each tree is allowed to be. THE MOST IMPORTANT SETTING HERE.
# Higher = the model can learn finer patterns, and is likelier to memorise.
# Lower = smoother, safer, may miss real detail.
# Sensible range: 7 to 63. Try 15 if you have less than a year of data.
# In testing, 15 beat the default 31 (error 1,055 against 1,117).
# Command line: --num-leaves 15

GBM_LEARNING_RATE = 0.05
# How big a correction each tree is allowed to make. Think of it as caution.
# Lower = more careful and usually more accurate, but you must raise
# GBM_ROUNDS to compensate, so it takes longer.
# Sensible range: 0.01 to 0.2. If you halve this, roughly double GBM_ROUNDS.
# Example: GBM_LEARNING_RATE = 0.025 together with GBM_ROUNDS = 600
# Command line: --learning-rate 0.025

GBM_ROUNDS = 300
# How many trees to build. More is not automatically better: past a point the
# extra trees only memorise. This works together with GBM_LEARNING_RATE --
# a low rate needs many rounds, a high rate needs few.
# Sensible range: 100 to 1000.
# Command line: --rounds 600

GBM_MIN_DATA_IN_LEAF = 20
# The fewest days that must support any conclusion the model draws. This is
# your main protection against memorising.
# Raise it (say to 40) if the model is overfitting. Lower it (say to 10) if you
# have very little data and the model seems to be learning nothing.
# Sensible range: 5 to 50.

GBM_FEATURE_FRACTION = 0.9
# What share of the available facts each tree is allowed to look at. Using
# slightly less than everything makes the trees differ from one another, which
# usually helps. Leave this alone unless you are experimenting.
# Sensible range: 0.5 to 1.0.


# =============================================================================
# PART 3. WHAT THE GBM LOOKS BACK AT
# =============================================================================
# These are the "how much did I spend N days ago" facts GBM gets. TimesFM does
# not use these -- it reads the raw history itself.

GBM_LAGS = (1, 2, 3, 4, 5, 6, 7, 10, 14, 21, 28, 35)
# Which previous days to look at, counting backwards. 1 means yesterday, 7 means
# the same weekday last week, 28 means roughly the same day last month.
# Keep the commas and the brackets exactly as they are.
# If your data is WEEKLY rather than daily, use smaller numbers:
#     GBM_LAGS = (1, 2, 3, 4, 8, 12)
# If you have a strong yearly pattern and years of data, add a long one:
#     GBM_LAGS = (1, 2, 3, 4, 5, 6, 7, 10, 14, 21, 28, 35, 364)
# WARNING: the longest number here plus the longest GBM_WINDOWS below is how
# much history GBM throws away before it can start learning. Long lags on a
# short sheet will cause a "need more history" error.

GBM_WINDOWS = (7, 14, 28)
# Over what stretches to summarise the recent past (average, spread, smallest,
# largest). 7 means "the last week", 28 means "the last month".
# For weekly data try: GBM_WINDOWS = (4, 8, 13)


# =============================================================================
# PART 4. DEFAULTS FOR YOUR DATA
# =============================================================================
# These save you retyping the same command-line flags every time. Whatever you
# type on the command line still wins over what is written here.

DEFAULT_TARGET = "purchases"
# What to forecast. Three choices, and the quotes are required:
#   "purchases"   spending only. Payments are stripped out.
#   "net_change"  spending AND payments, so the number can go negative.
#   "balance"     the outstanding amount itself, not the daily change.
# Command line: --target net_change

DEFAULT_HORIZON = 30
# How many steps ahead to forecast. On a weekends-dropped grid a step is a
# working day, so 20 is about four weeks. The script always prints the real
# dates it covered, so you never have to work this out yourself.
# Command line: --horizon 60

DEFAULT_ROLLING_ORIGINS = 6
# How many times to re-test the model on a different slice of your history.
# One test is one lucky or unlucky month and rankings flip on it. Six is enough
# to believe the answer. Higher is slower.
# Command line: --rolling 10

DEFAULT_GRID = "auto"
# Whether Saturdays and Sundays stay in the data. Quotes required.
#   "auto"      decide by looking at your sheet. Recommended.
#   "business"  always drop weekends.
#   "calendar"  always keep all seven days.
# "auto" keeps weekends when they carry real spending and drops them when they
# are empty padding. The script prints which it chose.
# Command line: --grid calendar


# =============================================================================
# PART 5. TIMESFM SETTINGS
# =============================================================================

TIMESFM_STRATEGY = "auto"
# How TimesFM handles a target that goes negative. Quotes required.
#   "auto"          pick automatically. Recommended.
#   "multichannel"  forecast spending, payments and balance together, then
#                   combine. What "auto" chooses for "net_change", and the best
#                   performer in testing.
#   "raw"           forecast the signed series directly. Simplest.
#   "signed-log"    squash the big payment spikes before forecasting.
# Command line: --strategy raw

TIMESFM_CONTEXT = None
# How much history to feed TimesFM. None means all of it, and None is written
# without quotes. Set a number to use only the most recent N steps.
# Use this if your older data is not comparable any more -- a changed credit
# limit, a different card, a business that changed shape, or a stretch of high
# inflation that makes 2019 amounts mean something different from 2026 amounts.
# For daily data: 365 is one year, 730 two years, 1825 five years.
# Example: TIMESFM_CONTEXT = 730
# Do not guess at this. Find it:
#     python sweep.py --excel yourfile.xlsx --only timesfm --preset full
# Command line: --context 365

TIMESFM_ZNORM = False
# Rescales your numbers before forecasting and undoes it afterwards. Worth
# trying if your amounts are very large or very small and the forecast looks
# wrong in an obvious way. Usually makes little difference.
# Command line: --znorm
