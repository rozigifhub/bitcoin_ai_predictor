2025-12-14 (Day 16)

# Post-mortem: Data Leakage (Min/Max Scaling)

## 1) Executive Summary
I found **data leakage** in my preprocessing pipeline: the min/max values used for normalization were computed from the **entire dataset** (including the test/future period). This makes the model evaluation look artificially good and not representative of real-world performance.

## 2) Symptoms / What I Noticed
- The printed `min`/`max` values looked like they were taken from the full CSV (train + test).
- Normalization was influenced by future (test) data even before training.

## 3) Root Cause
My initial `normalize_data` in `src/preprocess.py` computed `min()` and `max()` from the full DataFrame:

```py
nilai_max = nilai.max()
nilai_min = nilai.min()
```

Because `df` contained both train and test rows, the scaler parameters included future information (the test period).

Note: I split train/test later in `window.py`, but that split happened **after** normalization. The leakage already occurred at the moment the scaler was fit on the full dataset.

## 4) Fix / Solution
Principle: **fit the scaler only on training data**, then **transform** train and test using the same training scaler.

Implementation approach:
- Update `normalize_data` to support:
  - **FIT**: compute min/max from `df.iloc[:fit_end]` (training slice)
  - **TRANSFORM**: apply an existing `scaler` to any DataFrame

Example usage:
- Compute a time-based boundary: `split_raw = int(len(df) * 0.8)`
- Fit on train only, transform all rows:
  - `df_norm, scaler = normalize_data(df, fit_end=split_raw)`

Optional:
- If test values fall outside the training min/max range, `clip=True` can force normalized values into `[0, 1]`.

## 5) Verification
Checks to confirm the fix:
- Scaler equals training statistics:
  - `scaler["close"]["min"] == df_select.iloc[:split_raw]["close"].min()`
  - `scaler["close"]["max"] == df_select.iloc[:split_raw]["close"].max()`
- Test rows are transformed using the training scaler (not recomputed from test).

## 6) Lessons Learned / Prevention
- For time series, the split must be **time-based** (past = train, future = test).
- Any transform that “learns parameters” (scalers, imputers, PCA, etc.) must follow:
  - `fit` on train only
  - `transform` train and test using training parameters
- Safe pipeline order: **clean → choose split boundary → fit scaler → transform → windowing → train/evaluate**
