"""MOF screening candidate — ROOT workspace (the fixed starting point).

The public interface, called by the fixed external grader:

    rank(train_X, train_y, pool_X, *, k: int, seed: int) -> list[str]

`train_X` and `pool_X` are pandas DataFrames whose first column is
`material_id` and whose remaining columns are numeric descriptors sharing one
schema. `train_y` is aligned with `train_X` and measured in mol/kg. The return
value is `k` distinct ids drawn from `pool_X`, most recommended first.

What this starting point does — no more than it promises:

1. drop training descriptor columns that are entirely missing,
2. median-impute from the training set,
3. standardise with training-set statistics,
4. fit a ridge regression with a fixed penalty,
5. rank by prediction, breaking ties on material id so the output is
   deterministic.

Every transform here is fitted on the training rows only and then applied
unchanged to the pool. It is a working program and a weak one: the model is
linear, the penalty is a guess, the descriptors are used raw, and nothing about
the ranking is uncertainty-aware.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ALPHA = 1.0
ID_COLUMN = "material_id"


def rank(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    pool_X: pd.DataFrame,
    *,
    k: int,
    seed: int,
) -> list[str]:
    pool_ids = [str(value) for value in pool_X[ID_COLUMN].tolist()]

    train_features = train_X.drop(columns=[ID_COLUMN])
    pool_features = pool_X.drop(columns=[ID_COLUMN])

    # A column that is empty across the whole training set carries no signal and
    # cannot be imputed, so it leaves the schema for the pool too.
    usable = [name for name in train_features.columns if train_features[name].notna().any()]
    train_features = train_features.loc[:, usable]
    pool_features = pool_features.reindex(columns=usable)

    medians = train_features.median(axis=0)
    train_filled = train_features.fillna(medians)
    pool_filled = pool_features.fillna(medians)

    means = train_filled.mean(axis=0)
    scales = train_filled.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    train_scaled = (train_filled - means) / scales
    pool_scaled = (pool_filled - means) / scales

    model = Ridge(alpha=ALPHA, random_state=seed)
    model.fit(
        train_scaled.to_numpy(dtype=float),
        np.asarray(train_y, dtype=float).reshape(-1),
    )
    predictions = model.predict(pool_scaled.to_numpy(dtype=float))

    # Descending prediction, ascending id for ties: a stable, order-independent
    # ranking that does not depend on how the pool rows happen to be laid out.
    order = sorted(
        range(len(pool_ids)),
        key=lambda index: (-float(predictions[index]), pool_ids[index]),
    )
    return [pool_ids[index] for index in order[:k]]
