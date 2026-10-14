# Task: MOF screening program (`candidate.py`)

Write a program that ranks metal-organic frameworks (MOFs) by how much CO₂ they
adsorb, and returns the best candidates from a pool of materials whose measured
values you cannot see.

## What the score means

Absolute CO₂ uptake at **298 K, 1 bar (100 000 Pa), rigid framework, UFF force
field with DDEC charges**, in **mol/kg**. Higher is better. This is simulated
single-component adsorption data; it is not a mixed-gas selectivity and it is not
a statement about any industrial operating condition.

## Interface (must keep working)

```python
def rank(train_X, train_y, pool_X, *, k: int, seed: int) -> list[str]:
    """Fit using only `train_X`/`train_y`; return `k` distinct material ids from
    `pool_X`, ordered from most to least recommended."""
```

* `train_X`, `pool_X` — `pandas.DataFrame`, one row per material, first column
  `material_id`, remaining columns numeric descriptors with a shared schema.
  The id column is **not** a feature.
* `train_y` — aligned with `train_X` rows, in mol/kg.
* Return `k` **distinct** ids that all appear in `pool_X`, best first.
* Returning fewer than `k`, a duplicate, an id outside the pool, a non-string
  entry, or anything that is not an ordered sequence of ids scores **0** for that
  evaluation. So does raising, hanging, or exceeding the time limit.

## What is measured

A fixed external grader keeps the true uptake values for the pool. With
`K = 10` it computes

```
Q(A)  = mean true uptake over your K recommendations        (mol/kg)
Q*(D) = mean true uptake of the D best materials            (private upper bound)
S     = Q(A) / Q*(D)                                        (0..1, higher is better)
```

The grader never accepts a score you report about yourself, and it does not tell
you which materials were good, what `Q*` was, or where you lost points. You see
only `S` as the result of a whole evaluation.

Ties in the true labels are broken by material id, so `S` is well defined.

## Rules

* Use only **Python 3.11+, NumPy, pandas, and scikit-learn**. The grader imports
  `candidate.py` under an interpreter that has exactly those. Do not install
  anything and do not add other dependencies.
* **Imputation, scaling, feature selection and hyper-parameter choice may only be
  fitted on `train_X`/`train_y`**, using the same fitted transform on `pool_X`.
  Cross-validation inside the training set is fine.
* Do **not** look for, or try to read, the grading script or the pool's true
  labels. They are not in your workspace and reading them is not a shortcut that
  survives review.
* There is no network access. Do not attempt to fetch anything.
* Your program must work for **any** well-formed pool. Do not depend on row
  order, pool size, or hard-coded material ids.
* Keep the module importable and keep `rank` importable from it. Helper
  functions, classes and module-level constants are all fine.

## Cost of an evaluation

One evaluation imports your module, calls `rank` once, and stops. Fitting and
ranking together must finish inside the time limit announced with the call. A
timeout is a failure, not a retry.

## Where you start

`candidate.py` currently implements a deliberately plain baseline: median
imputation, standardisation, and a ridge regression with a fixed penalty. It is
a working program, not a good one. Improving feature treatment, the supervised
model, and how the ranking handles ties and uncertainty is the actual task.
