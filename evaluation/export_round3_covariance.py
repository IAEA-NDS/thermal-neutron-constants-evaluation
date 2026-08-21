"""Export the full posterior covariance of the round-3 TNC fit.

`benchmark_capote_round3.run_benchmark()` computes `result['cov_free']`, the
posterior covariance over the 30 free parameters, but only its diagonal ever
reaches `round3_summary.json` (as `unc_pct`). Anything that wants to USE the
round-3 result as a prior -- for instance replacing the Axton block in the GMA
standards database -- needs the off-diagonal terms as well, because those
parameters are strongly correlated and the diagonal alone would import a
tightening without the correlations that make it self-consistent.

This runs the same fit and writes the matrix. Nothing else is modified.

UNITS. `unc_pct = sqrt(cov_free[i,i])`, so `cov_free` is a RELATIVE
covariance expressed in percent^2. Divide by 1e4 for a plain relative
covariance; multiply by outer(value, value)/1e4 for an absolute one. Both
forms are written out.

Output: round3_covariance.json  (parameter order, values, relative and
absolute covariance, correlation matrix)
"""
import io
import json
import contextlib
from pathlib import Path

import numpy as np

import benchmark_capote_round3 as r3

OUT = Path(__file__).parent / 'round3_covariance.json'


def main():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result, all_fitted = r3.run_benchmark()

    cov_pct2 = np.array(result['cov_free'], dtype=float)
    n = cov_pct2.shape[0]
    # all_fitted is built free-parameters-first, in cov_free order, then the
    # derived quantities are appended; take the first n keys and VERIFY the
    # correspondence against the diagonal rather than trusting the ordering
    keys = list(all_fitted.keys())[:n]
    vals, uncs = [], []
    for k in keys:
        v = all_fitted[k]
        v, u = (v[0], v[1]) if isinstance(v, (tuple, list)) else (v['value'],
                                                                 v['unc_pct'])
        vals.append(float(v)); uncs.append(float(u))
    diag = np.sqrt(np.diag(cov_pct2))
    bad = [(k, a, b) for k, a, b in zip(keys, diag, uncs)
           if abs(a - b) > 1e-8 * max(1.0, abs(b))]
    if bad:
        raise IndexError(f'parameter order does not match the diagonal: {bad[:3]}')
    print(f'[COV] {n} free parameters, diagonal matches the reported unc_pct')

    vals = np.array(vals)
    rel = cov_pct2 / 1e4
    absolute = rel * np.outer(vals, vals)
    sd = np.sqrt(np.diag(rel))
    corr = rel / np.outer(sd, sd)

    off = corr - np.eye(n)
    i, j = np.unravel_index(np.argmax(np.abs(off)), off.shape)
    print(f'[COV] largest off-diagonal correlation: {keys[i]} / {keys[j]} '
          f'= {off[i, j]:+.4f}')
    print(f'[COV] |corr| > 0.5 on {int((np.abs(off) > 0.5).sum() // 2)} pairs, '
          f'> 0.9 on {int((np.abs(off) > 0.9).sum() // 2)}')
    ev = np.linalg.eigvalsh(rel)
    print(f'[COV] eigenvalues {ev.min():.3e} .. {ev.max():.3e}  '
          f'(positive definite: {ev.min() > 0})')

    json.dump({
        'round': 3, 'date': '2026-06-23',
        'note': 'posterior covariance of the 30 free parameters',
        'parameters': keys,
        'values': vals.tolist(),
        'unc_pct': uncs,
        'cov_percent2': cov_pct2.tolist(),
        'cov_relative': rel.tolist(),
        'cov_absolute': absolute.tolist(),
        'correlation': corr.tolist(),
    }, open(OUT, 'w'), indent=1)
    print(f'[COV] wrote {OUT}')


if __name__ == '__main__':
    main()
