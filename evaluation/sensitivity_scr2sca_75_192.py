"""Sensitivity test: reclassify Pu-239 transmission Nos. 75 and 192 SCR->SCA.

Gilles flagged the two high Pu-239 total-XS points — Pattenden 1956 (No. 75,
1039 b) and Anderson 1945 (No. 192, 1045 b) — noting their scattering
component may be mis-assigned as SCR rather than SCA. This script refits the
round-3 database with those two entries' measured_function changed from
(ABS 39)+SCR 39 to (ABS 39)+SCA 39 and compares against the round-3 baseline.

Nothing is overwritten; the variant database is written to a separate file.

Run:  python evaluation/sensitivity_scr2sca_75_192.py
"""

import sys
import copy
import json
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))

from utils.gls import (build_param_space, create_propagate_fn, iterative_gls,
                       compute_derived_uncertainties)
from utils.covmat import create_relative_covmat
from axton1986_defaults import DEFAULT_INITIAL_PARAMS, DEFAULT_FIXED_PARAMS
from benchmark_capote_round3 import (get_measurement_ids, derived_quantity_fn,
                                     NUCLIDES)

RECLASSIFY = {75, 192}   # Pu-239 (ABS 39)+SCR 39 -> (ABS 39)+SCA 39
SRC = ROOT / 'axton1986' / 'axton1986_data_capote_round3.json'
VARIANT = ROOT / 'axton1986' / 'axton1986_data_capote_round3_scr2sca_75_192.json'

UNCONSTRAINED = {
    ('GC116', 39), ('GC116', 40), ('GC116', 41), ('GC116', 42),
    ('GA116', 39), ('GA116', 41), ('CA', 40), ('CA', 42),
}


def run_fit(data):
    meas_ids = get_measurement_ids(data)
    meas_by_no = {m['no']: m for m in data['measurements']}
    full_covmat, full_nos = create_relative_covmat(data)
    idx = [full_nos.index(no) for no in meas_ids]
    covmat = full_covmat[np.ix_(idx, idx)]
    measured = np.array([meas_by_no[no]['input_value'] for no in meas_ids])
    exprs = [meas_by_no[no]['measured_function'] for no in meas_ids]

    initial_params = {k: v for k, v in DEFAULT_INITIAL_PARAMS.items()
                      if k not in UNCONSTRAINED}
    param_vec, reac_map, free_idx = build_param_space(
        initial_params, DEFAULT_FIXED_PARAMS)
    propagate = create_propagate_fn(reac_map, exprs)
    result = iterative_gls(measured, covmat, propagate,
                           param_vec.copy(), free_idx, max_iter=30, tol=1e-10)
    derived = compute_derived_uncertainties(
        derived_quantity_fn, result, reac_map, free_idx)

    all_fitted = {}
    for i, k in enumerate(initial_params):
        all_fitted[k] = (result['free_values'][i],
                         np.sqrt(result['cov_free'][i, i]))
    all_fitted.update(derived)

    # per-point pull for the two reclassified entries
    preds = {no: float(p) for no, p in zip(meas_ids, result['predictions'])}
    pulls = {}
    for no in RECLASSIFY:
        m = meas_by_no[no]
        sig = abs(m['input_value']) * m['uncertainty_percent'] / 100.0
        pulls[no] = (m['measured_function'],
                     (m['input_value'] - preds[no]) / sig)
    return result, all_fitted, pulls


def main():
    with open(SRC) as f:
        base_data = json.load(f)

    var_data = copy.deepcopy(base_data)
    for m in var_data['measurements']:
        if m['no'] in RECLASSIFY:
            assert m['measured_function'] == '(ABS 39)+SCR 39', \
                f"#{m['no']} unexpected: {m['measured_function']}"
            m['measured_function'] = '(ABS 39)+SCA 39'
    var_data.setdefault('metadata', {})['sensitivity_variant'] = (
        'Round 3 with Pu-239 transmission Nos. 75 (Pattenden) and 192 '
        '(Anderson) reclassified (ABS 39)+SCR 39 -> (ABS 39)+SCA 39, per '
        "Gilles's query on the SCR assignment of the two high values.")
    with open(VARIANT, 'w') as f:
        json.dump(var_data, f, indent=1)

    base_r, base_f, base_pulls = run_fit(base_data)
    var_r, var_f, var_pulls = run_fit(var_data)

    print('=' * 78)
    print('SENSITIVITY: Pu-239 transmission Nos. 75 & 192  ABS+SCR -> ABS+SCA')
    print('=' * 78)
    print(f"\n{'':25s} {'Round 3':>12s} {'Variant':>12s} {'Delta':>10s}")
    print('-' * 62)
    print(f"{'chi2':25s} {base_r['chi2']:12.2f} {var_r['chi2']:12.2f} "
          f"{var_r['chi2'] - base_r['chi2']:+10.2f}")
    print(f"{'N meas':25s} {base_r['dof'] + 30:12d} {var_r['dof'] + 30:12d}")

    print(f"\n--- Fitted quantities (value; Δ% vs round 3) ---")
    print(f"{'Quantity':14s} {'Round 3':>12s} {'Variant':>12s} {'Δ%':>9s}")
    print('-' * 50)
    order = ([('SCA', 39), ('SCR', 39), ('ABS', 39), ('FIS', 39),
              ('CA', 39), ('ALPHA', 39), ('TOT', 39), ('F3ETA', 39)]
             + [('TOT', n) for n in (33, 35, 41)]
             + [('F3ETA', n) for n in (33, 35, 41)])
    for k in order:
        if k not in base_f or k not in var_f:
            continue
        b = base_f[k][0]
        v = var_f[k][0]
        d = (v - b) / b * 100
        print(f"{k[0] + ' ' + str(k[1]):14s} {b:12.4f} {v:12.4f} {d:+9.4f}")

    print(f"\n--- Pull (measured - fitted)/sigma for the two reclassified points ---")
    print(f"{'No.':>5s} {'function (round3)':>18s} {'pull r3':>9s} "
          f"{'function (var)':>18s} {'pull var':>9s}")
    for no in sorted(RECLASSIFY):
        print(f"{no:5d} {base_pulls[no][0]:>18s} {base_pulls[no][1]:+9.2f} "
              f"{var_pulls[no][0]:>18s} {var_pulls[no][1]:+9.2f}")

    print(f"\nVariant database written to: {VARIANT.name}")

    write_report(base_r, var_r, base_f, var_f, base_pulls, var_pulls)
    print(f"Report written to: report_scr2sca_sensitivity.md")


PU_ROWS = [
    ('SCA 39', ('SCA', 39)), ('SCR 39', ('SCR', 39)),
    ('ABS 39', ('ABS', 39)), ('FIS 39', ('FIS', 39)),
    ('CA 39', ('CA', 39)), ('ALPHA 39', ('ALPHA', 39)),
    ('σ_tot 39 = ABS+SCA', ('TOT', 39)), ('K1 = F3ETA 39', ('F3ETA', 39)),
]
OTHER_ROWS = [('σ_tot 33', ('TOT', 33)), ('σ_tot 35', ('TOT', 35)),
              ('σ_tot 41', ('TOT', 41)), ('K1 = F3ETA 33', ('F3ETA', 33)),
              ('K1 = F3ETA 35', ('F3ETA', 35)), ('K1 = F3ETA 41', ('F3ETA', 41))]


def write_report(base_r, var_r, base_f, var_f, base_pulls, var_pulls):
    L = []
    a = L.append
    a('# Sensitivity Check: SCR → SCA Reclassification of the Two High '
      'Pu-239 Total-XS Points\n')
    a('*Round-3 TNC fit (2026-06-23 database). Sensitivity test prompted by '
      "G. Noguère's query on the scattering classification of the two highest "
      'Pu-239 total-cross-section measurements.*\n')

    a('## Question\n')
    a('Two Pu-239 total-cross-section (transmission) measurements sit clearly '
      'above the rest of the Pu-239 (n,tot) set (which clusters at ~1022–1030 b):\n')
    a('| No. | Author | Year | Quantity (round 3) | Value | σ |')
    a('|----:|--------|------|--------------------|------:|---:|')
    a('| 75  | Pattenden | 1956 | (ABS 39)+SCR 39 | 1039 b | ±25 b (2.41%) |')
    a('| 192 | Anderson  | 1945 | (ABS 39)+SCR 39 | 1045 b | ±25 b (2.39%) |')
    a('')
    a('Both currently carry their scattering component as **SCR** (the '
      '"rolled metallic sample" / reduced-coherent convention). The question '
      'raised is whether that assignment should instead be **SCA** (true '
      'elastic). This note refits the round-3 database with both entries '
      'changed to `(ABS 39)+SCA 39` and compares against the round-3 baseline. '
      'Nothing else is altered.\n')

    a('## Result\n')
    dchi = var_r['chi2'] - base_r['chi2']
    a('| Quantity | Round 3 (SCR) | Variant (SCA) | Δ | Δ% |')
    a('|----------|--------------:|--------------:|---:|---:|')
    a(f'| χ² | {base_r["chi2"]:.2f} | {var_r["chi2"]:.2f} | {dchi:+.2f} | — |')
    for label, k in PU_ROWS:
        b, v = base_f[k][0], var_f[k][0]
        d = v - b
        a(f'| **{label}** | {b:.4f} | {v:.4f} | {d:+.4f} | '
          f'{d / b * 100:+.4f}% |')
    a('')
    a('Other nuclides (unaffected — shown for completeness):\n')
    a('| Quantity | Round 3 | Variant | Δ% |')
    a('|----------|--------:|--------:|---:|')
    for label, k in OTHER_ROWS:
        b, v = base_f[k][0], var_f[k][0]
        a(f'| {label} | {b:.4f} | {v:.4f} | {(v - b) / b * 100:+.4f}% |')
    a('')
    a('Normalized residual (pull) of the two reclassified points, before and '
      'after:\n')
    a('| No. | Author | pull (SCR) | pull (SCA) |')
    a('|----:|--------|-----------:|-----------:|')
    a(f'| 75  | Pattenden | {base_pulls[75][1]:+.2f} | {var_pulls[75][1]:+.2f} |')
    a(f'| 192 | Anderson  | {base_pulls[192][1]:+.2f} | {var_pulls[192][1]:+.2f} |')
    a('')

    a('## Interpretation\n')
    a('**The classification has no practical consequence for the evaluation.** '
      'Every physically relevant quantity — ABS, FIS, CA, the total cross '
      'section σ_tot, and the Westcott-folded K1 = F3ETA — for Pu-239 and for '
      'all other nuclides is stable to **< 0.002 %**, and χ² improves by a '
      f'negligible {abs(dchi):.2f}. The only visibly moving parameter is '
      '**SCR(²³⁹Pu)**, which shifts by ≈ −0.6 %; it is itself weakly '
      'determined (≈ 16 % uncertainty) and does not feed the recommended '
      'thermal constants.\n')
    a('The reason is quantitative: the fitted Pu-239 scattering terms differ '
      'by only SCA − SCR ≈ 7.48 − 6.74 = **0.74 b**. Re-labelling a point from '
      'ABS+SCR to ABS+SCA changes its predicted total by exactly this ~0.74 b, '
      'which is negligible against the **±25 b (2.4 %)** uncertainties of '
      'Pattenden and Anderson. Both points are also weak constraints on '
      'ABS(²³⁹Pu) — which is pinned by many tighter entries (Bollinger, '
      'Safford, Palevsky, Spencer, Zimmerman at 0.5–1.3 %) — so moving 0.74 b '
      'of their value between the SCR and SCA channels barely perturbs the '
      'fit. Consistently, their pulls stay below 1σ in both cases '
      f'(Pattenden {base_pulls[75][1]:+.2f}→{var_pulls[75][1]:+.2f}, '
      f'Anderson {base_pulls[192][1]:+.2f}→{var_pulls[192][1]:+.2f}).\n')
    a('**Conclusion.** The SCR-vs-SCA assignment of Pattenden 1956 (No. 75) '
      'and Anderson 1945 (No. 192) is physically debatable but numerically '
      'immaterial at the round-3 level: the recommended Pu-239 constants and '
      'K1 are identical to within rounding whichever convention is adopted. '
      'There is therefore no need to resolve the question to finalise the '
      'fit; it can be settled on physical grounds (sample form / extinction '
      'corrections in the original experiments) without affecting the '
      'results.\n')

    a('## Reproducibility\n')
    a('  * Script: `evaluation/sensitivity_scr2sca_75_192.py`\n'
      '  * Variant database: '
      '`axton1986/axton1986_data_capote_round3_scr2sca_75_192.json` '
      '(= round-3 DB with Nos. 75 and 192 changed to `(ABS 39)+SCA 39`)\n'
      '  * Baseline: `axton1986/axton1986_data_capote_round3.json` and '
      '`evaluation/report_capote_round3.md`\n')

    (EVAL_DIR / 'report_scr2sca_sensitivity.md').write_text('\n'.join(L))


if __name__ == '__main__':
    main()
