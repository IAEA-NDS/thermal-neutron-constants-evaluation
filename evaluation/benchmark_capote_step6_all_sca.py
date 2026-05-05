"""Benchmark: Step 6 of the Capote 2026-05-04 fit chain — All SCR -> SCA, drop Lemmel SCA-SCR.

Step 6 finalises the chain:

  DELETE: Lemmel SCA-SCR datapoints (No. 89, 91, 93, 95)  — 4 entries
  CONVERT: ALL (ABS X)+SCR X  ->  (ABS X)+SCA X  for every transmission
           entry in the database (26 conversions). Per Roberto's
           parenthetical "reducing 4 fitting variables SCR", this drops
           SCR from the active fit since there are no more constraints
           on it.

Consequence: the SCR primary parameters (SCR 33, 35, 39, 41) become
unconstrained — no measurement function references them, and no SCA-SCR
datapoint constrains them either. They are therefore removed from the
parameter space (added to `unconstrained`), consistent with the existing
treatment of CA 40/42, GC116, GA116. Decision logged in the appendix.

This benchmark also targets Roberto's step-5-vs-step-6 comparison.

Data file: axton1986_data_capote_step6_all_sca.json (186 measurements).
"""

import sys
import re
import json
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

from utils.gls import (build_param_space, create_propagate_fn, iterative_gls,
                       compute_derived_uncertainties)
from utils.covmat import create_relative_covmat
from axton1986_defaults import DEFAULT_INITIAL_PARAMS, DEFAULT_FIXED_PARAMS


MAXWELLIAN_DERIVED_FUNCS = frozenset({
    'FA', 'FF', 'CAP', 'FFH',
    'F1ETA', 'F2ETA', 'F3ETA',
    'FLEM', 'F1BIG',
    'F1CAB', 'F2CAB', 'F3CAB', 'F4CAB', 'F5CAB',
    'GC116', 'GA116',
})

# Step 1: Lounsbury alpha INCLUDED (no longer in the exclusion set)
LOUNSBURY_ALPHA = set()
PARAM_EXCLUDED = {157, 158}  # CA 40, CA 42
KEPT_DESPITE_MAXWELLIAN = {5}  # CAP 34


def is_maxwellian_derived(expr):
    funcs = set(re.findall(r'([A-Z][A-Z0-9]+)', expr))
    return any(f in MAXWELLIAN_DERIVED_FUNCS for f in funcs)


def get_measurement_ids(data):
    excluded = LOUNSBURY_ALPHA | PARAM_EXCLUDED
    return [m['no'] for m in data['measurements']
            if (not is_maxwellian_derived(m['measured_function'])
                or m['no'] in KEPT_DESPITE_MAXWELLIAN)
            and m['no'] not in excluded]


# Axton Table 8 reference (no-Lounsbury microscopic) — kept for context
TABLE8_REFERENCE = {
    ('SCA', 33): (12.3204, 5.681),
    ('SCA', 35): (16.2388, 7.386),
    ('SCA', 39): (7.8966, 12.380),
    ('SCA', 41): (11.9740, 21.595),
    ('SCR', 33): (11.1480, 8.147),
    ('SCR', 35): (14.5259, 9.448),
    ('SCR', 39): (6.8131, 25.630),
    ('SCR', 41): (10.8285, 35.646),
    ('ABS', 33): (575.2391, 0.321),
    ('ABS', 35): (681.1651, 0.249),
    ('ABS', 39): (1018.8813, 0.362),
    ('ABS', 41): (1382.1872, 1.079),
    ('FIS', 33): (533.2694, 0.457),
    ('FIS', 35): (585.0603, 0.277),
    ('FIS', 39): (748.5181, 0.340),
    ('FIS', 41): (1020.5836, 1.126),
    ('CA', 33): (41.9697, 4.182),
    ('CA', 35): (96.1049, 1.811),
    ('CA', 39): (270.3632, 1.174),
    ('CA', 41): (361.6036, 1.711),
    ('CAP', 34): (95.8339, 2.071),
    ('NUB', 33): (2.4856, 0.218),
    ('NUB', 35): (2.4261, 0.188),
    ('NUB', 39): (2.8794, 0.207),
    ('NUB', 41): (2.9406, 0.220),
    ('NUB', 52): (3.7644, 0.132),
    ('ETA', 33): (2.3042, 0.271),
    ('ETA', 35): (2.0838, 0.255),
    ('ETA', 39): (2.1153, 0.311),
    ('ETA', 41): (2.1713, 0.399),
    ('ALPHA', 33): (0.0787, 4.494),
    ('ALPHA', 35): (0.1643, 1.962),
    ('ALPHA', 39): (0.3612, 1.280),
    ('ALPHA', 41): (0.3543, 1.668),
    ('WGA', 33): (0.9988, 0.120),
    ('WGA', 35): (0.9787, 0.092),
    ('WGA', 39): (1.0767, 0.269),
    ('WGA', 41): (1.0444, 0.191),
    ('WGF', 33): (0.9966, 0.201),
    ('WGF', 35): (0.9778, 0.092),
    ('WGF', 39): (1.0562, 0.275),
    ('WGF', 41): (1.0452, 0.670),
    ('F3ETA', 33): (746.4586, 0.683),
    ('F3ETA', 35): (721.2735, 0.507),
    ('F3ETA', 39): (1179.4821, 0.805),
    ('F3ETA', 41): (1693.1349, 1.774),
    ('CA', 40): (289.3312, 0.482),
    ('CA', 42): (18.5148, 2.161),
}

NUCLIDES = [33, 35, 39, 41]


def derived_quantity_fn(params_tf, reac_map):
    def get(q, n):
        return params_tf[reac_map[(q, n)]]
    derived = {}
    for n in NUCLIDES:
        ABS = get('ABS', n)
        FIS = get('FIS', n)
        NUB = get('NUB', n)
        derived[('CA', n)] = ABS - FIS
        derived[('ALPHA', n)] = (ABS - FIS) / FIS
        derived[('ETA', n)] = NUB * FIS / ABS
        if ('WGA', n) in reac_map and ('WGF', n) in reac_map:
            FA = ABS * get('WGA', n)
            FF = FIS * get('WGF', n)
            derived[('FA', n)] = FA
            derived[('FF', n)] = FF
            derived[('F1ETA', n)] = NUB * FF / FA
            derived[('F2ETA', n)] = NUB * FF
            derived[('F3ETA', n)] = NUB * FF - FA
    return derived


def run_benchmark():
    data_path = str(
        __import__('pathlib').Path(__file__).resolve().parent.parent
        / 'axton1986' / 'axton1986_data_capote_step6_all_sca.json'
    )
    with open(data_path) as f:
        data = json.load(f)

    meas_ids = get_measurement_ids(data)
    meas_by_no = {m['no']: m for m in data['measurements']}

    full_covmat, full_nos = create_relative_covmat(data)
    idx = [full_nos.index(no) for no in meas_ids]
    covmat = full_covmat[np.ix_(idx, idx)]
    measured = np.array([meas_by_no[no]['input_value'] for no in meas_ids])
    exprs = [meas_by_no[no]['measured_function'] for no in meas_ids]

    unconstrained = {
        ('GC116', 39), ('GC116', 40), ('GC116', 41), ('GC116', 42),
        ('GA116', 39), ('GA116', 41),
        ('CA', 40), ('CA', 42),
        # Step 6: SCR has no remaining constraints (all ABS+SCR converted to
        # ABS+SCA, all Lemmel SCA-SCR datapoints deleted). Removed from
        # parameter space — matches Roberto's "(reducing 4 fitting variables SCR)".
        ('SCR', 33), ('SCR', 35), ('SCR', 39), ('SCR', 41),
    }
    initial_params = {k: v for k, v in DEFAULT_INITIAL_PARAMS.items()
                      if k not in unconstrained}

    param_vec, reac_map, free_idx = build_param_space(
        initial_params, DEFAULT_FIXED_PARAMS)
    propagate = create_propagate_fn(reac_map, exprs)

    result = iterative_gls(
        measured, covmat, propagate,
        param_vec.copy(), free_idx, max_iter=30, tol=1e-10)

    derived = compute_derived_uncertainties(
        derived_quantity_fn, result, reac_map, free_idx)

    free_keys = list(initial_params.keys())
    all_fitted = {}
    for i, k in enumerate(free_keys):
        val = result['free_values'][i]
        unc_pct = np.sqrt(result['cov_free'][i, i])
        all_fitted[k] = (val, unc_pct)
    all_fitted.update(derived)

    print("=" * 78)
    print("CAPOTE 2026-05-04 FIT CHAIN — STEP 6: ALL SCR -> SCA, drop Lemmel SCA-SCR")
    print("=" * 78)

    print(f"\n{'Metric':<25s} {'This work':>12s}")
    print("-" * 40)
    print(f"{'Chi-squared':<25s} {result['chi2']:12.1f}")
    print(f"{'N measurements':<25s} {len(measured):12d}")
    print(f"{'N free parameters':<25s} {len(free_idx):12d}")
    print(f"{'Degrees of freedom':<25s} {result['dof']:12d}")
    print(f"{'Converged':<25s} {str(result['converged']):>12s}")
    print(f"{'Iterations':<25s} {result['n_iter']:12d}")

    print(f"\n--- Primary parameters (vs Axton Table 8 baseline) ---")
    print(f"{'Parameter':<12s} {'Fitted':>10s} {'Axton T8':>10s} "
          f"{'Diff%':>8s}  {'Unc%':>7s} {'Match':>6s}")
    print("-" * 64)
    n_pass = 0
    n_total = 0
    primary_order = ['SCA', 'SCR', 'ABS', 'FIS', 'NUB',
                     'WGA', 'WGF', 'CAP', 'CA']
    for q in primary_order:
        for n in NUCLIDES + ([34, 52] if q in ('CAP', 'NUB') else
                             ([40, 42] if q == 'CA' else [])):
            k = (q, n)
            if k not in TABLE8_REFERENCE or k not in all_fitted:
                continue
            ref_val, ref_unc = TABLE8_REFERENCE[k]
            fit_val, fit_unc = all_fitted[k]
            diff_pct = (fit_val - ref_val) / ref_val * 100
            n_total += 1
            threshold = 0.5 if ref_unc < 1.0 else 2.0
            passed = abs(diff_pct) < threshold
            if passed:
                n_pass += 1
            status = "OK" if passed else "DIFF"
            name = f"{q} {n}"
            unc_str = f"{fit_unc:.3f}" if fit_unc is not None else "  -  "
            print(f"  {name:<10s} {fit_val:10.4f} {ref_val:10.4f} "
                  f"{diff_pct:+8.4f}%  {unc_str:>7s} {status:>6s}")

    for n in NUCLIDES:
        k = ('ALPHA', n)
        if k not in all_fitted:
            continue
        ref_val, ref_unc = TABLE8_REFERENCE[k]
        fit_val, fit_unc = all_fitted[k]
        diff_pct = (fit_val - ref_val) / ref_val * 100
        n_total += 1
        threshold = 0.5 if ref_unc < 1.0 else 2.0
        passed = abs(diff_pct) < threshold
        if passed:
            n_pass += 1
        status = "OK" if passed else "DIFF"
        unc_str = f"{fit_unc:.3f}" if fit_unc is not None else "  -  "
        print(f"  {f'ALPHA {n}':<10s} {fit_val:10.4f} {ref_val:10.4f} "
              f"{diff_pct:+8.4f}%  {unc_str:>7s} {status:>6s}")

    print(f"\n--- Westcott-folded derived quantities (for K1 = F3ETA) ---")
    print(f"{'Quantity':<12s} {'Value':>10s} {'Unc%':>8s}  "
          f"{'Axton T8':>10s} {'Diff%':>8s}")
    print("-" * 60)
    for q in ['FA', 'FF', 'F1ETA', 'F2ETA', 'F3ETA']:
        for n in NUCLIDES:
            k = (q, n)
            if k not in all_fitted:
                continue
            fit_val, fit_unc = all_fitted[k]
            unc_str = f"{fit_unc:.3f}" if fit_unc is not None else "  -  "
            if k in TABLE8_REFERENCE:
                ref_val, _ = TABLE8_REFERENCE[k]
                diff_pct = (fit_val - ref_val) / ref_val * 100
                ref_str = f"{ref_val:10.4f}"
                diff_str = f"{diff_pct:+8.4f}%"
            else:
                ref_str = f"{'  -  ':>10s}"
                diff_str = f"{'  -  ':>8s}"
            name = f"{q} {n}"
            print(f"  {name:<10s} {fit_val:10.4f} {unc_str:>7s}  "
                  f"{ref_str} {diff_str}")

    print("-" * 64)
    print(f"Parameters within Table 8 tolerance: {n_pass}/{n_total}")
    print(f"Note: deviations vs Table 8 reflect cumulative chain through step 4.")

    return result, all_fitted


if __name__ == '__main__':
    run_benchmark()
