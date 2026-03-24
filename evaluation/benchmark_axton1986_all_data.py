"""Benchmark: Reproduce Axton 1986 Table 4 (all data, 167 measurements).

Runs the iterative GLS fit using all 167 measurements from the JSON
database and compares fitted values with Table 4 of the Axton 1986
report (GE/PH/01/86).

Expected results from the report:
  chi2 = 103, 167 measurements, 38 parameters, 129 dof
"""

import sys
import json
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

from utils.gls import build_param_space, create_propagate_fn, iterative_gls
from utils.covmat import create_relative_covmat
from axton1986_defaults import DEFAULT_INITIAL_PARAMS, DEFAULT_FIXED_PARAMS


# =========================================================================
# Table 4 reference values from Axton 1986 (all data fit)
# =========================================================================
# (parameter_name, nuclide): (fitted_value, relative_uncertainty_percent)

TABLE4_REFERENCE = {
    # Primary fitted parameters
    ('SCA', 33): (12.1861, 5.483),
    ('SCA', 35): (15.9828, 6.914),
    ('SCA', 39): (7.8966, 12.333),
    ('SCA', 41): (12.1878, 21.470),
    ('SCR', 33): (10.9169, 8.050),
    ('SCR', 35): (14.2071, 8.919),
    ('SCR', 39): (6.8000, 25.726),
    ('SCR', 41): (11.1717, 32.337),
    ('ABS', 33): (576.2174, 0.226),
    ('ABS', 35): (681.8303, 0.192),
    ('ABS', 39): (1018.9667, 0.282),
    ('ABS', 41): (1373.2025, 0.662),
    ('FIS', 33): (530.6953, 0.253),
    ('FIS', 35): (582.7836, 0.196),
    ('FIS', 39): (747.6236, 0.269),
    ('FIS', 41): (1011.8730, 0.652),
    ('NUB', 33): (2.4950, 0.161),
    ('NUB', 35): (2.4334, 0.147),
    ('NUB', 39): (2.8822, 0.177),
    ('NUB', 41): (2.9463, 0.196),
    ('NUB', 52): (3.7676, 0.126),
    ('WGA', 33): (0.9995, 0.107),
    ('WGA', 35): (0.9789, 0.084),
    ('WGA', 39): (1.0782, 0.227),
    ('WGA', 41): (1.0442, 0.190),
    ('WGF', 33): (0.9955, 0.142),
    ('WGF', 35): (0.9774, 0.083),
    ('WGF', 39): (1.0555, 0.212),
    ('WGF', 41): (1.0445, 0.527),
    ('CA', 40): (289.3296, 0.482),
    ('CA', 42): (18.5144, 2.161),
    ('CAP', 34): (95.8369, 2.071),
    ('GC116', 39): (1.3265, 6.056),
    ('GC116', 40): (1.0860, 2.000),
    ('GC116', 41): (1.1085, 2.457),
    ('GC116', 42): (1.1335, 2.670),
    ('GA116', 39): (1.1846, 1.521),
    ('GA116', 41): (1.1073, 0.819),
    # Derived quantities
    ('CA', 33): (45.5221, 1.535),
    ('CA', 35): (99.0467, 0.750),
    ('CA', 39): (271.3431, 0.786),
    ('CA', 41): (361.3294, 1.367),
    ('ETA', 33): (2.2979, 0.180),
    ('ETA', 35): (2.0799, 0.167),
    ('ETA', 39): (2.1147, 0.243),
    ('ETA', 41): (2.1710, 0.353),
    ('ALPHA', 33): (0.0858, 1.634),
    ('ALPHA', 35): (0.1700, 0.792),
    ('ALPHA', 39): (0.3629, 0.841),
    ('ALPHA', 41): (0.3571, 1.376),
    ('F3ETA', 33): (742.2525, 0.319),
    ('F3ETA', 35): (718.5735, 0.309),
    ('F3ETA', 39): (1175.7377, 0.475),
    ('F3ETA', 41): (1679.8787, 0.846),
}

REPORT_CHI2 = 103
REPORT_N_MEAS = 167
REPORT_N_PARAMS = 38
REPORT_DOF = 129


def compute_derived_quantities(result, reac_map):
    """Compute derived quantities from fitted primary parameters."""
    p = result['params']

    def get(q, n):
        return p[reac_map[(q, n)]]

    derived = {}
    for n in [33, 35, 39, 41]:
        FA = get('ABS', n) * get('WGA', n)
        FF = get('FIS', n) * get('WGF', n)
        derived[('CA', n)] = get('ABS', n) - get('FIS', n)
        derived[('ETA', n)] = get('NUB', n) * get('FIS', n) / get('ABS', n)
        derived[('ALPHA', n)] = derived[('CA', n)] / get('FIS', n)
        derived[('F3ETA', n)] = get('NUB', n) * FF - FA

    return derived


def run_benchmark():
    """Run the full benchmark and print comparison with Table 4."""
    data_path = str(
        __import__('pathlib').Path(__file__).resolve().parent.parent
        / 'axton1986' / 'axton1986_data.json'
    )
    with open(data_path) as f:
        data = json.load(f)

    # Build pipeline
    covmat, _ = create_relative_covmat(data)
    param_vec, reac_map, free_idx = build_param_space(
        DEFAULT_INITIAL_PARAMS, DEFAULT_FIXED_PARAMS)
    exprs = [m['measured_function'] for m in data['measurements']]
    measured = np.array([m['input_value'] for m in data['measurements']])
    propagate = create_propagate_fn(reac_map, exprs)

    # Run GLS
    result = iterative_gls(
        measured, covmat, propagate,
        param_vec.copy(), free_idx, max_iter=30, tol=1e-10)

    # Compute derived quantities
    derived = compute_derived_quantities(result, reac_map)

    # Collect all fitted values (primary + derived)
    free_keys = list(DEFAULT_INITIAL_PARAMS.keys())
    all_fitted = {}
    for i, k in enumerate(free_keys):
        val = result['free_values'][i]
        unc_pct = np.sqrt(result['cov_free'][i, i])
        all_fitted[k] = (val, unc_pct)
    for k, val in derived.items():
        all_fitted[k] = (val, None)  # uncertainty for derived not computed here

    # Print results
    print("=" * 78)
    print("BENCHMARK: Axton 1986 Table 4 — All data (167 measurements)")
    print("=" * 78)

    print(f"\n{'Metric':<25s} {'This work':>12s} {'Report':>12s} {'Match':>8s}")
    print("-" * 60)

    chi2_match = "OK" if abs(result['chi2'] - REPORT_CHI2) < 2 else "FAIL"
    print(f"{'Chi-squared':<25s} {result['chi2']:12.1f} {REPORT_CHI2:12d} {chi2_match:>8s}")
    print(f"{'N measurements':<25s} {len(measured):12d} {REPORT_N_MEAS:12d}")
    print(f"{'N free parameters':<25s} {len(free_idx):12d} {REPORT_N_PARAMS:12d}")
    print(f"{'Degrees of freedom':<25s} {result['dof']:12d} {REPORT_DOF:12d}")
    print(f"{'Converged':<25s} {str(result['converged']):>12s}")
    print(f"{'Iterations':<25s} {result['n_iter']:12d}")

    print(f"\n{'Parameter':<12s} {'Fitted':>10s} {'Report':>10s} "
          f"{'Diff%':>8s}  {'Unc%':>7s} {'Rep Unc%':>8s} {'Match':>6s}")
    print("-" * 72)

    n_pass = 0
    n_total = 0
    for k in TABLE4_REFERENCE:
        ref_val, ref_unc = TABLE4_REFERENCE[k]
        if k not in all_fitted:
            continue
        fit_val, fit_unc = all_fitted[k]
        diff_pct = (fit_val - ref_val) / ref_val * 100
        n_total += 1

        # Pass if within 0.1% for well-constrained, 1% for poorly constrained
        threshold = 0.1 if ref_unc < 1.0 else 0.5
        passed = abs(diff_pct) < threshold
        if passed:
            n_pass += 1
        status = "OK" if passed else "DIFF"

        name = f"{k[0]} {k[1]}"
        unc_str = f"{fit_unc:.3f}" if fit_unc is not None else "  -  "
        print(f"  {name:<10s} {fit_val:10.4f} {ref_val:10.4f} "
              f"{diff_pct:+8.4f}%  {unc_str:>7s} {ref_unc:8.3f} {status:>6s}")

    print("-" * 72)
    print(f"Parameters matching: {n_pass}/{n_total} "
          f"(threshold: 0.1% for unc<1%, 0.5% for unc>=1%)")
    print(f"Chi-squared: {result['chi2']:.1f} (report: {REPORT_CHI2})")

    return result, all_fitted


if __name__ == '__main__':
    run_benchmark()
