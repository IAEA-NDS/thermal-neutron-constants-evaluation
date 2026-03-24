"""Benchmark: Axton 1986 microscopic fit without Lounsbury alpha measurements.

Hypothesis: the report's Table 8 (101 measurements) excludes Lounsbury
alpha measurements (21-23) which bias FIS 33 downward. Including WGA/WGF
(54-61) and CAP 34 (5) as additional measurements gives 96-3+8+1=102,
close to the report's 101.

Expected results from the report:
  chi2 = 48, 101 measurements, 38 parameters, 63 dof
"""

import sys
import re
import json
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

from utils.gls import build_param_space, create_propagate_fn, iterative_gls
from utils.covmat import create_relative_covmat
from axton1986_defaults import DEFAULT_INITIAL_PARAMS, DEFAULT_FIXED_PARAMS


MAXWELLIAN_DERIVED_FUNCS = frozenset({
    'FA', 'FF', 'CAP', 'FFH',
    'F1ETA', 'F2ETA', 'F3ETA',
    'FLEM', 'F1BIG',
    'F1CAB', 'F2CAB', 'F3CAB', 'F4CAB', 'F5CAB',
    'GC116', 'GA116',
})

# Lounsbury alpha measurements — excluded based on analysis showing
# they bias FIS 33 downward through a high alpha(233U) value
LOUNSBURY_ALPHA = {21, 22, 23}

# Measurements excluded because their parameters are unconstrained
# without Maxwellian data and the Cabell special functions
PARAM_EXCLUDED = {157, 158}  # CA 40, CA 42


def is_maxwellian_derived(expr):
    funcs = set(re.findall(r'([A-Z][A-Z0-9]+)', expr))
    return any(f in MAXWELLIAN_DERIVED_FUNCS for f in funcs)


def get_measurement_ids(data):
    """Microscopic measurements: exclude Maxwellian derived, Lounsbury alpha,
    and measurements with unconstrained parameters. Include WGA, WGF, CAP 34."""
    excluded = LOUNSBURY_ALPHA | PARAM_EXCLUDED
    # CAP 34 (meas 5) is kept: for non-fissile U-234, CAP ≈ sigma_gamma
    kept_despite_maxwellian = {5}
    return [m['no'] for m in data['measurements']
            if (not is_maxwellian_derived(m['measured_function'])
                or m['no'] in kept_despite_maxwellian)
            and m['no'] not in excluded]


# =========================================================================
# Table 8 reference values from Axton 1986
# =========================================================================

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

REPORT_CHI2 = 48
REPORT_N_MEAS = 101
REPORT_N_PARAMS = 38
REPORT_DOF = 63


def compute_derived_quantities(result, reac_map):
    p = result['params']
    def get(q, n):
        return p[reac_map[(q, n)]]
    derived = {}
    for n in [33, 35, 39, 41]:
        derived[('CA', n)] = get('ABS', n) - get('FIS', n)
        derived[('ETA', n)] = get('NUB', n) * get('FIS', n) / get('ABS', n)
        derived[('ALPHA', n)] = derived[('CA', n)] / get('FIS', n)
        if ('WGA', n) in reac_map and ('WGF', n) in reac_map:
            FA = get('ABS', n) * get('WGA', n)
            FF = get('FIS', n) * get('WGF', n)
            derived[('F3ETA', n)] = get('NUB', n) * FF - FA
    return derived


def run_benchmark():
    data_path = str(
        __import__('pathlib').Path(__file__).resolve().parent.parent
        / 'axton1986' / 'axton1986_data.json'
    )
    with open(data_path) as f:
        data = json.load(f)

    meas_ids = get_measurement_ids(data)
    meas_by_no = {m['no']: m for m in data['measurements']}

    # Build covariance matrix for selected subset
    full_covmat, full_nos = create_relative_covmat(data)
    idx = [full_nos.index(no) for no in meas_ids]
    covmat = full_covmat[np.ix_(idx, idx)]
    measured = np.array([meas_by_no[no]['input_value'] for no in meas_ids])
    exprs = [meas_by_no[no]['measured_function'] for no in meas_ids]

    # Parameter space: exclude unconstrained params (GC116, GA116, CA 40/42)
    unconstrained = {
        ('GC116', 39), ('GC116', 40), ('GC116', 41), ('GC116', 42),
        ('GA116', 39), ('GA116', 41),
        ('CA', 40), ('CA', 42),
    }
    initial_params = {k: v for k, v in DEFAULT_INITIAL_PARAMS.items()
                      if k not in unconstrained}

    param_vec, reac_map, free_idx = build_param_space(
        initial_params, DEFAULT_FIXED_PARAMS)
    propagate = create_propagate_fn(reac_map, exprs)

    result = iterative_gls(
        measured, covmat, propagate,
        param_vec.copy(), free_idx, max_iter=30, tol=1e-10)

    derived = compute_derived_quantities(result, reac_map)

    free_keys = list(initial_params.keys())
    all_fitted = {}
    for i, k in enumerate(free_keys):
        val = result['free_values'][i]
        unc_pct = np.sqrt(result['cov_free'][i, i])
        all_fitted[k] = (val, unc_pct)
    for k, val in derived.items():
        all_fitted[k] = (val, None)

    # Print results
    print("=" * 78)
    print("BENCHMARK: Axton 1986 Table 8 — No Lounsbury alpha, with WGA/WGF/CAP34")
    print("=" * 78)

    print(f"\n{'Metric':<25s} {'This work':>12s} {'Report':>12s} {'Match':>8s}")
    print("-" * 60)

    chi2_match = "OK" if abs(result['chi2'] - REPORT_CHI2) < 5 else "CHECK"
    n_match = "OK" if len(measured) == REPORT_N_MEAS else "DIFF"
    p_match = "OK" if len(free_idx) == REPORT_N_PARAMS else "DIFF"
    print(f"{'Chi-squared':<25s} {result['chi2']:12.1f} {REPORT_CHI2:12d} {chi2_match:>8s}")
    print(f"{'N measurements':<25s} {len(measured):12d} {REPORT_N_MEAS:12d} {n_match:>8s}")
    print(f"{'N free parameters':<25s} {len(free_idx):12d} {REPORT_N_PARAMS:12d} {p_match:>8s}")
    print(f"{'Degrees of freedom':<25s} {result['dof']:12d} {REPORT_DOF:12d}")
    print(f"{'Converged':<25s} {str(result['converged']):>12s}")
    print(f"{'Iterations':<25s} {result['n_iter']:12d}")

    print(f"\n{'Parameter':<12s} {'Fitted':>10s} {'Report':>10s} "
          f"{'Diff%':>8s}  {'Unc%':>7s} {'Rep Unc%':>8s} {'Match':>6s}")
    print("-" * 72)

    n_pass = 0
    n_total = 0
    for k in TABLE8_REFERENCE:
        ref_val, ref_unc = TABLE8_REFERENCE[k]
        if k not in all_fitted:
            continue
        fit_val, fit_unc = all_fitted[k]
        diff_pct = (fit_val - ref_val) / ref_val * 100
        n_total += 1

        threshold = 0.5 if ref_unc < 1.0 else 2.0
        passed = abs(diff_pct) < threshold
        if passed:
            n_pass += 1
        status = "OK" if passed else "DIFF"

        name = f"{k[0]} {k[1]}"
        unc_str = f"{fit_unc:.3f}" if fit_unc is not None else "  -  "
        print(f"  {name:<10s} {fit_val:10.4f} {ref_val:10.4f} "
              f"{diff_pct:+8.4f}%  {unc_str:>7s} {ref_unc:8.3f} {status:>6s}")

    print("-" * 72)
    print(f"Parameters matching: {n_pass}/{n_total}")
    print(f"Chi-squared: {result['chi2']:.1f} (report: {REPORT_CHI2})")
    print(f"Measurements: {len(measured)} (report: {REPORT_N_MEAS})")
    print(f"Free params: {len(free_idx)} (report: {REPORT_N_PARAMS})")

    return result, all_fitted


if __name__ == '__main__':
    run_benchmark()
