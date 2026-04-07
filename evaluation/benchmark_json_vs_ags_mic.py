"""Benchmark: JSON vs AGS/thermalcst microscopic data comparison.

Compares the GLS fit using the JSON database (restricted to the 102
microscopic measurement IDs present in ags_axton1986_jan2026.mic /
thermalcst_axton1986_jan2026.mic) against the fit using the AGS +
thermalcst files directly.

Both pipelines use the same measurement expressions, the same forward
model, and the same parameter space. Differences arise from:
  - Small value rounding (JSON from PDF vs thermalcst from Gilles files)
  - Uncertainty differences (No. 147 uncorrelated component)
  - Column-shift in correlation codes affecting off-diagonal covariance

Reference values from Axton 1986 Table 8 (microscopic fit) are shown
alongside for context.
"""

import sys
import json
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

from utils.gls import (build_param_space, create_propagate_fn, iterative_gls,
                       get_measurement_exprs, compute_derived_uncertainties)
from utils.covmat import (create_relative_covmat,
                          create_relative_covmat_from_ags,
                          load_thermalcst_data)
from axton1986_defaults import DEFAULT_INITIAL_PARAMS, DEFAULT_FIXED_PARAMS


# =========================================================================
# Table 8 reference values from Axton 1986 (microscopic fit)
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
    ('NUB', 33): (2.4856, 0.218),
    ('NUB', 35): (2.4261, 0.188),
    ('NUB', 39): (2.8794, 0.207),
    ('NUB', 41): (2.9406, 0.220),
    ('NUB', 52): (3.7644, 0.132),
    ('WGA', 33): (0.9988, 0.120),
    ('WGA', 35): (0.9787, 0.092),
    ('WGA', 39): (1.0767, 0.269),
    ('WGA', 41): (1.0444, 0.191),
    ('WGF', 33): (0.9966, 0.201),
    ('WGF', 35): (0.9778, 0.092),
    ('WGF', 39): (1.0562, 0.275),
    ('WGF', 41): (1.0452, 0.670),
    ('CA', 33): (41.9697, 4.182),
    ('CA', 35): (96.1049, 1.811),
    ('CA', 39): (270.3632, 1.174),
    ('CA', 41): (361.6036, 1.711),
    ('ETA', 33): (2.3042, 0.271),
    ('ETA', 35): (2.0838, 0.255),
    ('ETA', 39): (2.1153, 0.311),
    ('ETA', 41): (2.1713, 0.399),
    ('ALPHA', 33): (0.0787, 4.494),
    ('ALPHA', 35): (0.1643, 1.962),
    ('ALPHA', 39): (0.3612, 1.280),
    ('ALPHA', 41): (0.3543, 1.668),
    ('F3ETA', 33): (746.4586, 0.683),
    ('F3ETA', 35): (721.2735, 0.507),
    ('F3ETA', 39): (1179.4821, 0.805),
    ('F3ETA', 41): (1693.1349, 1.774),
}


def derived_quantity_fn(params_tf, reac_map):
    """Compute derived quantities as TF operations for autodiff."""
    def get(q, n):
        return params_tf[reac_map[(q, n)]]
    derived = {}
    for n in [33, 35, 39, 41]:
        derived[('CA', n)] = get('ABS', n) - get('FIS', n)
        derived[('ETA', n)] = get('NUB', n) * get('FIS', n) / get('ABS', n)
        ca = get('ABS', n) - get('FIS', n)
        derived[('ALPHA', n)] = ca / get('FIS', n)
        FA = get('ABS', n) * get('WGA', n)
        FF = get('FIS', n) * get('WGF', n)
        derived[('F3ETA', n)] = get('NUB', n) * FF - FA
    return derived


def _run_json_pipeline(data, mic_nos):
    """Run GLS using JSON database, restricted to mic_nos."""
    meas_by_no = {m['no']: m for m in data['measurements']}

    full_covmat, full_nos = create_relative_covmat(data)
    idx = [full_nos.index(no) for no in mic_nos]
    covmat = full_covmat[np.ix_(idx, idx)]
    measured = np.array([meas_by_no[no]['input_value'] for no in mic_nos])
    exprs = [meas_by_no[no]['measured_function'] for no in mic_nos]

    # Mic-only: exclude parameters not constrained by microscopic data
    unconstrained = {
        ('GC116', 39), ('GC116', 40), ('GC116', 41), ('GC116', 42),
        ('GA116', 39), ('GA116', 41),
        ('CA', 40), ('CA', 42), ('CAP', 34),
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

    return result, all_fitted


def _run_ags_pipeline(mic_nos, exprs):
    """Run GLS using AGS covariance + thermalcst values for mic data."""
    base = str(
        __import__('pathlib').Path(__file__).resolve().parent.parent
        / 'axton1986' / 'Axton1986'
    )
    ags_mic = base + '/ags_axton1986_jan2026.mic'
    tc_mic = base + '/thermalcst_axton1986_jan2026.mic'

    # Load values from thermalcst
    tc_nos, tc_vals, tc_uncs = load_thermalcst_data(tc_mic)
    tc_by_no = {no: val for no, val in zip(tc_nos, tc_vals)}
    measured = np.array([tc_by_no[no] for no in mic_nos])

    # Load covariance from AGS
    ags_covmat, ags_nos = create_relative_covmat_from_ags(ags_mic)
    ags_idx_map = {no: i for i, no in enumerate(ags_nos)}
    idx = [ags_idx_map[no] for no in mic_nos]
    covmat_frac = ags_covmat[np.ix_(idx, idx)]
    # Convert from fraction^2 to percent^2
    covmat = covmat_frac * 1e4

    # Same parameter space as JSON pipeline
    unconstrained = {
        ('GC116', 39), ('GC116', 40), ('GC116', 41), ('GC116', 42),
        ('GA116', 39), ('GA116', 41),
        ('CA', 40), ('CA', 42), ('CAP', 34),
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

    return result, all_fitted


def _print_comparison(json_result, json_fitted, ags_result, ags_fitted):
    """Print side-by-side comparison of JSON and AGS fit results."""
    print("=" * 95)
    print("BENCHMARK: JSON vs AGS/thermalcst — microscopic data (102 measurements)")
    print("=" * 95)

    print(f"\n{'Metric':<25s} {'JSON':>12s} {'AGS':>12s} {'Report':>12s}")
    print("-" * 65)
    print(f"{'Chi-squared':<25s} {json_result['chi2']:12.1f} "
          f"{ags_result['chi2']:12.1f} {'48':>12s}")
    print(f"{'N measurements':<25s} {json_result['dof'] + len(json_result['free_values']):12d} "
          f"{ags_result['dof'] + len(ags_result['free_values']):12d} {'101':>12s}")
    print(f"{'N free parameters':<25s} {len(json_result['free_values']):12d} "
          f"{len(ags_result['free_values']):12d} {'38':>12s}")
    print(f"{'Degrees of freedom':<25s} {json_result['dof']:12d} "
          f"{ags_result['dof']:12d} {'63':>12s}")
    print(f"{'Converged':<25s} {str(json_result['converged']):>12s} "
          f"{str(ags_result['converged']):>12s}")
    print(f"{'Iterations':<25s} {json_result['n_iter']:12d} "
          f"{ags_result['n_iter']:12d}")

    print(f"\n{'Parameter':<12s} {'JSON':>10s} {'AGS':>10s} {'Report':>10s} "
          f"{'J-A diff%':>10s}  {'J unc%':>7s} {'A unc%':>7s} {'R unc%':>7s}")
    print("-" * 90)

    all_keys = list(TABLE8_REFERENCE.keys())
    # Add any fitted keys not in reference
    for k in sorted(set(json_fitted.keys()) | set(ags_fitted.keys())):
        if k not in all_keys:
            all_keys.append(k)

    n_match = 0
    n_total = 0
    for k in all_keys:
        j_val, j_unc = json_fitted.get(k, (None, None))
        a_val, a_unc = ags_fitted.get(k, (None, None))
        r_val, r_unc = TABLE8_REFERENCE.get(k, (None, None))

        if j_val is None and a_val is None:
            continue

        name = f"{k[0]} {k[1]}"

        if j_val is not None and a_val is not None:
            diff_pct = (j_val - a_val) / a_val * 100
            diff_str = f"{diff_pct:+10.4f}%"
            n_total += 1
            if abs(diff_pct) < 0.05:
                n_match += 1
        else:
            diff_str = "     -    "

        j_str = f"{j_val:10.4f}" if j_val is not None else "     -    "
        a_str = f"{a_val:10.4f}" if a_val is not None else "     -    "
        r_str = f"{r_val:10.4f}" if r_val is not None else "     -    "
        ju_str = f"{j_unc:.3f}" if j_unc is not None else "  -  "
        au_str = f"{a_unc:.3f}" if a_unc is not None else "  -  "
        ru_str = f"{r_unc:.3f}" if r_unc is not None else "  -  "

        print(f"  {name:<10s} {j_str} {a_str} {r_str} "
              f"{diff_str}  {ju_str:>7s} {au_str:>7s} {ru_str:>7s}")

    print("-" * 90)
    print(f"JSON vs AGS agreement: {n_match}/{n_total} parameters within 0.05%")
    print(f"Chi-squared: JSON={json_result['chi2']:.1f}, "
          f"AGS={ags_result['chi2']:.1f} (report: 48)")


def run_benchmark():
    base = str(
        __import__('pathlib').Path(__file__).resolve().parent.parent
        / 'axton1986'
    )
    data_path = base + '/axton1986_data.json'

    with open(data_path) as f:
        data = json.load(f)

    # Get mic measurement IDs from thermalcst file
    tc_mic = base + '/Axton1986/thermalcst_axton1986_jan2026.mic'
    mic_nos, _, _ = load_thermalcst_data(tc_mic)
    mic_nos = list(mic_nos)

    # Get expressions from JSON (both pipelines use the same forward model)
    meas_by_no = {m['no']: m for m in data['measurements']}
    exprs = [meas_by_no[no]['measured_function'] for no in mic_nos]

    # Run both pipelines
    json_result, json_fitted = _run_json_pipeline(data, mic_nos)
    ags_result, ags_fitted = _run_ags_pipeline(mic_nos, exprs)

    _print_comparison(json_result, json_fitted, ags_result, ags_fitted)

    return json_result, json_fitted, ags_result, ags_fitted


if __name__ == '__main__':
    run_benchmark()
