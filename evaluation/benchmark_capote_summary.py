"""Summary benchmark for the Capote 2026-05-04 fit chain (steps 0-6).

Runs each step's benchmark in sequence, collects fitted values, uncertainties,
chi-squared, measurement counts, and parameter counts. Emits a single dict
saved as JSON so the report draft can read it without re-running anything.

Output: evaluation/capote_chain_summary.json
"""

import sys
import io
import json
import contextlib
import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))

import benchmark_capote_step0_baseline as s0
import benchmark_capote_step1_lounsbury as s1
import benchmark_capote_step2_adamchuk as s2
import benchmark_capote_step3_wallner as s3
import benchmark_capote_step4_sca_list as s4
import benchmark_capote_step5_transmission as s5
import benchmark_capote_step5b_double_sca_scr as s5b
import benchmark_capote_step6_all_sca as s6


STEP_MODULES = [
    ('step0_baseline',     s0,  'Axton T8 baseline (no Lounsbury)'),
    ('step1_lounsbury',    s1,  '+ Lounsbury alpha (21-23)'),
    ('step2_adamchuk',     s2,  '+ Adamchuk 1988 alpha-235U'),
    ('step3_wallner',      s3,  '+ Wallner 2014 U-235 capture'),
    ('step4_sca_list',     s4,  'Sears SCA list + Pu-241 estimate; delete Lemmel SCA'),
    ('step5_transmission', s5,  '+ transmission entries; 3 SCR->SCA reclassifications'),
    ('step6_all_sca',      s6,  'All ABS+SCR -> ABS+SCA; drop Lemmel SCA-SCR; SCR partially removed'),
    # Sensitivity check (placed at the end so the main per-step loop in the
    # report builder still iterates 0..6 by index)
    ('step5b_double_scascr', s5b, 'Step 5 with Lemmel SCA-SCR doubled (sensitivity test)'),
]


def to_float(x):
    if x is None:
        return None
    return float(x)


def serialize_fitted(all_fitted):
    out = {}
    for k, v in all_fitted.items():
        val, unc = v
        out[f"{k[0]} {k[1]}"] = {
            "value": to_float(val),
            "unc_pct": to_float(unc),
        }
    return out


def main():
    summary = {"steps": []}
    for tag, module, description in STEP_MODULES:
        print(f"\n{'#'*78}")
        print(f"# Running {tag}: {description}")
        print('#'*78)
        # Suppress the per-benchmark print output; we already validated each
        # individually. Just capture the returned dicts.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result, all_fitted = module.run_benchmark()
        summary["steps"].append({
            "tag": tag,
            "description": description,
            "chi2": float(result['chi2']),
            "n_meas": int(len(result.get('residuals', []) or buf.getvalue())) or None,
            "n_free_params": int(len(result['free_values'])),
            "dof": int(result['dof']),
            "converged": bool(result['converged']),
            "n_iter": int(result['n_iter']),
            "fitted": serialize_fitted(all_fitted),
        })
        print(f"  chi2={result['chi2']:.2f}  free={len(result['free_values'])}  "
              f"dof={result['dof']}  converged={result['converged']}")

    # Recover n_meas from the result dict properly
    for i, (tag, module, _) in enumerate(STEP_MODULES):
        # Re-extract n_meas via len(result.dof + n_free)
        s = summary["steps"][i]
        s["n_meas"] = s["dof"] + s["n_free_params"]

    out_path = (__import__('pathlib').Path(__file__).resolve().parent
                / "capote_chain_summary.json")
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n\nWrote summary: {out_path}")

    # Quick textual summary
    print(f"\n{'='*78}")
    print(f"{'Step':<22s} {'chi2':>7s} {'N_meas':>7s} {'N_free':>7s} {'DOF':>5s}")
    print('-' * 78)
    for s in summary["steps"]:
        print(f"  {s['tag']:<20s} {s['chi2']:7.2f} {s['n_meas']:>7d} "
              f"{s['n_free_params']:>7d} {s['dof']:>5d}")


if __name__ == '__main__':
    main()
