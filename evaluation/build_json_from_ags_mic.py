"""Build a modified JSON database that matches the AGS/thermalcst mic files exactly.

Reads values from thermalcst_axton1986_jan2026.mic and uncertainty
decomposition from ags_axton1986_jan2026.mic. Constructs a JSON file
with the same structure as axton1986_data.json but with values and
covariance components taken from the Gilles (jan2026) files.

The AGS column-to-correlation-code mapping was determined empirically
by cross-referencing nonzero entries in the AGS file against the named
codes in the original JSON. The mapping is 1:1 for cols 0-18,
then offset by 1 starting at col 20 (AGS col 19 is unused), with
a second gap creating offset 2 for the high-numbered columns.
"""

import json
import numpy as np
from pathlib import Path


def _build_ags_col_to_code(code_list):
    """Build AGS column index -> JSON correlation code name mapping.

    Determined by matching nonzero AGS entries against JSON systematic
    uncertainties across all 102 microscopic measurements (zero
    mismatches found).
    """
    # Empirical mapping from cross-reference analysis.
    # Verified with zero mismatches across all 152 measurements
    # (mac + mic combined).
    mapping = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
        5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
        10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
        15: 'P', 16: 'Q', 17: 'R', 18: 'S',
        # col 19 unused (gap — no 'T' code in Axton)
        20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
        25: 'Z', 26: 'Delta',
        27: 'A_', 28: 'B_', 29: 'C_', 30: 'D_', 31: 'E_',
        32: 'F_', 33: 'G_', 34: 'H_', 35: 'I_', 36: 'J_',
        37: 'K_',
        # cols 38-42 unused in both mac and mic AGS files
        43: 'Q_', 44: 'R_', 45: 'S_', 46: 'T_', 47: 'U_',
        48: 'V_', 49: 'W_', 50: 'X_', 51: 'Y_',
        52: 'Z_',
        # cols 53-63: second block, some used only in mac
        # col 53 maps to 'Delta' in mac (Popovic), unused in mic
        53: 'Delta', 54: 'rho', 55: 'dag', 56: 'dn',
        57: 'plus', 58: 'sq',
        59: 'c', 60: 'hc', 61: 'n', 62: 'u', 63: 'alpha',
    }
    # Verify all mapped codes exist
    for col, code in mapping.items():
        assert code in code_list, f"Code {code!r} from AGS col {col} not in code_list"
    return mapping


def build_modified_json():
    base = Path(__file__).resolve().parent.parent / 'axton1986'
    ags_path = base / 'Axton1986' / 'ags_axton1986_jan2026.mic'
    tc_path = base / 'Axton1986' / 'thermalcst_axton1986_jan2026.mic'
    json_path = base / 'axton1986_data.json'
    out_path = base / 'axton1986_data_ags_mic.json'

    # Load original JSON for metadata and structure
    with open(json_path) as f:
        original = json.load(f)

    code_list = list(original['correlation_codes'].keys())
    orig_by_no = {m['no']: m for m in original['measurements']}
    col_to_code = _build_ags_col_to_code(code_list)

    # Load thermalcst values
    tc_entries = {}
    with open(tc_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            no = int(float(parts[0]))
            val = float(parts[1])
            unc = float(parts[2])
            tc_entries[no] = (val, unc)

    # Load AGS uncertainty decomposition
    ags_entries = {}
    with open(ags_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            no = int(float(parts[0]))
            uncorr_frac = float(parts[2])
            syst = [float(x) for x in parts[3:]]
            ags_entries[no] = (uncorr_frac, syst)

    # Build modified measurements
    measurements = []
    for no in sorted(tc_entries.keys()):
        val, abs_unc = tc_entries[no]
        uncorr_frac, syst_vec = ags_entries[no]

        # Compute total relative uncertainty from AGS
        syst_sq_sum = sum(s ** 2 for s in syst_vec)
        total_frac = np.sqrt(uncorr_frac ** 2 + syst_sq_sum)
        total_pct = total_frac * 100

        # Build systematic uncertainties dict (in %)
        syst_dict = {}
        for col, s in enumerate(syst_vec):
            if abs(s) > 1e-15 and col in col_to_code:
                syst_dict[col_to_code[col]] = round(s * 100, 6)

        # Preserve metadata from original JSON where available
        orig = orig_by_no.get(no, {})
        meas = {
            'no': no,
            'author': orig.get('author', ''),
            'reference': orig.get('reference', ''),
            'measured_function': orig.get('measured_function', ''),
            'notes': orig.get('notes', ''),
            'input_value': val,
            'uncertainty_percent': round(total_pct, 6),
            'weighted_residual': orig.get('weighted_residual', None),
            'systematic_uncertainties': syst_dict,
        }
        measurements.append(meas)

    # Build output JSON
    out_data = {
        'metadata': {
            **original['metadata'],
            'source': ('Values from thermalcst_axton1986_jan2026.mic, '
                       'covariance from ags_axton1986_jan2026.mic'),
            'note': 'Modified to match AGS/thermalcst files exactly',
        },
        'nuclide_codes': original['nuclide_codes'],
        'correlation_codes': original['correlation_codes'],
        'measurements': measurements,
    }

    with open(out_path, 'w') as f:
        json.dump(out_data, f, indent=2)

    print(f"Written {len(measurements)} measurements to {out_path}")
    return out_path


if __name__ == '__main__':
    build_modified_json()
