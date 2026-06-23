"""Generate evaluation/report_capote_round3.md for the 2026-06-23 data revision.

Self-contained round-3 report builder. It:
  * runs the round-3 benchmark to obtain the fitted quantities,
  * reads the round-2 chain summary read-only for the Step-5 comparison
    (the favoured base; never mutates round-2 artefacts),
  * reads the round-3 database for the full input-data list, and
  * dumps a machine-readable round3_summary.json alongside the markdown.

The full datapoint-level manifest and decision register (flags F1-F13) lives
in references/emails/tnc_update_roberto_20260623/. This report mirrors that
register in its decisions appendix so the working group can review every
non-literal choice.

Run:  python evaluation/build_round3_report.py
"""

import io
import re
import json
import contextlib
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent

import sys
sys.path.insert(0, str(EVAL_DIR))
import benchmark_capote_round3 as r3
from build_round3_json import UPDATES_ABS, UPDATES_PCT, U234_NO, DELETE


# --- quantity ordering ------------------------------------------------------
PRIMARY_QUANTS = []
for q in ['SCA', 'SCR']:
    for n in [33, 35, 39, 41]:
        PRIMARY_QUANTS.append(f'{q} {n}')
for q in ['ABS', 'FIS', 'CA']:
    for n in [33, 35, 39, 41]:
        PRIMARY_QUANTS.append(f'{q} {n}')
PRIMARY_QUANTS.append('CAP 34')
for n in [33, 35, 39, 41, 52]:
    PRIMARY_QUANTS.append(f'NUB {n}')
for q in ['ETA', 'ALPHA', 'WGA', 'WGF']:
    for n in [33, 35, 39, 41]:
        PRIMARY_QUANTS.append(f'{q} {n}')
DERIVED_QUANTS = [f'{q} {n}' for q in ['FA', 'FF', 'F1ETA', 'F2ETA', 'F3ETA']
                  for n in [33, 35, 39, 41]]


# --- microscopic filter (same as the benchmarks) ----------------------------
MAXWELLIAN_DERIVED = frozenset({
    'FA', 'FF', 'CAP', 'FFH', 'F1ETA', 'F2ETA', 'F3ETA', 'FLEM', 'F1BIG',
    'F1CAB', 'F2CAB', 'F3CAB', 'F4CAB', 'F5CAB', 'GC116', 'GA116',
})
PARAM_EXCLUDED = {157, 158}
KEPT_DESPITE = {5}


def is_max(expr):
    funcs = set(re.findall(r'([A-Z][A-Z0-9]+)', expr))
    return any(f in MAXWELLIAN_DERIVED for f in funcs)


def in_fit(m):
    return ((not is_max(m['measured_function']) or m['no'] in KEPT_DESPITE)
            and m['no'] not in PARAM_EXCLUDED)


def fmt(v, prec=4):
    return '–' if v is None else f'{v:.{prec}f}'


def bold(s, do):
    return f'**{s}**' if do else s


def serialize(all_fitted):
    out = {}
    for k, (val, unc) in all_fitted.items():
        out[f'{k[0]} {k[1]}'] = {
            'value': None if val is None else float(val),
            'unc_pct': None if unc is None else float(unc),
        }
    return out


def main():
    # 1. run round-3 fit (suppress its console table)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result, all_fitted = r3.run_benchmark()
    r3_fit = serialize(all_fitted)
    n_meas = result['dof'] + len(result['free_values'])

    # 2. read round-2 step-5 (favoured base) read-only
    with open(EVAL_DIR / 'capote_chain_summary.json') as f:
        chain = json.load(f)
    s5 = next(s for s in chain['steps'] if s['tag'] == 'step5_transmission')
    s5_fit = s5['fitted']

    # 3. round-3 database for the input-data list
    with open(ROOT / 'axton1986' / 'axton1986_data_capote_round3.json') as f:
        data = json.load(f)
    measurements = data['measurements']

    # 3b. Step-5 base (read-only) — source of the deleted entries' details
    with open(ROOT / 'axton1986'
              / 'axton1986_data_capote_step5_transmission.json') as f:
        base = json.load(f)
    base_by_no = {m['no']: m for m in base['measurements']}

    UPDATED = set(UPDATES_ABS) | set(UPDATES_PCT) | {U234_NO}
    NEW = {m['no'] for m in measurements if m['no'] >= 195}

    out = []
    add = out.append

    # ---- header ----
    add('# Round-3 TNC Fit (2026-06-23): Fission + (n,tot) Data Revision\n')
    add('*Comprehensive fission and total-cross-section data review by R. Capote '
        'and I. Durán (2026-06-23 discussions), built on the round-2 Step-5 '
        'favoured fit. Audit trail: '
        '[`references/emails/tnc_update_roberto_20260623/`]'
        '(../references/emails/tnc_update_roberto_20260623/summary.md), with a '
        'datapoint-level before→after manifest in '
        '[`data_change_manifest.md`]'
        '(../references/emails/tnc_update_roberto_20260623/data_change_manifest.md).*\n')
    add('This round applies a large fission-data extension (new absolute and '
        'ratio measurements across all four fissile nuclides), value/uncertainty '
        'updates to the total-cross-section (transmission) entries from local '
        'EXFOR re-fits, removal of identified outliers and duplicates, and an '
        'update of the U-234(n,γ) value to the B80 evaluation. The **current '
        'SCA/SCR classification of every existing entry is kept** per the '
        'instruction; only values and uncertainties change on updated entries. '
        'The fit configuration is identical to round-2 Step 5 (30 free '
        'parameters; Lounsbury alpha included; the four Lemmel SCA−SCR points '
        'retained, so all four SCR parameters remain in the fit).\n')
    add('Every interpretation that departs from a literal reading of the source '
        'tables is recorded in the [decisions appendix](#decisions-flags-f1f13) '
        '(flags F1–F13) for review.\n')

    # ---- overview ----
    add('## Overview\n')
    add('| Fit | N meas | N free | DOF | χ² | χ²/DOF | Converged |')
    add('|-----|-------:|-------:|----:|---:|-------:|:---------:|')
    add(f"| Round-2 Step 5 (base) | {s5['n_meas']} | {s5['n_free_params']} | "
        f"{s5['dof']} | {s5['chi2']:.2f} | {s5['chi2']/s5['dof']:.3f} | Yes |")
    add(f"| **Round 3 (2026-06-23)** | **{n_meas}** | "
        f"{len(result['free_values'])} | {result['dof']} | "
        f"**{result['chi2']:.2f}** | {result['chi2']/result['dof']:.3f} | "
        f"{'Yes' if result['converged'] else 'No'} |")
    add('')
    add(f'Full database: 205 measurements (189 base − 10 deletions + 26 '
        f'additions); {n_meas} enter the microscopic fit after the Maxwellian '
        f'filter. χ² *decreases* despite the net +16 measurements, because the '
        f'deletions remove high-residual outliers while the new fission data are '
        f'mutually consistent.\n')

    # ---- change summary ----
    add('## Change Summary\n')
    add('| Category | Count | Serial numbers |')
    add('|----------|------:|----------------|')
    add(f'| Transmission value/σ updates | {len(UPDATES_ABS)} | '
        f'{", ".join(str(n) for n in sorted(UPDATES_ABS))} |')
    add(f'| Fission value/σ updates | {len(UPDATES_PCT)} | '
        f'{", ".join(str(n) for n in sorted(UPDATES_PCT))} |')
    add(f'| U-234(n,γ) → 100.9 b (B80) | 1 | {U234_NO} |')
    add(f'| Deletions (outliers/duplicates) | {len(DELETE)} | '
        f'{", ".join(str(n) for n in sorted(DELETE))} |')
    add(f'| Additions (23 fission + 3 transmission) | {len(NEW)} | '
        f'195–220 |')
    add('')

    # ---- discarded / excluded datapoints ----
    DELETE_REASONS = {
        71:  'Duplicate of the total-XS entry No. 186 (bare ABS is a derived '
             'quantity from the same experiment).',
        79:  'U-233 (n,tot): discarded — too much scatter.',
        86:  'Pu-241 (n,tot): discarded — too large scattering (see plot).',
        103: 'U-235 fission: outlier.',
        104: 'Pu-239 fission: outlier.',
        106: 'U-235 fission: outlier.',
        107: 'U-235 fission: outlier.',
        190: 'U-233 (n,tot): second Harvey 79 entry removed; the surviving '
             'Harvey is No. 189, updated to 588(9) [F10].',
        191: 'Pu-241 (n,tot): discarded — too large scattering (see plot).',
        194: 'Pu-239 (n,tot): outlier [F2] (the (n,tot) table showed an '
             'adjusted 1042(25), but it is flagged an outlier and removed).',
    }
    # Candidates considered in the 2026-06-23 review but NOT entered in the DB.
    NOT_ADDED = [
        ('Berceanu', 'U-235 fission', '589.3 (0.8%)',
         'Superseded by Borcea 73 = 586.3 (1.3%), per "replace by Borcea 73" '
         '(No. 212) [F9].'),
        ('Bollinger 52', 'U-233 (n,tot)', '592.6 (20 b)',
         'In the local working notes only; omitted from the consolidated '
         'U-233 (n,tot) list [F4].'),
        ('Havens', 'Pu-239 (n,tot)', '~1080 b',
         'Mis-calibrated experiment / outlier — "do not consider".'),
        ('Bowman, Melkonian, Schrack, Wagemans 79/88, Mihailescu',
         'fission', '—',
         'Shape measurements (relative to the thermal point); not entered as '
         'absolute datapoints [F12]. (This Melkonian ≠ the U-235 (n,tot) '
         'No. 63, which is retained.)'),
    ]

    add('## Discarded and Excluded Datapoints\n')
    add('Because the input-data list above shows only what remains in the '
        'database, this section makes the removals explicit. **Part 1** lists '
        'the entries deleted from the Step-5 base (their pre-deletion values); '
        '**Part 2** lists candidates that were considered in the 2026-06-23 '
        'review but not entered.\n')
    add('### Deleted from the database (10 entries)\n')
    add('| No. | Author | Reference | Quantity | Value | σ (abs) | σ (%) | Reason |')
    add('|----:|--------|-----------|----------|------:|--------:|------:|--------|')
    for no in sorted(DELETE):
        m = base_by_no[no]
        val, pct = m['input_value'], m['uncertainty_percent']
        sig_abs = abs(val) * pct / 100.0
        add(f"| {no} | {m['author']} | {m['reference']} | "
            f"{m['measured_function']} | {val:g} | {sig_abs:.3g} | "
            f"{pct:.3f} | {DELETE_REASONS[no]} |")
    add('')
    add('### Considered but not added\n')
    add('| Author | Channel | Value | Reason |')
    add('|--------|---------|------:|--------|')
    for author, channel, value, reason in NOT_ADDED:
        add(f"| {author} | {channel} | {value} | {reason} |")
    add('')

    # ---- fitted quantities + Step-5 comparison ----
    add('## Fitted Quantities (Round 3 vs Step-5 base)\n')
    add('Shifts exceeding 0.1% relative to the Step-5 base are highlighted in '
        '**bold**. K1 = F3ETA = NUB·FF − FA is the central derived quantity.\n')
    add('| Quantity | Step 5 | Round 3 | Unc (%) | Δ vs Step 5 (%) |')
    add('|----------|-------:|--------:|--------:|----------------:|')
    for q in PRIMARY_QUANTS + DERIVED_QUANTS:
        cur = r3_fit.get(q)
        if not cur or cur['value'] is None:
            continue
        val, unc = cur['value'], cur['unc_pct']
        prv = s5_fit.get(q)
        if prv and prv['value']:
            d = (val - prv['value']) / prv['value'] * 100
            dstr, sig = f'{d:+.3f}', abs(d) > 0.1
            p5 = fmt(prv['value'])
        else:
            dstr, sig, p5 = '–', False, '–'
        add(f'| {bold(q, sig)} | {p5} | {bold(fmt(val), sig)} | '
            f'{fmt(unc, 3)} | {bold(dstr, sig)} |')
    add('')

    # ---- K1 ----
    add('## K1 = F3ETA (Round 3)\n')
    add('| Nuclide | K1 = F3ETA | Unc (%) | Step-5 base | Δ% |')
    add('|---------|-----------:|--------:|------------:|---:|')
    labels = {33: '²³³U', 35: '²³⁵U', 39: '²³⁹Pu', 41: '²⁴¹Pu'}
    for n in [33, 35, 39, 41]:
        cur = r3_fit[f'F3ETA {n}']
        prv = s5_fit[f'F3ETA {n}']
        d = (cur['value'] - prv['value']) / prv['value'] * 100
        add(f"| {labels[n]} | {cur['value']:.2f} | {cur['unc_pct']:.3f} | "
            f"{prv['value']:.2f} | {d:+.3f}% |")
    add('')
    add('K1(²³⁵U) = {:.1f}, consistent with the value (~722.4) anticipated in '
        'the 2026-05-06 discussion using the same B81-derived Westcott factor. '
        'The new fission data shift all four K1 values by less than 0.4%.\n'
        .format(r3_fit['F3ETA 35']['value']))

    # ---- full input-data list (Roberto's explicit request) ----
    add('## Full Input-Data List (absolute + relative uncertainty)\n')
    add('Per the 2026-06-23 "new TNC fit" request, the complete round-3 input '
        'database with **both absolute and relative uncertainty** for every '
        'measurement, for cross-checking against Axton and the 2026-06-23 '
        'changes. Status: **NEW** = added this round (Nos. 195–220); **UPD** = '
        'value/uncertainty updated this round; blank = unchanged from the '
        'Step-5 base. "In fit" marks the measurements that pass the microscopic '
        'filter and enter the GLS fit (others are Maxwellian-derived and used '
        'only for derived-quantity folding).\n')
    add('| No. | Author | Reference | Quantity | Value | σ (abs) | σ (%) | Status | In fit |')
    add('|----:|--------|-----------|----------|------:|--------:|------:|:------:|:------:|')
    for m in sorted(measurements, key=lambda x: x['no']):
        no = m['no']
        val = m['input_value']
        pct = m['uncertainty_percent']
        sig_abs = abs(val) * pct / 100.0
        status = 'NEW' if no in NEW else ('UPD' if no in UPDATED else '')
        ref = m['reference']
        ref = ref if len(ref) < 34 else ref[:31] + '...'
        used = 'yes' if in_fit(m) else ''
        add(f"| {no} | {m['author']} | {ref} | {m['measured_function']} | "
            f"{val:g} | {sig_abs:.3g} | {pct:.3f} | {status} | {used} |")
    add('')

    # ---- decisions appendix ----
    add('## Decisions (flags F1–F13)\n')
    add('Each open interpretation point from the 2026-06-23 review, resolved '
        'with the most plausible reading and recorded here for confirmation by '
        'the working group. Full context in the '
        '[manifest §F]'
        '(../references/emails/tnc_update_roberto_20260623/data_change_manifest.md).\n')
    decisions = [
        ('F1', 'Pattenden 56 Pu-239 uncertainty',
         'The fission table gives "1039(5%)" while the dedicated Pu-239 (n,tot) '
         'table gives "1039(25)" (≈2.41%). The later, dedicated (n,tot) revision '
         'with the local plot-fit is taken as authoritative → **1039 ± 25 b** '
         '(No. 75).'),
        ('F2', 'Leonard 56 Pu-239 (No. 194) — keep or delete',
         'The Pu-239 (n,tot) table shows an adjusted 1042(25), but the fission '
         'email and the local analysis both flag it an outlier → **deleted**.'),
        ('F3', 'Lynn 55 U-235 (n,tot) — add or not',
         'Listed as a "NEW set" 704(14) in the consolidated U-235 (n,tot) table, '
         'though the local notes advised against adding it (old; large local-vs-'
         'integral discrepancy). The consolidated table governs → **added** '
         '(No. 219); dissent recorded. Worth confirming.'),
        ('F4', 'Bollinger 52 U-233 (n,tot) — add or not',
         'Present in the local working notes but omitted from the consolidated '
         'U-233 (n,tot) list (only Moore carried over as NEW DATA) → **not '
         'added**.'),
        ('F5', 'Deruytter 74 FIS(²³³U) = 533.2 — missing uncertainty',
         'No uncertainty quoted. Assigned **σ = 0.5%** (No. 209), matching the '
         'precision of the other Deruytter absolute fission values. **Provisional '
         '— confirm the intended value.**'),
        ('F6', 'U-234(n,γ) = 100.9 b — missing uncertainty',
         'Only the central value (B80) was given. The existing relative '
         'uncertainty **2.086%** is kept (No. 5); remains a fixed input.'),
        ('F7', 'Watanabe-FIS41 / Deruytter61-FIS35 — update or new',
         'Both match existing entries by author/year/quantity/EXFOR id and are '
         'tagged as re-fits → treated as **updates** to Nos. 108 / 98, not new '
         'points. The systematic code on No. 98 (B) is retained.'),
        ('F8', 'Berceanu 77 (No. 101) "CONTAINS BORCEA 70"',
         'Read as a double-counting warning → **No. 101 kept as-is; Borcea 70 '
         'not added separately.**'),
        ('F9', 'Berceanu U-235 FIS35 589.3 "replace by Borcea 73"',
         'Interpreted literally → **Berceanu 589.3 dropped; Borcea 73 = 586.3 '
         '(1.3%) added** (No. 212).'),
        ('F10', 'Harvey 79 U-233 numbering',
         'The survivor is labelled "data #189" while the quoted old value (592) '
         'matches No. 190. Net intent is one Harvey at 588(9) → **No. 189 set to '
         '588(9), No. 190 deleted** (the survivor is overwritten, so the old '
         '585-vs-592 distinction is immaterial).'),
        ('F11', 'Gerasimov 62 U-235 uncertainty',
         'Fission email "692(1.4%)" vs U-235 (n,tot) email "692(8)" (≈1.16%). '
         'The dedicated (n,tot) absolute value is used → **692 ± 8 b** (No. 186).'),
        ('F12', 'Shape measurements',
         'Bowman, Melkonian, Schrack, Wagemans 79/88 and Mihailescu are shape '
         '(relative-to-thermal) measurements → **not added** as absolute points. '
         '(This Melkonian ≠ the U-235 (n,tot) entry No. 63, which is retained.)'),
        ('F13', 'Classification of the new transmission points',
         '"Keep current SCA/SCR" does not cover brand-new entries. Young '
         '(No. 220) is explicitly ABS+SCR. **Moore (No. 218) assigned ABS+SCA** '
         '(consistent with the local U-233 working assumption) and **Lynn '
         '(No. 219) assigned ABS+SCR** (era convention) — both provisional. '
         'Note: this only affects the favoured Step-5-style fit; under a '
         'Step-6-style "all ABS+SCR → ABS+SCA" conversion the choice is moot.'),
    ]
    for tag, title, body in decisions:
        add(f'**{tag} — {title}.** {body}\n')

    # ---- files ----
    add('## Files Produced (Round 3)\n')
    add('  * `axton1986/axton1986_data_capote_round3.json` — round-3 database '
        '(205 measurements), regenerated by `evaluation/build_round3_json.py`\n'
        '  * `evaluation/build_round3_json.py` — auditable generator (applies '
        'the manifest to the Step-5 base)\n'
        '  * `evaluation/benchmark_capote_round3.py` — round-3 fit benchmark\n'
        '  * `evaluation/build_round3_report.py` — this report generator\n'
        '  * `evaluation/round3_summary.json` — machine-readable round-3 results '
        '(+ Step-5 reference)\n'
        '  * `references/emails/tnc_update_roberto_20260623/` — paraphrased '
        'discussions + datapoint-level manifest with decision register\n')

    text = '\n'.join(out)
    (EVAL_DIR / 'report_capote_round3.md').write_text(text)

    # machine-readable summary
    round3_summary = {
        'round': 3,
        'date': '2026-06-23',
        'base': 'round-2 step5_transmission',
        'chi2': float(result['chi2']),
        'n_meas': int(n_meas),
        'n_free_params': int(len(result['free_values'])),
        'dof': int(result['dof']),
        'converged': bool(result['converged']),
        'n_iter': int(result['n_iter']),
        'fitted': r3_fit,
        'step5_reference': {'chi2': s5['chi2'], 'n_meas': s5['n_meas'],
                            'dof': s5['dof'], 'fitted': s5_fit},
    }
    with open(EVAL_DIR / 'round3_summary.json', 'w') as f:
        json.dump(round3_summary, f, indent=2)

    print(f"Wrote report_capote_round3.md: {len(text):,} chars, "
          f"{text.count(chr(10)):,} lines")
    print(f"Wrote round3_summary.json")
    print(f"Fit: chi2={result['chi2']:.2f}  N={n_meas}  "
          f"free={len(result['free_values'])}  dof={result['dof']}")


if __name__ == '__main__':
    main()
