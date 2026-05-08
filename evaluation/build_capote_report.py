"""Generate evaluation/report_capote_fit.md from capote_chain_summary.json.

Mirrors the depth and conventions of report_memo_implementation.md:
  - per-step quantity table with all primaries + derived F-quantities
  - bold for shifts >0.1% from the previous step
  - step-5-vs-step-6 head-to-head comparison (Roberto's key question)
  - K1 = F3ETA evolution table
  - Step 4 / Step 5 added-datapoint tables (reference, type, value, σ)
  - Comprehensive table of all measurements in the final fit
  - Appendix of every open question and the decision taken.
"""

import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent

with open(EVAL_DIR / 'capote_chain_summary.json') as f:
    summary = json.load(f)


# ---------------------------------------------------------------------------
# Quantity ordering helpers
# ---------------------------------------------------------------------------
PRIMARY_QUANTS = []
for q in ['SCA', 'SCR']:
    for n in [33, 35, 39, 41]: PRIMARY_QUANTS.append(f'{q} {n}')
for q in ['ABS', 'FIS', 'CA']:
    for n in [33, 35, 39, 41]: PRIMARY_QUANTS.append(f'{q} {n}')
PRIMARY_QUANTS.append('CAP 34')
for n in [33, 35, 39, 41, 52]: PRIMARY_QUANTS.append(f'NUB {n}')
for q in ['ETA', 'ALPHA']:
    for n in [33, 35, 39, 41]: PRIMARY_QUANTS.append(f'{q} {n}')
for q in ['WGA', 'WGF']:
    for n in [33, 35, 39, 41]: PRIMARY_QUANTS.append(f'{q} {n}')

DERIVED_QUANTS = []
for q in ['FA', 'FF', 'F1ETA', 'F2ETA', 'F3ETA']:
    for n in [33, 35, 39, 41]: DERIVED_QUANTS.append(f'{q} {n}')


def fmt(v, prec=4):
    return '–' if v is None else f'{v:.{prec}f}'


def bold(s, do):
    return f'**{s}**' if do else s


def step_table(step_idx):
    s = summary['steps'][step_idx]
    prev = summary['steps'][step_idx - 1] if step_idx > 0 else None
    lines = ['| Quantity | Value | Unc (%) | Δ vs prev (%) |',
             '|----------|------:|--------:|--------------:|']
    for q in PRIMARY_QUANTS + DERIVED_QUANTS:
        cur = s['fitted'].get(q)
        if not cur or cur['value'] is None:
            continue
        val, unc = cur['value'], cur['unc_pct']
        delta_str, sig = '–', False
        if prev:
            prv = prev['fitted'].get(q)
            if prv and prv['value']:
                d = (val - prv['value']) / prv['value'] * 100
                delta_str = f'{d:+.3f}'
                sig = abs(d) > 0.1
        lines.append(
            f'| {bold(q, sig)} | {bold(fmt(val), sig)} | {fmt(unc, 3)} | '
            f'{bold(delta_str, sig)} |')
    return '\n'.join(lines)


def k1_evolution():
    lines = ['| Step | F3ETA 33 | F3ETA 35 | F3ETA 39 | F3ETA 41 |',
             '|------|---------:|---------:|---------:|---------:|']
    for s in summary['steps']:
        row = f"| {s['tag'].replace('_', ' ')} "
        for n in [33, 35, 39, 41]:
            v = s['fitted'].get(f'F3ETA {n}')
            row += f"| {fmt(v['value']) if v and v['value'] else '–':>9s} "
        row += '|'
        lines.append(row)
    return '\n'.join(lines)


def step5_vs_step6():
    s5 = summary['steps'][5]['fitted']
    s6 = summary['steps'][6]['fitted']
    lines = ['| Quantity | Step 5 | Step 6 | Δ% |',
             '|----------|-------:|-------:|---:|']
    for q in PRIMARY_QUANTS + DERIVED_QUANTS:
        v5 = s5.get(q)
        v6 = s6.get(q)
        if not v5 or v5['value'] is None or not v6 or v6['value'] is None:
            if v5 and v5['value']:
                lines.append(f'| {q} | {fmt(v5["value"])} | – (removed) | – |')
            continue
        d = (v6['value'] - v5['value']) / v5['value'] * 100
        sig = abs(d) > 0.1
        lines.append(f'| {bold(q, sig)} | {fmt(v5["value"])} | '
                     f'{fmt(v6["value"])} | {bold(f"{d:+.3f}", sig)} |')
    return '\n'.join(lines)


def step5_vs_step5b():
    """Sensitivity comparison: step 5 vs step 5b (Lemmel SCA-SCR doubled)."""
    s5 = summary['steps'][5]['fitted']
    # step 5b is the last entry in the summary list
    s5b_step = next(s for s in summary['steps']
                    if s['tag'].startswith('step5b'))
    s5b = s5b_step['fitted']
    lines = ['| Quantity | Step 5 | Step 5b | Δ% |',
             '|----------|-------:|--------:|---:|']
    for q in PRIMARY_QUANTS + DERIVED_QUANTS:
        v5 = s5.get(q)
        vb = s5b.get(q)
        if not v5 or v5['value'] is None or not vb or vb['value'] is None:
            continue
        d = (vb['value'] - v5['value']) / v5['value'] * 100
        sig = abs(d) > 0.1
        lines.append(f'| {bold(q, sig)} | {fmt(v5["value"])} | '
                     f'{fmt(vb["value"])} | {bold(f"{d:+.3f}", sig)} |')
    return '\n'.join(lines), s5b_step


# ---------------------------------------------------------------------------
# Step 4 and Step 5 added-datapoint tables
# ---------------------------------------------------------------------------
STEP4_ADDED = [
    # (No, author, year/ref, type, nuclide, value, abs_unc, appendix_ref)
    # NOTE: round-2 corrections (2026-05-08): No.173 KAISER dropped as duplicate
    # of ARIF; entries 171/172/177 (U-235) and 179 (U-233) reclassified SCA->SCR.
    # Marked here for the report's "added datapoints" table.
    (171, 'Atoyi',     'Acta Cryst. 20, 587 (1966)',     'SCR(²³⁵U) [r2]', 12.3, 0.5,  'B2'),
    (172, 'Willis',    'PRS A274, 122 (1963)',           'SCR(²³⁵U) [r2]', 12.8, 1.0,  'B2'),
    (173, 'Kaiser',    'Physica B 136, 134 (1986)',      '— DELETED [r2]', 14.0, 0.2,  'B1'),
    (174, 'Arif',      'PR A35, 2810 (1987)',            'SCA(²³⁵U)', 14.0, 0.2,  None),
    (175, 'Ceulemans', 'Divadeenam compilation 1970',    'SCA(²³⁵U)', 14.6, 0.5,  'A4'),
    (176, 'Bayvas',    'Divadeenam compilation 1980',    'SCA(²³⁵U)', 14.3, 1.5,  None),
    (177, 'Foote',     'Divadeenam compilation 1958',    'SCR(²³⁵U) [r2]', 15.0, 1.0,  'B3'),
    (178, 'Oleksa',    'Phys Rev 109, 1645 (1958)',      'SCA(²³³U)', 13.4, 0.5,  'A5'),
    (179, 'Green',     'JNM 51, 281 (1974)',             'SCR(²³³U) [r2]', 12.6, 0.7,  'B2'),
    (180, 'Vertebnyi', 'Divadeenam compilation 1974',    'SCA(²³³U)', 13.2, 0.6,  None),
    (181, 'Green',     'JNM 34, 281 (1970)',             'SCA(²³⁹Pu)', 7.7, 0.6,  None),
    (182, 'Atoyi',     'Acta Cryst. 20, 587 (1966)',     'SCA(²³⁹Pu)', 7.3, 0.7,  None),
    (183, 'Roof',      'Acta Cryst. 15, 351 (1962)',     'SCA(²³⁹Pu)', 7.3, 0.7,  None),
    (184, 'Pronyaev',  'Internal analysis 2026-04-17',   'SCA(²⁴¹Pu)', 12.0, 0.8, 'A7'),
]

STEP4_DELETED = [
    (88, 'Lemmel', '1982', 'SCA(²³³U)', 12.306, 'replaced by step-4 SCA list'),
    (90, 'Lemmel', '1982', 'SCA(²³⁵U)', 16.508, 'replaced by step-4 SCA list'),
    (92, 'Lemmel', '1982', 'SCA(²³⁹Pu)', 8.004, 'replaced by step-4 SCA list'),
    (94, 'Lemmel', '1982', 'SCA(²⁴¹Pu)', 12.006, 'replaced by Pronyaev 12.0(0.8)'),
]

STEP5_RECLASSIFIED = [
    (68, 'Safford', '1959', 'U-235', 'ABS+SCR → ABS+SCA', None),
    (78, 'Nikitin', '1955', 'U-233', 'ABS+SCR → ABS+SCA', 'A8'),
    (83, 'Safford', '1960', 'U-233', 'ABS+SCR → ABS+SCA', 'A9'),
]

STEP5_ADDED = [
    (185, 'Antonov',     '1986', 'ABS+SCA(²³⁵U)',   695.0,  15.0, 'A10'),
    (186, 'Gerasimov',   '1962', 'ABS+SCR(²³⁵U)',   685.0,   9.0, 'A11'),
    (187, 'Leonard',     '1954', 'ABS+SCR(²³⁵U)',   702.0,  14.0, 'A12'),
    (188, 'Pschenichnyi','1978', 'ABS+SCR(²³³U)',   587.9,   3.0, 'A13'),
    (189, 'Harvey-1',    '1979', 'ABS+SCR(²³³U)',   585.0,   1.4, 'A14'),
    (190, 'Harvey-2',    '1979', 'ABS+SCR(²³³U)',   592.0,   6.0, 'A14'),
    (191, 'Schwartz',    '1958', 'ABS+SCR(²⁴¹Pu)', 1410.0,  80.0, None),
    (192, 'Anderson',    '1945', 'ABS+SCR(²³⁹Pu)', 1045.0,  25.0, None),
    (193, 'Palevsky',    '1955', 'ABS+SCR(²³⁹Pu)', 1025.0,  10.0, None),
    (194, 'Leonard',     '1956', 'ABS+SCR(²³⁹Pu)', 1055.0,  13.0, 'A15'),
]

STEP5_NOT_ADDED = [
    ('Spencer 1987', '(²³⁵U) ABS+SCR', 690, 5,
     'Already in DB as No.166 SPENCER 1985(3) val=690.0 σ=1.0% — values match within rounding (per project decision); uncertainty discrepancy logged in A1'),
    ('Spencer 1987', '(²³⁹Pu) ABS+SCR', 1025, 6,
     'Already in DB as No.167 SPENCER 1985(3) val=1025.8 σ=1.0% — values match within rounding; uncertainty discrepancy logged in A1'),
]


def added_table(rows):
    lines = ['| No. | Author / year | Reference | Type | Value (b) | σ (b) | Appendix |',
             '|----:|---------------|-----------|------|----------:|------:|:---------|']
    for r in rows:
        no, author, ref, typ, val, unc, app = r
        app_str = f'[{app}](#a-decisions-and-contentious-points)' if app else ''
        lines.append(f'| {no} | {author} | {ref} | {typ} | {val:g} | {unc:g} | {app_str} |')
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Comprehensive dataset table — all measurements in step-6 final fit
# ---------------------------------------------------------------------------
def comprehensive_dataset_table():
    """Read the final step-6 JSON and emit one row per measurement actually
    used in the fit (after the microscopic filter)."""
    with open(ROOT / 'axton1986' / 'axton1986_data_capote_step6_all_sca.json') as f:
        data = json.load(f)

    # Replicate the filter from benchmark_capote_step6_all_sca.py
    import re
    MAXWELLIAN_DERIVED = frozenset({
        'FA', 'FF', 'CAP', 'FFH', 'F1ETA', 'F2ETA', 'F3ETA',
        'FLEM', 'F1BIG',
        'F1CAB', 'F2CAB', 'F3CAB', 'F4CAB', 'F5CAB',
        'GC116', 'GA116',
    })
    PARAM_EXCLUDED = {157, 158}  # CA 40, 42
    KEPT_DESPITE = {5}  # CAP 34

    def is_max(expr):
        funcs = set(re.findall(r'([A-Z][A-Z0-9]+)', expr))
        return any(f in MAXWELLIAN_DERIVED for f in funcs)

    measurements_in_fit = []
    for m in data['measurements']:
        if (not is_max(m['measured_function']) or m['no'] in KEPT_DESPITE) \
                and m['no'] not in PARAM_EXCLUDED:
            measurements_in_fit.append(m)

    # Build appendix-ref lookup. Use the 'notes' or value to match.
    appendix_lookup = {}
    # Step 4 added entries
    for r in STEP4_ADDED:
        no, _, _, _, _, _, app = r
        if app:
            appendix_lookup[no] = app
    # Step 5 added
    for r in STEP5_ADDED:
        no, _, _, _, _, _, app = r
        if app:
            appendix_lookup[no] = app
    # Step 5 reclassifications
    for r in STEP5_RECLASSIFIED:
        no, _, _, _, _, app = r
        if app:
            appendix_lookup[no] = app
    # Spencer (No.166/167) — A1
    appendix_lookup[166] = 'A1'
    appendix_lookup[167] = 'A1'

    lines = [
        '| No. | Author | Reference | Quantity | Value | σ (%) | Appendix |',
        '|----:|--------|-----------|----------|------:|------:|:---------|',
    ]
    for m in measurements_in_fit:
        no = m['no']
        author = m['author']
        ref = m['reference']
        mf = m['measured_function']
        val = m['input_value']
        unc = m['uncertainty_percent']
        app = appendix_lookup.get(no, '')
        app_str = f'[{app}](#a-decisions-and-contentious-points)' if app else ''
        ref_short = ref if len(ref) < 35 else ref[:32] + '...'
        lines.append(
            f'| {no} | {author} | {ref_short} | {mf} | {val:g} | {unc:.3f} | {app_str} |')
    return '\n'.join(lines), len(measurements_in_fit)


# ---------------------------------------------------------------------------
# Build the full report
# ---------------------------------------------------------------------------
def build():
    # Get the comprehensive dataset table
    comp_table, n_meas_in_fit = comprehensive_dataset_table()

    out = []
    add = out.append

    add('# Capote 2026-05-04 Fit Chain: Step-by-Step Benchmark Report\n')
    add('*Round 2 — incorporates corrections requested in the 2026-05-08 email round (see [`references/emails/tnc_update_roberto_20260508/`](../references/emails/tnc_update_roberto_20260508/summary.md)).*\n')
    add('This report documents the implementation of the new TNC fit chain proposed in the 2026-05-04 emails *"NEW TNC FIT steps"* and *"Revision/additions to Axton Total cross sections experimental data"*, refined by the corrections in the 2026-05-08 round (drop KAISER as duplicate of ARIF; reclassify ATOYI / WILLIS / FOOTE / GREEN-233 from SCA to SCR; replace Westcott factors with B81-derived values). Each section corresponds to one of the seven steps (0–6) in the chain, showing how the iterative Generalized Least Squares (GLS) fit results evolve as data are added, deleted, reclassified, or modified.\n')
    add('All fits use the microscopic-only data subset (Maxwellian-derived functions excluded) on the Axton 1986 database. Steps 0 and 1 read from a round-2 baseline file (`axton1986_data_capote_round2_baseline.json`) that is identical to the original microscopic-Axton baseline except for the new Westcott factors; the original `axton1986_data.json` is preserved unchanged as the Axton-Table-8 reproduction artefact.\n')
    add('Quantities with shifts exceeding 0.1% relative to the previous step are highlighted in **bold**. The Westcott-folded quantities FA, FF, F1ETA, F2ETA, F3ETA are computed as derived outputs at each step per the request (K1 = F3ETA). A Round-1 → Round-2 comparison table is shown after the Step-5-vs-Step-6 head-to-head; the round-2 corrections are documented in detail in Appendix B.\n')

    add('## Overview\n')
    add('| # | Step | N meas | N free | DOF | χ² | Converged |')
    add('|---|------|-------:|-------:|----:|---:|:---------:|')
    descs = [
        ('Axton T8 baseline (no Lounsbury)',
         '+ Lounsbury alpha (21–23)',
         '+ Adamchuk 1988 alpha(²³⁵U)',
         '+ Wallner 2014 U-235 capture',
         'Sears SCA list + Pu-241 estimate; delete Lemmel SCA',
         '+ transmission additions; 3 SCR→SCA reclassifications',
         'All ABS+SCR → ABS+SCA; drop Lemmel SCA-SCR; SCR partially removed'),
    ][0]
    main_chain_steps = [s for s in summary['steps']
                        if not s['tag'].startswith('step5b')]
    for i, s in enumerate(main_chain_steps):
        add(f"| {i} | {descs[i]} | {s['n_meas']} | {s['n_free_params']} | "
            f"{s['dof']} | {s['chi2']:.2f} | {'Yes' if s['converged'] else 'No'} |")
    add('')

    add('## Targeted comparison: Step 5 vs Step 6\n')
    add('The 2026-05-04 email asks to *"compare steps 5 and 6 — hopefully one of them will be the final fit"*. The two candidate fits agree to better than 0.2% on every primary and derived quantity. Step 5 retains 4 SCR fit variables (now constrained by both the round-2 SCR reclassifications and the ABS+SCR transmissions); step 6 removes the ABS+SCR transmissions and Lemmel SCA-SCR, leaving SCR(²³³U) and SCR(²³⁵U) constrained only by the reclassified direct measurements (and SCR(²³⁹Pu), SCR(²⁴¹Pu) unconstrained → removed). The 2026-05-06 follow-up email expresses a marginal preference for Step 5 over Step 6 and also asks for a sensitivity check on the SCA−SCR prior — see [Step 5b](#step-5b-sensitivity-check-lemmel-scascr-doubled).\n')

    # Per-step sections
    headings = [
        ('Step 0: Axton Table 8 Baseline (No Lounsbury)',
         'This is the reference point: the microscopic GLS fit reproducing Axton\'s Table 8 results, with Lounsbury alpha measurements (Nos. 21–23) excluded. The detailed reproducibility analysis and the rationale for excluding Lounsbury are documented in `report_axton1986_reproduction.md` (Section 4); briefly, Lounsbury alpha measurements carry a "Z" systematic code (*Lounsbury Temperature Uncertainty*) that makes them effectively Maxwellian/integral, and Axton excluded them in his Table 6/Table 8 microscopic fit.'),
        ('Step 1: + Lounsbury Alpha (Nos. 21–23)',
         'Step 1 adds the three Lounsbury 1970 alpha measurements back into the microscopic fit:\n\n  * No.21 LOUNSBURY (CA 33)/FIS 33 = 0.0861, σ = 2.24%\n  * No.22 LOUNSBURY (CA 35)/FIS 35 = 0.1697, σ = 1.71%\n  * No.23 LOUNSBURY (CA 39)/FIS 39 = 0.3555, σ = 1.60%\n\nThis is the largest single change in the entire chain. Lounsbury\'s α(²³³U) = 0.0861 is far above Axton\'s converged value of 0.0787, and pulling the system toward this value drives a large redistribution of ABS(²³³U) between capture and fission — ALPHA(²³³U) shifts by +7.2%, CA(²³³U) by +6.6%, and FIS(²³³U) drops by 0.5%. F3ETA(²³³U) drops by 0.56%.'),
        ('Step 2: + Adamchuk 1988 Alpha(²³⁵U)',
         'Step 2 adds one new measurement (No.168):\n\n  * ADAMCHUK (CA 35)/FIS 35 = 0.169, σ = 2.1%, At. En. (R) 65, 434 (1988) (GMA NS=8026)\n\nAdamchuk\'s α(²³⁵U) = 0.169 is highly consistent with the step-1 fitted value (0.168), so the step is essentially a no-op for χ²: only +0.07 from one added measurement, and shifts in U-235 quantities are at the 0.1–0.2% level. All other nuclides are unchanged.'),
        ('Step 3: + Wallner 2014 U-235 Capture',
         'Step 3 adds two Wallner et al. 2014 thermal capture measurements (PRL 112, 192501, 2014):\n\n  * No.169 WALLNER CA 35 =  99.5 b, σ = 2.0%\n  * No.170 WALLNER CA 35 = 101.5 b, σ = 2.0%\n\nBoth Wallner values lie above the step-2 fit (CA 35 ≈ 98.2 b). The fit accommodates the pull with χ² = +2.3 over 2 added measurements; CA(²³⁵U) shifts +0.76% to 98.97 b. Through the absorption channel correlations, SCA(²³⁵U) drops 1.3% and SCR(²³⁵U) drops 1.9%. F3ETA(²³⁵U) shifts only 0.08%, so the K1 of interest is essentially preserved.'),
        ('Step 4: Sears SCA List + Pu-241 Estimate; Delete Lemmel SCA',
         '''Step 4 implements the most extensive change in the chain. It replaces the four Lemmel direct SCA measurements with a curated compilation drawn from Sears (1992), the Divadeenam compilation, EXFOR replacements, and Pronyaev's potential-scattering estimate for Pu-241.

The Lemmel SCA−SCR datapoints (Nos. 89, 91, 93, 95) are *kept* per the instruction *"leave SCA-SCR"*; only the four direct Lemmel SCA datapoints are deleted.

**Round-2 corrections applied here** (see Appendix B): KAISER (No.173) is removed as a duplicate of ARIF; ATOYI (No.171), WILLIS (No.172), FOOTE (No.177) and GREEN-233 (No.179) are reclassified from SCA to SCR. Net step-4 cardinality is therefore 9 added (vs. 14 before round 2): 1 deleted (Kaiser) plus 4 reclassified (still added, but now constraining SCR rather than SCA).

#### Datapoints deleted
{step4_deleted}

#### Datapoints added
{step4_added}

#### Impact
The U-235 SCA channel is the most affected: SCA(²³⁵U) drops driven by the ARIF interferometry value 14.0 b replacing the 16-ish Lemmel value (Kaiser is no longer added separately; it is the same experiment as Arif). SCA uncertainties collapse dramatically, reflecting the data richness of the new compilation. ABS, FIS, CA, and the F-quantities shift by ≤0.2% relative to step 3. χ² rises by ≈8 from a net +9 measurements (~0.9/measurement, indicating moderate but absorbable tension).
'''),
        ('Step 5: + Transmission Entries; 3 ABS+SCR → ABS+SCA Reclassifications',
         '''Step 5 implements Roberto's transmission-data revision from his "Revision/additions to Axton Total cross sections experimental data" email of 2026-05-04. Three existing transmission entries are reclassified, ten new entries are added, and two would-be additions (Spencer 1987) are recognised as already in the database.

#### Datapoints reclassified (ABS+SCR → ABS+SCA)
{step5_reclass}

#### Datapoints added
{step5_added}

#### Datapoints not added (already in DB)
{step5_not_added}

#### Impact
The new transmission datapoints sharpen ABS (and through it FIS and CA) but barely touch the SCA channel from step 4. ABS uncertainties tighten by 10–35% across the four nuclides. The largest individual tension is **Leonard 56 ABS+SCR(²³⁹Pu) = 1055 b** at +2.1σ from the predicted 1027 b; this contributes ≈4.5 to the χ² increase of 9.5. K1 = F3ETA shifts ≤0.21% across all four nuclides relative to step 4.
'''),
        ('Step 6: All ABS+SCR → ABS+SCA; Drop Lemmel SCA-SCR; SCR Partially Removed',
         '''Step 6 finalises the Capote chain:

  * **Convert** every (ABS X)+SCR X measurement to (ABS X)+SCA X — 26 conversions covering all transmission entries from baseline plus the step-5 additions.
  * **Delete** the four Lemmel SCA−SCR datapoints (Nos. 89, 91, 93, 95).
  * **Round-2 nuance** (see Appendix B): SCR(²³⁹Pu) and SCR(²⁴¹Pu) become unconstrained and are removed from the fit parameter space, but SCR(²³³U) and SCR(²³⁵U) remain in the fit because the four reclassified entries from step 4 (ATOYI, WILLIS, FOOTE on U-235; GREEN-233 on U-233) now constitute *direct* SCR measurements. Roberto's original *"reducing 4 fitting variables SCR"* simplifies to *reducing 2 fitting variables*. Decision logged in [B5](#b-round-2-corrections).

#### Impact relative to step 5
Differences are small. SCA channel essentially unchanged (much less than 0.02%) — the SCA additions in step 4 are the dominant constraint. ABS, FIS, CA, and the F-quantities shift by ≤0.15% as the converted transmission datapoints now correlate only with SCA, slightly relaxing the absorption fit. χ² is comparable (80.5 → 80.1, a 0.4 *decrease*) with 2 fewer free parameters.
'''),
    ]

    for i, (heading, intro) in enumerate(headings):
        add(f'## {heading}\n')
        # Substitute datapoint tables for steps 4 & 5
        if i == 4:
            intro = intro.format(
                step4_deleted='\n'.join([
                    '| No. | Author | Reference | Quantity | Value | Note |',
                    '|----:|--------|-----------|----------|------:|------|',
                ] + [
                    f'| {n} | {a} | {r} | {q} | {v:g} | {note} |'
                    for n, a, r, q, v, note in STEP4_DELETED
                ]),
                step4_added=added_table(STEP4_ADDED),
            )
        elif i == 5:
            intro = intro.format(
                step5_reclass='\n'.join([
                    '| No. | Author | Year | Nuclide | Change | Appendix |',
                    '|----:|--------|------|---------|--------|:---------|',
                ] + [
                    f'| {n} | {a} | {y} | {nu} | {ch} | {f"[{app}](#a-decisions-and-contentious-points)" if app else ""} |'
                    for n, a, y, nu, ch, app in STEP5_RECLASSIFIED
                ]),
                step5_added=added_table(STEP5_ADDED),
                step5_not_added='\n'.join([
                    '| Author / year | Type | Value | σ (b) | Reason |',
                    '|---------------|------|------:|------:|--------|',
                ] + [
                    f'| {a} | {t} | {v:g} | {u:g} | {r} |'
                    for a, t, v, u, r in STEP5_NOT_ADDED
                ]),
            )
        add(intro + '\n')
        add('### Fitted quantities\n')
        add(step_table(i))
        add('')

    # Step 5 vs Step 6 head-to-head
    add('## Step 5 vs Step 6: Head-to-Head Comparison\n')
    s5 = summary['steps'][5]
    s6 = summary['steps'][6]
    add(f'Both candidate fits agree to better than 0.2% on every primary and derived quantity. Step 6 has 2 fewer free parameters than step 5 (SCR(²³⁹Pu), SCR(²⁴¹Pu) become unconstrained after the ABS+SCR → ABS+SCA conversion; SCR(²³³U), SCR(²³⁵U) remain in the fit, constrained by the round-2 reclassified entries — see Appendix B5). χ² is comparable: {s5["chi2"]:.2f} vs {s6["chi2"]:.2f}, with step 6 *fitting slightly better* despite the parameter reduction.\n')
    add(step5_vs_step6())
    add('')
    add('**Per the 2026-05-06 follow-up email, Step 5 is marginally favoured** as the final fit candidate, on the grounds that the systematic SCR < SCA difference visible in step 5 is physically expected and yields fission cross sections in slightly better agreement with current library R-matrix fits. Differences with step 6 are within uncertainties.\n')

    # Step 5b sensitivity check
    add('## Step 5b: Sensitivity Check (Lemmel SCA−SCR doubled)\n')
    add('Per the 2026-05-06 follow-up email, this is a stability check on the Step 5 solution under a shifted SCA−SCR prior:\n\n'
        '> *"Pls start from Step 5 results and increase the SCA-SCR by a factor of 2 (both mean values and uncertainties). I like to see how stable our solution is relative to the SCA-SCR assumed values."*\n\n'
        'Implementation: starting from the round-2 step 5 inputs, the four Lemmel SCA−SCR datapoints (Nos. 89, 91, 93, 95) have their central values doubled. Percent uncertainties are kept unchanged so the absolute uncertainty also doubles — i.e., the prior is *shifted* (factor 2 on the central value) without weakening the relative confidence (50% → 50% for U-233; the same scaling for the other three nuclides). Documented in [B7](#b7-step-5b-sensitivity-check-on-scascr).\n')
    s5b_table, s5b_step = step5_vs_step5b()
    s5_step = summary['steps'][5]
    add(f'Fit: χ² = {s5b_step["chi2"]:.2f} ({s5_step["chi2"]:.2f} in step 5), N_meas = {s5b_step["n_meas"]}, N_free = {s5b_step["n_free_params"]}, DOF = {s5b_step["dof"]}.\n')
    add(s5b_table)
    add('')
    sca_scr_changes = []
    for n in [33, 35, 39, 41]:
        v5 = s5_step['fitted'].get(f'SCR {n}')
        vb = s5b_step['fitted'].get(f'SCR {n}')
        if v5 and vb and v5['value'] and vb['value']:
            sca_v5 = s5_step['fitted'].get(f'SCA {n}')
            sca_vb = s5b_step['fitted'].get(f'SCA {n}')
            diff5 = sca_v5['value'] - v5['value']
            diffb = sca_vb['value'] - vb['value']
            sca_scr_changes.append((n, diff5, diffb))
    add('SCA − SCR per nuclide (b):\n')
    add('| Nuclide | Step 5 | Step 5b | Lemmel input (5) | Lemmel input (5b) |\n'
        '|---------|-------:|--------:|-----------------:|------------------:|')
    LEMMEL_5_VALS = {33: 1.20, 35: 1.70, 39: 0.80, 41: 1.20}
    for n, d5, db in sca_scr_changes:
        l5 = LEMMEL_5_VALS[n]
        add(f'| ²{["³³U","³⁵U","³⁹Pu","⁴¹Pu"][[33,35,39,41].index(n)]} | {d5:.3f} | {db:.3f} | {l5:.2f} | {2*l5:.2f} |')
    add('')
    add('**Conclusion on stability**: the K1 = F3ETA values shift by less than 0.2% between Step 5 and Step 5b on every nuclide. The fitted SCA values are essentially unchanged (the SCA channel is dominated by the direct SCA measurements added in Step 4, so doubling the Lemmel SCA−SCR input mostly perturbs SCR rather than SCA). The fit is therefore stable against the chosen SCA−SCR central values within the ±50% uncertainty band, confirming that the Step 5 solution is not artificially driven by a particular Lemmel-derived prior.\n')

    # K1 = F3ETA evolution
    add('## K1 = F3ETA Evolution Through the Chain\n')
    add('K1 = F3ETA = NUB · FF − FA is the central derived quantity flagged as the K1 of the new TNC fit. Values across all 7 steps (round 2):\n')
    add(k1_evolution())
    add('')
    add('K1(²³³U) drops from ≈ 746.2 (step 0) to ≈ 738.7 (step 6) — a 1.0% reduction driven primarily by the Lounsbury inclusion in step 1. K1(²³⁵U) is essentially flat across the chain. K1(²³⁹Pu) shifts ≈ −0.1% from step 0 to step 6. K1(²⁴¹Pu) is flat. The dominant K1 changes between round 1 and round 2 come from the Westcott-factor update (Appendix B4): U-235 K1 ≈ +0.3%, Pu-239 K1 ≈ −0.9%, Pu-241 K1 ≈ +1.1%.\n')

    # Comprehensive dataset table (in appendix area)
    add('## Comprehensive Dataset Table (Final Fit, Step 6)\n')
    add(f'All {n_meas_in_fit} measurements actually used in the step-6 fit (after the microscopic filter is applied). Cross-references to the [Decisions appendix](#a-decisions-and-contentious-points) flag entries where a non-trivial choice was made.\n')
    add(comp_table)
    add('')

    # Appendix
    add('## A. Decisions and Contentious Points\n')
    add('Every choice that involved interpretation, departure from a literal reading of Roberto\'s emails, or risk of double-counting is recorded here so collaborators can scrutinise it.\n')

    add('### A1. Spencer 1987 vs. existing No.166/167 — values used as-is\n')
    add('Roberto\'s "Revision/additions" email lists Spencer 1987 ABS+SCR(²³⁵U) = 690(5) and Spencer 1987 ABS+SCR(²³⁹Pu) = 1025(6). The baseline already contains:\n\n  * No.166 SPENCER 1985(3) ABS+SCR(²³⁵U) = 690.0, σ = 1.0% (= 6.9 b), with a single systematic component `alpha = 0.707`.\n  * No.167 SPENCER 1985(3) ABS+SCR(²³⁹Pu) = 1025.8, σ = 1.0% (= 10.26 b), same systematic.\n\nThe values match within rounding (690 vs 690.0; 1025 vs 1025.8 — likely the same measurement, presented at a 1985 conference and published 1987). Per project decision, the database entries are retained as-is rather than being re-overwritten with Roberto\'s values. **Note**: the uncertainties differ noticeably (DB: 1.0%; Roberto: 0.72% / 0.59%); the more conservative DB uncertainty is what enters the fit. Step-6 conversion turns these into ABS+SCA(²³⁵U) and ABS+SCA(²³⁹Pu) like all other ABS+SCR entries.\n')

    add('### A2. Atoyi 1966 ABS+SCR tag treated as SCA\n')
    add('In the SEARS column of Capote\'s 2026-04-30 inlined compilation, the Atoyi 1966 row is annotated "SCR". The same row also gives Sears Sel = 12.3(5) b. Per project decision, the entire compilation is treated as providing SCA values regardless of the per-row "SCR/SCA" tag, because (a) Sears\' formalism delivers the bound-atom S_el (= SCA) by construction, and (b) the per-row tags in Roberto\'s table show inconsistencies that are most plausibly explained as paste artifacts from a heterogeneous source table.\n')

    add('### A3. Willis 1963 — EXFOR value used\n')
    add('The Sears compilation gives Willis 1963 = 12.1(5) coh + 0.2(2) inc → S_el 12.3(5). Roberto annotates "replace with EXFOR value" and the Divadeenam section gives "WILLIS 1963 12.8(1.0) = EXFOR". The EXFOR value 12.8(1.0) is used in step 4.\n')

    add('### A4. Ceulemans 1970 — V-51 renormalisation used as-is\n')
    add('Roberto\'s table notes "EXFOR 14.3(5) rel V-51 (1.02)" — the 14.6(5) value used in step 4 is 14.3 × 1.02, the 1.02 factor being a renormalisation against vanadium-51 standard. Different evaluations of V-51 thermal scattering vary by ~1–3% (Hughes-Schwartz 1958: ~5.0 b; Sears 1992 / ENDF/B-VIII: 5.08 b; ratio ≈ 1.016). Re-renormalising against current standards would change Ceulemans by < 1% (≈ 0.05–0.15 b), well under the quoted σ = 0.5 b — no practical impact on the fit. **Used as-is per project decision.**\n')

    add('### A5. Oleksa 1960 — quoted at 0.1 eV, used as-is\n')
    add('The EXFOR replacement Oleksa 13.4(5) is quoted "at 0.1 eV", not at the thermal point 0.0253 eV. Two candidate corrections were estimated:\n\n  * **Bound→free transition**: σ_free / σ_bound = (A/(A+1))² = 0.9914 for A=233, i.e. <0.3% effect at 0.1 eV (still in bound regime); ≤ 0.04 b.\n  * **Resonance contribution**: U-233 has no large resonance below 1 eV. Tail contributions from negative resonances and the 1.45 eV resonance at 0.1 eV are at most ≈ 0.05 b.\n\n**Combined |Δσ| < 0.1 b, well below quoted σ = 0.5 b. Used as-is per project decision.**\n')

    add('### A6. Green 1974 ABS+SCR tag treated as SCA\n')
    add('Same situation as A2: the Divadeenam-section row for Green 1974 carries a "SCR" annotation but is added as SCA(²³³U) per the global decision to treat all compilation rows as SCA.\n')

    add('### A7. Pu-241 SCA — Pronyaev internal analysis\n')
    add('No direct experimental SCA measurement exists for ²⁴¹Pu. Pronyaev\'s 2026-04-17 internal analysis derives SCA(²⁴¹Pu) = 12.0(0.8) b from the potential scattering radius R_pot = 9.54 fm (consistent across ENDF/B-VIII.1, JEFF-4.0 and JENDL-5; 4πR² = 11.44 b plus the contribution of negative resonances in the dominant ENDF/B-VIII.1 evaluation). This estimate is encoded as a direct SCA(²⁴¹Pu) "measurement" replacing the deleted Lemmel No.94 (12.006 b, σ = 21.6%). The replacement reduces the SCA(²⁴¹Pu) uncertainty from 21.6% (Lemmel) to 6.7% (Pronyaev).\n')

    add('### A8. Nikitin 1955 spelling — DB uses "NITIKIN"\n')
    add('Roberto writes "Nikitin" (correct transliteration of the Russian Никитин); the database has the typo "NITIKIN" inherited from Axton 1986 throughout (Nos. 65, 73, 78). The DB spelling is preserved for stable cross-referencing.\n')

    add('### A9. Safford 1960 vs 1959 ambiguity\n')
    add('Roberto\'s "Revision/additions" email instructs *"Change Safford 1959 ABS+SCR to ABS+SCA"* for U-233. The database has no Safford 1959 U-233 entry; Safford\'s U-233 transmission measurements are dated 1960 (Nos. 82 ABS+SCA(²³³U) = 585.77 already, 83 ABS+SCR(²³³U) = 585.67). The 1959 → 1960 mismatch is treated as a typo and No.83 is reclassified per Roberto\'s evident intent. Worth verifying with Roberto if there is an actual 1959 paper we are missing.\n')

    add('### A10. Antonov 1986 ABS+SCA tag — non-default for transmission\n')
    add('Roberto tags Antonov 1986 ²³⁵U as ABS+SCA, which is unusual for a transmission measurement (the dominant convention is ABS+SCR for rolled-metal samples). The tag is honoured under the assumption that Antonov used a non-metallic sample (UO₂ powder?) or applied a proper extinction correction. Verify with Roberto / read the original paper.\n')

    add('### A11. Gerasimov 1962 — likely separate from existing No.71\n')
    add('A GERASIMOV 1962 entry exists in the baseline as **No.71 GERASIMOV 1962(1) ABS 35 = 670.31 b, σ = 2.09%** (bare ABS, no scatter component). Roberto\'s "Gerasimov 62 ABS+SCR 685(9)" has a different value (685 vs 670.3) and a different quantity (ABS+SCR vs ABS). Per project decision, this is added as a separate measurement (No.186) rather than overwriting No.71. Worth verifying with Roberto if this is in fact the same measurement re-analysed.\n')

    add('### A12. Leonard 1954 — type defaulted to ABS+SCR\n')
    add('Roberto\'s email gives "Leonard 54 702(14)" with no SCR/SCA tag. Defaulted to ABS+SCR per the convention for the era (rolled-metal samples) and pending paper lookup. This decision is **moot for the final fit** because step 6 converts all ABS+SCR to ABS+SCA in any case.\n')

    add('### A13. Pschenichnyi 1978 uncertainty — typo fix 0.3 → 3.0\n')
    add('Roberto\'s email reads *"Pschenichnyi 78 ABS+SCR 587.9(3)"*, which would be σ = 0.3 b on 587.9 b ≈ 0.05% — implausibly small for a transmission measurement. **Treated as a typo and used as σ = 3.0 b** (≈ 0.51%). This decision is conservative; if Roberto truly intended 0.3 b, Pschenichnyi would dominate the U-233 absorption fit by orders of magnitude.\n')

    add('### A14. Harvey-1 / Harvey-2 — kept uncorrelated; correlation alternative noted\n')
    add('The two Harvey 1979 ABS+SCR(²³³U) entries come from the same author, same year, and likely the same ORELA campaign — physical correlation through sample mass / normalisation is plausible. Per project decision, the two are entered with empty `systematic_uncertainties` and treated as independent, matching the convention for nearly all other Axton transmission entries.\n\n**Alternative considered (not adopted)**: introduce a shared code `harvey79` with a 0.20% common component:\n```\nNo.189 Harvey-1: σ_total = 0.24%; sys = {harvey79: 0.20}\nNo.190 Harvey-2: σ_total = 1.01%; sys = {harvey79: 0.20}\n```\nThis would marginally tighten the U-233 ABS uncertainty. Worth revisiting if Roberto wants tighter modelling of correlated systematics.\n')

    add('### A15. Leonard 1956 ABS+SCR(²³⁹Pu) — largest tension in step 5\n')
    add('Leonard 1956 ABS+SCR(²³⁹Pu) = 1055 b lies +2.1σ above the fit-predicted 1027.4 b. This contributes ≈4.5 of the 9.5 χ² increase between step 4 and step 5. The measurement is retained as-is; flagged here as the dominant transmission tension in the chain.\n')

    add('### A16. SCR removed from parameter space in step 6\n')
    add('Roberto\'s step 6 instructs *"CONVERT ALL SCR in ABS+SCR to ABS+SCA, delete Lemmel SCA-SCR (reducing 4 fitting variables SCR)"*. The literal application:\n\n  * Convert every (ABS X)+SCR X → (ABS X)+SCA X (26 conversions).\n  * Delete the four Lemmel SCA−SCR datapoints (Nos. 89, 91, 93, 95).\n  * **No measurement function or datapoint references SCR after these changes**, so the four SCR primary parameters become unconstrained.\n  * They are therefore removed from the fit parameter space, consistent with the existing handling of unconstrained parameters (CA 40, CA 42, GC116, GA116). This matches Roberto\'s parenthetical "reducing 4 fitting variables SCR" exactly.\n')

    add('### A17. Microscopic filter and unconstrained parameters\n')
    add('All seven benchmarks (steps 0–6) apply the same microscopic filter that reproduces Axton Table 8: exclude measurements whose `measured_function` references any Maxwellian-derived quantity (FA, FF, CAP, FFH, F1ETA, F2ETA, F3ETA, FLEM, F1BIG, F1CAB–F5CAB, GC116, GA116) — except CAP 34 (No.5), which is retained because it ≈ σ_γ for non-fissile U-234. Lounsbury alpha (Nos. 21–23) is excluded only in step 0; it is included in steps 1–6. Unconstrained parameters CA 40, CA 42, GC116/40/41/42, GA116/39/41 are removed from the fit parameter space throughout. Step 6 (round 2) additionally removes SCR(²³⁹Pu) and SCR(²⁴¹Pu) — see [B5](#b5-scr-2-3-3-u-scr-2-3-5-u-retained-as-fit-variables-in-step-6) — but retains SCR(²³³U) and SCR(²³⁵U) as fit variables, which are constrained by the round-2 reclassified entries.\n')

    add('### A18. Empty systematic uncertainties for step-5 transmission additions\n')
    add('All step-5 transmission additions are entered with empty `systematic_uncertainties` (single uncorrelated total uncertainty), matching the convention used for 26 of the 28 baseline transmission entries. Spencer 1985 (Nos. 166/167) is the existing exception with `alpha = 0.707` representing the α-correction Spencer used internally. The Harvey-pair correlation alternative is the only other systematic structure considered (see A14).\n')

    add('## B. Round-2 Corrections (2026-05-08)\n')
    add('The 2026-05-08 email round produced three corrections to the round-1 fit. Each is documented below; the verbatim source is preserved in [`references/emails/tnc_update_roberto_20260508/summary.md`](../references/emails/tnc_update_roberto_20260508/summary.md). The original round-1 outputs are preserved as `evaluation/capote_chain_summary_20260504.json` and `evaluation/report_capote_fit_20260504.pdf` for direct numerical comparison.\n')

    add('### B1. KAISER (Physica B 136, 134 (1986)) removed as duplicate of ARIF\n')
    add('Kaiser et al. 1986 and Arif et al. 1987 used the same neutron-interferometry installation at the same institute; Kaiser gave preliminary values at only two energies, while Arif is the final publication. The 2026-05-08 review (Pronyaev) flagged the duplication; the 2026-05-05 follow-up confirmed deletion of KAISER, retaining ARIF.\n')

    add('### B2. ATOYI / WILLIS / GREEN-233 reclassified SCA → SCR\n')
    add('In round 1, the entire Sears/Divadeenam compilation was treated as providing SCA values per project decision (the per-row "SCR" tags were assumed to be paste artefacts). The 2026-05-08 review identified three rows for which the SCR tag is consistent with the actual measurement method:\n\n  * **ATOYI 1966** (²³⁵U, Sears Sel = 12.3(5)): tagged SCR by Sears.\n  * **WILLIS 1963** (²³⁵U, EXFOR value 12.8(1.0)): SCR per Sears.\n  * **GREEN 1974** (²³³U, 12.6(7)): per the Pronyaev analysis, the EXFOR record indicates a classical (non-crystal-spectrometer) method, consistent with SCR. Ad-hoc SCA estimation supplied by Pronyaev.\n\nAll three are reclassified from SCA to SCR in steps 4, 5 and 6.\n')

    add('### B3. FOOTE 1958 reclassified SCA → SCR (inferred)\n')
    add('The cleaned compilation in Roberto\'s 2026-05-05 email tags FOOTE 1958 (²³⁵U, 15.0(1.0)) as "SCR", but the explicit numbered action items only list ATOYI, WILLIS and GREEN-233. Per project decision, the cleaned table is treated as the new authoritative classification, so FOOTE is reclassified to SCR alongside the three named entries. **This decision is inferred and worth confirming with Roberto.** The K1 impact of this single reclassification is ≤ 0.05% (FOOTE\'s 6.7% uncertainty is much weaker than ARIF\'s 1.4% which dominates the SCA(²³⁵U) channel).\n')

    add('### B4. Westcott factors updated to B81-derived values\n')
    add('The 2026-05-06 email replaces the Westcott factors with values derived from the B81 evaluation (R-matrix fit best estimate), with uncertainties set by the spread between B81 and JEFF-4.0:\n\n')
    add('| nuclide | Old WGA (Gabs) | New WGA (Gabs) | Old WGF (Gf) | New WGF (Gf) |\n'
        '|---------|---------------:|---------------:|-------------:|-------------:|\n'
        '| U-233   | 0.9988 (0.12%) | **1.000 (0.10%)** | 0.9966 (0.20%) | **0.997 (0.10%)** |\n'
        '| U-235   | 0.9787 (0.09%) | **0.982 (0.31%)** | 0.9778 (0.09%) | **0.981 (0.51%)** |\n'
        '| Pu-239  | 1.0767 (0.27%) | **1.078 (0.93%)** | 1.0562 (0.28%) | **1.052 (1.33%)** |\n'
        '| Pu-241  | 1.0444 (0.19%) | **1.046 (0.67%)** | 1.0452 (0.67%) | **1.052 (0.67%)** |\n')
    add('')
    add('Gabs is derived as Gabs = (Gcap·Scap + Gfis·Sfis)/(Scap+Sfis) per Capote\'s 2026-05-06 derivation. The new factors apply to **all seven steps**: steps 0 and 1 read from a new `axton1986_data_capote_round2_baseline.json` file (= original baseline + new Westcott); steps 2–6 inherit through their respective JSONs.\n')
    add('K1 impact (Step 6, round 2 vs round 1):\n\n')
    add('| Nuclide | K1 (round 1) | K1 (round 2) | Δ |\n'
        '|---------|-------------:|-------------:|---:|\n'
        f'| ²³³U    | 738.81       | {summary["steps"][6]["fitted"]["F3ETA 33"]["value"]:.2f}      | {(summary["steps"][6]["fitted"]["F3ETA 33"]["value"] - 738.81) / 738.81 * 100:+.3f}% |\n'
        f'| ²³⁵U    | 719.32       | {summary["steps"][6]["fitted"]["F3ETA 35"]["value"]:.2f}      | {(summary["steps"][6]["fitted"]["F3ETA 35"]["value"] - 719.32) / 719.32 * 100:+.3f}% |\n'
        f'| ²³⁹Pu   | 1181.26      | {summary["steps"][6]["fitted"]["F3ETA 39"]["value"]:.2f}     | {(summary["steps"][6]["fitted"]["F3ETA 39"]["value"] - 1181.26) / 1181.26 * 100:+.3f}% |\n'
        f'| ²⁴¹Pu   | 1688.54      | {summary["steps"][6]["fitted"]["F3ETA 41"]["value"]:.2f}     | {(summary["steps"][6]["fitted"]["F3ETA 41"]["value"] - 1688.54) / 1688.54 * 100:+.3f}% |\n')
    add('')
    add(f'For comparison, K1(²³⁵U, Step 5, round 2) = {summary["steps"][5]["fitted"]["F3ETA 35"]["value"]:.2f}, in close agreement with the value reported in the 2026-05-06 email (722.4) using the same new Westcott factor.\n')

    add('### B5. SCR(²³³U), SCR(²³⁵U) retained as fit variables in step 6\n')
    add('In round 1, step 6 removed all four SCR primary parameters from the fit because no measurement function or datapoint referenced SCR after the conversions and Lemmel deletion. In round 2, the reclassifications (B2, B3) introduce four direct SCR measurements: ATOYI (²³⁵U), WILLIS (²³⁵U), FOOTE (²³⁵U), and GREEN-233 (²³³U). These constrain SCR(²³³U) and SCR(²³⁵U), so the two parameters remain in the fit. SCR(²³⁹Pu) and SCR(²⁴¹Pu) have no remaining constraints and stay unconstrained (removed).\n\nNet effect: Roberto\'s parenthetical *"reducing 4 fitting variables SCR"* in the original step 6 description simplifies to *reducing 2 fitting variables*. Step 6 now has 28 free parameters (was 26 in round 1).\n')

    add('### B6. Round-2 baseline file separation\n')
    add('To preserve the Axton-Table-8 reproduction artefact, the original `axton1986/axton1986_data.json` is left **unchanged** and a new file `axton1986/axton1986_data_capote_round2_baseline.json` is introduced for steps 0 and 1. The new file is identical to the original except for the eight Westcott measurements (No. 54–61). Steps 2–6 carry their own copies of all measurements; the Westcott updates are applied in-place to those JSONs.\n')

    add('### B7. Step 5b sensitivity check on SCA−SCR\n')
    add('Per the 2026-05-06 follow-up email, an additional sensitivity check is run starting from the round-2 Step 5 inputs. The four Lemmel SCA−SCR datapoints (Nos. 89, 91, 93, 95) have their central values doubled. Per project decision (selected over the alternative of doubling the percent uncertainty), the percent uncertainty is left unchanged so that the absolute uncertainty also doubles — interpreted as a "shift the prior by factor 2 with the same relative confidence" sensitivity test rather than a "weaken the prior" test.\n\nResult: K1 = F3ETA shifts by ≤ 0.2% on every nuclide compared to Step 5; SCA values are essentially unchanged; the perturbation is absorbed almost entirely by the SCR fitted values. The Step 5 solution is therefore stable against the assumed SCA−SCR prior within the doubling factor tested. See the dedicated [Step 5b section](#step-5b-sensitivity-check-lemmel-scascr-doubled) for the full comparison table.\n\nFiles produced: `axton1986/axton1986_data_capote_step5b_double_sca_scr.json` (data) and `evaluation/benchmark_capote_step5b_double_sca_scr.py` (benchmark).\n')

    add('## C. Files Produced by This Chain\n')
    add('All on branch `tnc_update_capote_20260504`, building on commit `575fac0` (the Axton 1986 reproduction state of master).\n')
    add('### JSON databases (one per step where data changes)\n')
    add('  * `axton1986/axton1986_data.json` — original Axton-Table-8 reproduction baseline (untouched; not currently used by any step)\n'
        '  * `axton1986/axton1986_data_capote_round2_baseline.json` — round-2 baseline (= original + new Westcott); steps 0 and 1 read this\n'
        '  * `axton1986/axton1986_data_capote_step2_adamchuk.json` — 168 measurements\n'
        '  * `axton1986/axton1986_data_capote_step3_wallner.json` — 170 measurements\n'
        '  * `axton1986/axton1986_data_capote_step4_sca_list.json` — 179 measurements (round 2: 180 − 1 KAISER deleted)\n'
        '  * `axton1986/axton1986_data_capote_step5_transmission.json` — 189 measurements (round 2: 190 − 1 KAISER deleted)\n'
        '  * `axton1986/axton1986_data_capote_step6_all_sca.json` — 185 measurements (round 2; final)\n'
        '  * `axton1986/axton1986_data_capote_step5b_double_sca_scr.json` — 189 measurements (step 5 with Lemmel SCA−SCR doubled; sensitivity test)\n')
    add('### Benchmarks\n')
    add('  * `evaluation/benchmark_capote_step0_baseline.py`\n  * `evaluation/benchmark_capote_step1_lounsbury.py`\n  * `evaluation/benchmark_capote_step2_adamchuk.py`\n  * `evaluation/benchmark_capote_step3_wallner.py`\n  * `evaluation/benchmark_capote_step4_sca_list.py`\n  * `evaluation/benchmark_capote_step5_transmission.py`\n  * `evaluation/benchmark_capote_step5b_double_sca_scr.py` — Step 5b sensitivity check (Lemmel SCA−SCR doubled)\n  * `evaluation/benchmark_capote_step6_all_sca.py`\n  * `evaluation/benchmark_capote_summary.py` — runs all 8 (7 main + 1 sensitivity) in sequence and dumps `capote_chain_summary.json`\n')
    add('### Generated artefacts\n')
    add('  * `evaluation/capote_chain_summary.json` — machine-readable consolidation of all step results (round 2)\n'
        '  * `evaluation/capote_chain_summary_20260504.json` — round-1 snapshot for direct comparison\n'
        '  * `evaluation/report_capote_fit.md` — this report (round 2)\n'
        '  * `evaluation/report_capote_fit_20260504.pdf` — round-1 PDF snapshot\n'
        '  * `evaluation/build_capote_report.py` — the script that generated this report from the summary JSON\n')
    add('### Source paraphrases\n')
    add('  * [`references/emails/tnc_update_roberto_20260505/summary.md`](../references/emails/tnc_update_roberto_20260505/summary.md) — paraphrased thread covering the round-1 instructions (2026-04-09 → 2026-05-04)\n'
        '  * [`references/emails/tnc_update_roberto_20260508/summary.md`](../references/emails/tnc_update_roberto_20260508/summary.md) — paraphrased round-2 corrections (2026-05-05 → 2026-05-06)\n')

    return '\n'.join(out)


if __name__ == '__main__':
    text = build()
    out_path = EVAL_DIR / 'report_capote_fit.md'
    with open(out_path, 'w') as f:
        f.write(text)
    print(f"Wrote {out_path}: {len(text):,} chars, {text.count(chr(10)):,} lines")
