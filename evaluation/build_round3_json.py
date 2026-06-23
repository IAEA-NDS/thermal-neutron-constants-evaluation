"""Build the round-3 TNC database from the round-2 Step-5 favoured base.

Applies the 2026-06-23 fission and total-cross-section (n,tot) data revision
(Capote / Duran). The complete datapoint-level rationale and decision register
is in references/emails/tnc_update_roberto_20260623/data_change_manifest.md.

Operations (keeping the current SCA/SCR classification of every existing
entry — only input_value and uncertainty_percent change on updates):

  - §A  15 transmission value/uncertainty updates
  - §B   2 fission value/uncertainty updates (existing Nos. 98, 108)
  - §C  10 deletions (outliers / duplicates)
  - §D  26 additions (23 fission + 3 new transmission, Nos. 195-220)
  - §E   U-234(n,g) central value -> 100.9 b (No. 5), uncertainty kept

Input : axton1986/axton1986_data_capote_step5_transmission.json (189 meas)
Output: axton1986/axton1986_data_capote_round3.json (205 meas)

Run:  python evaluation/build_round3_json.py
"""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / 'axton1986'
SRC = BASE / 'axton1986_data_capote_step5_transmission.json'
DST = BASE / 'axton1986_data_capote_round3.json'


# --- §A. Transmission value/uncertainty updates (value, absolute sigma) ---
# uncertainty_percent is recomputed from the absolute sigma below.
UPDATES_ABS = {
    63:  (699.0, 10.0),    # Melkonian U5  (ABS35)+SCR35
    64:  (700.0, 5.0),     # Palevsky  U5  (ABS35)+SCR35
    65:  (707.0, 10.0),    # Nitikin   U5  (ABS35)+SCA35
    186: (692.0, 8.0),     # Gerasimov U5  (ABS35)+SCR35   [F11: 8 b, not 1.4%]
    187: (701.0, 14.0),    # Leonard   U5  (ABS35)+SCR35
    73:  (1030.0, 20.0),   # Nitikin   Pu9 (ABS39)+SCA39
    74:  (1029.0, 10.0),   # Bollinger Pu9 (ABS39)+SCR39
    76:  (1030.0, 15.0),   # Safford   Pu9 (ABS39)+SCR39
    193: (1030.0, 10.0),   # Palevsky  Pu9 (ABS39)+SCR39
    75:  (1039.0, 25.0),   # Pattenden Pu9 (ABS39)+SCR39   [F1: 25 b, not 5%]
    78:  (580.0, 20.0),    # Nitikin   U3  (ABS33)+SCA33
    188: (588.0, 9.0),     # Pschenichnyj U3 (ABS33)+SCR33
    189: (588.0, 9.0),     # Harvey-1  U3  (ABS33)+SCR33   [F10: survivor]
    85:  (1385.0, 15.0),   # Simpson   Pu41 (ABS41)+SCA41
    87:  (1395.0, 10.0),   # Smith     Pu41 (ABS41)+SCR41
}

# --- §B. Fission value/uncertainty updates (value, uncertainty_percent) ---
UPDATES_PCT = {
    98:  (587.0, 1.0),     # Deruytter 1961 FIS35  (keeps systematic code B) [F7]
    108: (1019.3, 2.4),    # Watanabe 1964 FIS41                              [F7]
}

# --- §E. U-234(n,g) central-value update, uncertainty kept (No. 5) ---
U234_NO = 5
U234_NEW_VALUE = 100.9   # B80 value; uncertainty_percent left unchanged [F6]

# --- §C. Deletions ---
DELETE = {71, 79, 86, 103, 104, 106, 107, 190, 191, 194}

# --- §D.1 New fission datapoints (No. assigned in order from 195) ---
# (author, year, X4, measured_function, value, uncertainty_percent, note)
NEW_FISSION = [
    ("TOVESSON", 2010, 14271, "(FIS 39)/FIS 35", 1.286, 2.0,
     "Pu-239; calc. using B70 U5=585 and measured 752.4"),
    ("GWIN", 1984, 12905, "FIS 39", 750.2, 1.7, "Pu-239; I1=Nacho"),
    ("ZHURAVLEV", 1976, 40476, "(FIS 39)/FIS 35", 1.274, 2.0, "Pu-239"),
    ("DERUYTTER", 1970, 20143, "FIS 39", 749.3, 0.5,
     "Pu-239; adjusted by Nacho 741.9->749.3"),
    ("DERUYTTER", 1972, 20132, "FIS 39", 748.8, 1.0,
     "Pu-239; renormalized to Deruytter 70 (Nacho)"),
    ("STAVISSKII", 1965, 41635, "FIS 39", 748.0, 1.0, "Pu-239"),
    ("EGELSTAFF", 1955, 23526, "(FIS 41)/FIS 39", 1.33, 3.0, "Pu-241"),
    ("JAFFEY", 1955, 12521, "(FIS 41)/FIS 39", 1.362, 1.7, "Pu-241"),
    ("JAMES", 1965, 20940, "FIS 41", 1014.4, 2.0, "Pu-241; adjusted by Nacho"),
    ("WESTON", 1978, 10768, "FIS 41", 1018.8, 2.7, "Pu-241; adjusted by Nacho"),
    ("TOVESSON", 2010, 14271, "(FIS 41)/FIS 35", 1.73, 2.0,
     "Pu-241; used B70 U5=585"),
    ("BERCEANU", 1982, 30750, "(FIS 33)/FIS 35", 0.892, 2.0, "U-233"),
    ("ZHURAVLEV", 1976, 40476, "(FIS 33)/FIS 35", 0.908, 1.5, "U-233"),
    ("WENHAI", 1975, 32624, "(FIS 33)/FIS 35", 0.916, 1.0, "U-233"),
    ("DERUYTTER", 1974, 20411, "FIS 33", 533.2, 0.5,
     "U-233; F5: uncertainty not given, assumed 0.5% pending confirmation"),
    ("DERUYTTER", 1971, 20131, "FIS 35", 586.6, 0.8, "U-235; Nacho fit"),
    ("GWIN", 1984, 12905, "FIS 35", 586.1, 0.3,
     "U-235; Nacho fit, renormalized to Nacho I3"),
    ("BORCEA", 1973, 30140, "FIS 35", 586.3, 1.3,
     "U-235; Mihailescu relative to Borcea; F9: used instead of Berceanu 589.3"),
    ("AMADUCCI", 2022, 23453, "FIS 35", 586.2, 0.5, "U-235"),
    ("DERUYTTER", 1970, 20143, "FIS 35", 587.6, 0.4, "U-235"),
    ("YE", 1964, 32827, "FIS 35", 582.0, 1.5, "U-235; relative to B-nat 764 b"),
    ("FRAYSSE", 1965, 20071, "FIS 35", 584.3, 1.5,
     "U-235; rel. B-nat 761 b, rescaled by 764/761"),
    ("SAFFORD", 1959, 12439, "FIS 35", 580.0, 1.2, "U-235"),
]

# --- §D.2 New transmission datapoints (value, absolute sigma) ---
# (author, year, X4, measured_function, value, sigma_abs, note)
NEW_TRANSMISSION = [
    ("MOORE", 1960, 12341, "(ABS 33)+SCA 33", 587.0, 18.0,
     "U-233; NEW; F13: classification ABS+SCA assumed (Nacho U3 assumption)"),
    ("LYNN", 1955, 21027, "(ABS 35)+SCR 35", 704.0, 14.0,
     "U-235; NEW; F3: Nacho advised against adding; F13: ABS+SCR assumed"),
    ("YOUNG", 1968, 10406, "(ABS 41)+SCR 41", 1390.0, 10.0,
     "Pu-241; NEW; explicit ABS+SCR, metallic foil"),
]


def make_record(no, author, year, x4, func, value, unc_pct, note):
    return {
        "no": no,
        "author": author,
        "reference": f"{year} (X4 {x4})",
        "measured_function": func,
        "notes": note,
        "input_value": value,
        "uncertainty_percent": round(unc_pct, 4),
        "weighted_residual": 0.0,
        "systematic_uncertainties": {},
    }


def main():
    with open(SRC) as f:
        data = json.load(f)

    by_no = {m["no"]: m for m in data["measurements"]}
    log = []

    # §A transmission updates (absolute sigma -> percent)
    for no, (val, sig) in UPDATES_ABS.items():
        m = by_no[no]
        old = (m["input_value"], m["uncertainty_percent"])
        m["input_value"] = val
        m["uncertainty_percent"] = round(100.0 * sig / val, 4)
        log.append(f"  A update #{no:<3} {m['author']:<12} {m['measured_function']:<18}"
                   f" {old[0]}@{old[1]}% -> {val}@{m['uncertainty_percent']}% ({sig} b)")

    # §B fission updates (percent given directly)
    for no, (val, pct) in UPDATES_PCT.items():
        m = by_no[no]
        old = (m["input_value"], m["uncertainty_percent"])
        m["input_value"] = val
        m["uncertainty_percent"] = pct
        log.append(f"  B update #{no:<3} {m['author']:<12} {m['measured_function']:<18}"
                   f" {old[0]}@{old[1]}% -> {val}@{pct}%")

    # §E U-234(n,g)
    m = by_no[U234_NO]
    old = m["input_value"]
    m["input_value"] = U234_NEW_VALUE
    log.append(f"  E update #{U234_NO:<3} {m['author']:<12} {m['measured_function']:<18}"
               f" {old} -> {U234_NEW_VALUE} (unc kept {m['uncertainty_percent']}%)")

    # §C deletions
    kept = []
    for m in data["measurements"]:
        if m["no"] in DELETE:
            log.append(f"  C delete #{m['no']:<3} {m['author']:<12} {m['measured_function']:<18}"
                       f" val={m['input_value']}")
        else:
            kept.append(m)
    data["measurements"] = kept

    # §D additions — number from after the original max (194) so new entries
    # never reuse a deleted entry's serial number (emails reference those).
    next_no = max(by_no) + 1
    for author, year, x4, func, val, pct, note in NEW_FISSION:
        rec = make_record(next_no, author, year, x4, func, val, pct, note)
        data["measurements"].append(rec)
        log.append(f"  D add    #{next_no:<3} {author:<12} {func:<18} {val}@{pct}%  [{note[:40]}]")
        next_no += 1
    for author, year, x4, func, val, sig, note in NEW_TRANSMISSION:
        pct = round(100.0 * sig / val, 4)
        rec = make_record(next_no, author, year, x4, func, val, pct, note)
        data["measurements"].append(rec)
        log.append(f"  D add    #{next_no:<3} {author:<12} {func:<18} {val}@{pct}% ({sig} b)  [new transmission]")
        next_no += 1

    # metadata provenance
    data.setdefault("metadata", {}).setdefault("round3_corrections", [])
    data["metadata"]["round3_corrections"] = [
        "2026-06-23: fission + (n,tot) data revision (Capote/Duran). Built from "
        "step-5 favoured base. 15 transmission updates, 2 fission updates, 10 "
        "deletions, 26 additions (Nos. 195-220), U-234(n,g)->100.9 b. "
        "SCA/SCR classification of existing entries unchanged. See "
        "references/emails/tnc_update_roberto_20260623/data_change_manifest.md",
    ]

    with open(DST, "w") as f:
        json.dump(data, f, indent=1)

    print("\n".join(log))
    print("-" * 70)
    print(f"Base measurements : {len(by_no)}")
    print(f"Deleted           : {len(DELETE)}")
    print(f"Added             : {len(NEW_FISSION) + len(NEW_TRANSMISSION)}")
    print(f"Final measurements: {len(data['measurements'])}")
    print(f"Written to        : {DST}")


if __name__ == "__main__":
    main()
