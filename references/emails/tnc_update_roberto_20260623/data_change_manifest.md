# Round-3 Data-Change Manifest (2026-06-23)

Datapoint-level before→after for every change implied by the
[2026-06-23 email round](summary.md). "Before" values are the authoritative
current entries read from `axton1986/axton1986_data_capote_step5_transmission.json`
(the round-2 Step-5 favoured base, 189 measurements). Absolute uncertainties
quoted by Roberto/Nacho are converted to the `uncertainty_percent` field used
by the database; both are shown so the final input list (per the "new TNC
fit" request) can carry absolute *and* relative uncertainty.

**Convention preserved:** SCA/SCR classification of every existing entry is
unchanged; only `input_value` and `uncertainty_percent` are touched. The
`measured_function` string is therefore retained verbatim on all updates.

Legend: ✅ unambiguous · ⚠️ involves a documented decision (see §F /
decision register).

---

## §A. Value / uncertainty updates to existing transmission entries

| No. | Author | measured_function (kept) | Before value | Before σ% | After value | After σ (abs) | After σ% | Flag |
|----:|--------|--------------------------|-------------:|----------:|------------:|--------------:|---------:|:----:|
| 63 | MELKONIAN | (ABS 35)+SCR 35 | 694.32 | 2.016% | 699 | 10 b | 1.431% | ✅ |
| 64 | PALEVSKY | (ABS 35)+SCR 35 | 700.32 | 1.428% | 700 | 5 b | 0.714% | ✅ |
| 65 | NITIKIN | (ABS 35)+SCA 35 | 710.32 | 2.956% | 707 | 10 b | 1.414% | ✅ |
| 186 | GERASIMOV | (ABS 35)+SCR 35 | 685.0 | 1.314% | 692 | 8 b | 1.156% | ⚠️ F11 |
| 187 | LEONARD | (ABS 35)+SCR 35 | 702.0 | 1.994% | 701 | 14 b | 1.997% | ✅ |
| 73 | NITIKIN | (ABS 39)+SCA 39 | 1040.5 | 2.979% | 1030 | 20 b | 1.942% | ✅ |
| 74 | BOLLINGER | (ABS 39)+SCR 39 | 1022.5 | 1.369% | 1029 | 10 b | 0.972% | ✅ |
| 76 | SAFFORD | (ABS 39)+SCR 39 | 1018.5 | 0.727% | 1030 | 15 b | 1.456% | ✅ |
| 193 | PALEVSKY | (ABS 39)+SCR 39 | 1025.0 | 0.976% | 1030 | 10 b | 0.971% | ✅ |
| 75 | PATTENDEN | (ABS 39)+SCR 39 | 1012.5 | 3.003% | 1039 | 25 b | 2.406% | ⚠️ F1 |
| 78 | NITIKIN | (ABS 33)+SCA 33 | 577.26 | 3.638% | 580 | 20 b | 3.448% | ✅ |
| 188 | PSCHENICHNYI | (ABS 33)+SCR 33 | 587.9 | 0.510% | 588 | 9 b | 1.531% | ✅ |
| 189 | HARVEY-1 | (ABS 33)+SCR 33 | 585.0 | 0.239% | 588 | 9 b | 1.531% | ⚠️ F10 |
| 85 | SIMPSON | (ABS 41)+SCA 41 | 1386.6 | 3.606% | 1385 | 15 b | 1.083% | ✅ |
| 87 | SMITH | (ABS 41)+SCR 41 | 1395.6 | 1.433% | 1395 | 10 b | 0.717% | ✅ |

Kept explicitly "as is" (no change): #69 Block-U5, #66 Simpson-U5, #67/#68
Safford-U5, #70 Saplakoglu-U5, #185 Antonov, #166 Spencer-U5, #84 Block-U3,
#82/#83 Safford-U3, #77 Muether-U3, #81 Simpson-U3, #72 Zimmerman-Pu9,
#192 Anderson-Pu9, #167 Spencer-Pu9.

## §B. Value / uncertainty updates to existing fission entries

| No. | Author | measured_function | Before value | Before σ% | After value | After σ% | Flag |
|----:|--------|-------------------|-------------:|----------:|------------:|---------:|:----:|
| 98 | DERUYTTER (1961) | FIS 35 | 589.37 | 1.325% | 587 | 1.0% | ⚠️ F7 |
| 108 | WATANABE (1964) | FIS 41 | 985.45 | 4.569% | 1019.3 | 2.4% | ⚠️ F7 |

> F7: these match existing entries by author/year/quantity/EXFOR id and are
> tagged "adjusted by Nacho", so treated as **updates** rather than new
> points. #98 carries a systematic component `B:0.232`; it is retained
> unless Roberto indicates the re-fit removes it.

## §C. Deletions

| No. | Author | measured_function | Value | Reason (per email) | Flag |
|----:|--------|-------------------|------:|--------------------|:----:|
| 71 | GERASIMOV | ABS 35 | 670.31 | "same exp as total, derived quantity" (dup of #186) | ✅ |
| 79 | PATTENDEN | (ABS 33)+SCR 33 | 596.27 | U3(n,tot): "discard, too much scatter" | ✅ |
| 86 | CRAIG | (ABS 41)+SCA 41 | 1380.6 | Pu41(n,tot): "discard, too large scattering" | ✅ |
| 103 | RAFFLE (1959) | FIS 35 | 581.97 | "delete, outlier" | ✅ |
| 104 | RAFFLE (1959) | 39 FH1 39 | 1714.8 | "OUTLIER DISCARD" | ✅ |
| 106 | MASLIN | FIS 35 | 583.71 | "OUTLIER" | ✅ |
| 107 | SAPLAKOGLU (1959) | FIS 35 | 593.17 | "OUTLIER" | ✅ |
| 190 | HARVEY-2 | (ABS 33)+SCR 33 | 592.0 | U3(n,tot): "Delete Harvey 79 data # 190" | ⚠️ F10 |
| 191 | SCHWARTZ | (ABS 41)+SCR 41 | 1410.0 | Pu41(n,tot): "discard, too large scattering" | ✅ |
| 194 | LEONARD (1956) | (ABS 39)+SCR 39 | 1055.0 | "Delete Leonard 56 (OUTLIER)" | ⚠️ F2 |

## §D. New datapoints

Proposed numbering continues from the current max (#194) → new entries
**#195 onward**. Final numbers assigned at implementation.

### D.1 New fission datapoints

| Author | measured_function | Value | σ (given) | σ% | EXFOR | Flag |
|--------|-------------------|------:|----------:|---:|------:|:----:|
| Tovesson 10 (Pu-239) | (FIS 39)/FIS 35 | 1.286 | 2% | 2.0% | 14271 | ✅ |
| Gwin 84 (Pu-239) | FIS 39 | 750.2 | 1.7% | 1.7% | 12905 | ✅ |
| Zhuravlev 76 (Pu-239) | (FIS 39)/FIS 35 | 1.274 | 2% | 2.0% | 40476 | ✅ |
| Deruytter 70 (Pu-239) | FIS 39 | 749.3 | 0.5% | 0.5% | 20143 | ✅ |
| Deruytter 72 (Pu-239) | FIS 39 | 748.8 | 1.0% | 1.0% | 20132 | ✅ |
| Stavisskii 65 (Pu-239) | FIS 39 | 748 | 1% | 1.0% | 41635 | ✅ |
| Egelstaff 55 (Pu-241) | (FIS 41)/FIS 39 | 1.33 | 3% | 3.0% | 23526 | ✅ |
| Jaffey 55 (Pu-241) | (FIS 41)/FIS 39 | 1.362 | 1.7% | 1.7% | 12521 | ✅ |
| James 65 (Pu-241) | FIS 41 | 1014.4 | 2% | 2.0% | 20940 | ✅ |
| Weston 78 (Pu-241) | FIS 41 | 1018.8 | 2.7% | 2.7% | 10768 | ✅ |
| Tovesson 10 (Pu-241) | (FIS 41)/FIS 35 | 1.73 | 2% | 2.0% | 14271 | ✅ |
| Berceanu 82 (U-233) | (FIS 33)/FIS 35 | 0.892 | 2% | 2.0% | 30750 | ✅ |
| Zhuravlev 76 (U-233) | (FIS 33)/FIS 35 | 0.908 | 1.5% | 1.5% | 40476 | ✅ |
| Wenhai 75 (U-233) | (FIS 33)/FIS 35 | 0.916 | 1% | 1.0% | 32624 | ✅ |
| Deruytter 74 (U-233) | FIS 33 | 533.2 | **none given** | **0.5% (assumed)** | 20411 | ⚠️ F5 |
| Deruytter 71 (U-235) | FIS 35 | 586.6 | 0.8% | 0.8% | 20131 | ✅ |
| Gwin 84 (U-235) | FIS 35 | 586.1 | 0.3% | 0.3% | 12905 | ✅ |
| Borcea 73 (U-235) | FIS 35 | 586.3 | 1.3% | 1.3% | 30140 | ⚠️ F9 |
| Amaducci 22 (U-235) | FIS 35 | 586.2 | 0.5% | 0.5% | 23453 | ✅ |
| Deruytter 70 (U-235) | FIS 35 | 587.6 | 0.4% | 0.4% | 20143 | ✅ |
| Ye 64 (U-235) | FIS 35 | 582 | 1.5% | 1.5% | 32827 | ✅ |
| Fraysse 65 (U-235) | FIS 35 | 584.3 | 1.5% | 1.5% | 20071 | ✅ |
| Safford 59 (U-235) | FIS 35 | 580 | 1.2% | 1.2% | 12439 | ✅ |

Not added (per email): Berceanu U-235 589.3 ("replace by Borcea 73", F9);
Raffle 59 U-235 582 ("delete, outlier"); Bowman / Melkonian / Schrack /
Wagemans 79,88 / Mihailescu (shape measurements, F12); Bollinger 52 U-233
(in Nacho's note, dropped from Roberto's list, F4).

### D.2 New transmission datapoints — classification to confirm

| Author | Nuclide | Proposed measured_function | Value | σ (abs) | σ% | Classification basis | Flag |
|--------|---------|----------------------------|------:|--------:|---:|----------------------|:----:|
| Moore 60 | U-233 | (ABS 33)+SCA 33 | 587 | 18 b | 3.066% | Nacho assumed U3(n,tot)=ABS33+SCA33; no rolled-metal note | ⚠️ F13 |
| Lynn 55 | U-235 | (ABS 35)+SCR 35 | 704 | 14 b | 1.989% | default era convention (metallic); Nacho advised against adding at all | ⚠️ F3/F13 |
| Young 68 | Pu-241 | (ABS 41)+SCR 41 | 1390 | 10 b | 0.719% | **explicit** "ABS+SCR, metallic foil" | ✅ |

## §E. Fixed-input update

| No. | Author | measured_function | Before | After | Notes | Flag |
|----:|--------|-------------------|-------:|------:|-------|:----:|
| 5 | LEMMEL | CAP 34 | 95.9 b, 2.086% | **100.9 b** | B80 value; σ% kept at 2.086% (= 2.105 b) for lack of a stated uncertainty | ⚠️ F6 |

## §F. Open interpretation points (decision register — for Roberto's review)

Each resolved with the most plausible reading; all flagged here and to be
mirrored in the report's new round-3 decisions appendix.

- **F1 — Pattenden 56 Pu-239 σ.** Fission email "1039(5%)" vs Pu9(n,tot)
  email "1039(25)" (=2.406%). → **Use 1039 ± 25 b**; the dedicated, later
  (n,tot) revision with Nacho's plot-fit is taken as authoritative.
- **F2 — Leonard 56 Pu-239 (#194).** Pu9 table shows an adjusted 1042(25),
  but fission email + Nacho say outlier. → **Delete #194.**
- **F3 — Lynn 55 U-235 (n,tot).** Roberto lists as "NEW set"; Nacho advised
  against. → **Add, 704 ± 14 b** (Roberto's summary governs); dissent noted.
- **F4 — Bollinger 52 U-233 (n,tot).** In Nacho's note, omitted from
  Roberto's list. → **Not added.**
- **F5 — Deruytter 74 FIS 33 = 533.2, no σ.** → **Provisional σ = 0.5%**
  (matches Deruytter's other absolute fission precisions); confirm value.
- **F6 — U-234(n,γ) = 100.9 b, no σ.** → **Keep σ = 2.086%** (2.105 b);
  remains a fixed input as before.
- **F7 — Watanabe-FIS41 / Deruytter61-FIS35.** → Treated as **updates** to
  #108 / #98 (not new points).
- **F8 — Berceanu 77 (#101) "CONTAINS BORCEA 70".** Read as a
  double-counting warning. → **Keep #101 as-is; do not add Borcea 70
  separately.**
- **F9 — Berceanu U-235 FIS35 589.3 "replace by Borcea 73".** → **Drop
  Berceanu 589.3; add Borcea 73 = 586.3 (1.3%).**
- **F10 — Harvey 79 U-233 numbering.** Roberto labels the survivor "#189"
  but quotes Georg value 592 (which is #190; #189=585). → **Keep one Harvey
  = 588(9) in #189; delete #190.** (Survivor is overwritten, so the old
  585-vs-592 distinction is immaterial.)
- **F11 — Gerasimov 62 U-235 σ.** Fission email "692(1.4%)" vs U5 email
  "692(8)"=1.156%. → **Use 692 ± 8 b.**
- **F12 — Shape measurements.** Bowman, Melkonian, Schrack, Wagemans 79/88,
  Mihailescu = shape (relative to thermal). → **Not added as absolute
  points.** (This Melkonian ≠ the U-235 (n,tot) #63, which is retained.)
- **F13 — New-transmission classification.** "Keep current SCA/SCR" does not
  cover brand-new points. → Young = ABS+SCR (explicit); **Moore = ABS+SCA,
  Lynn = ABS+SCR provisional**, to confirm. Note: under a Step-6-style "all
  ABS+SCR → ABS+SCA" conversion this choice is moot; it only affects the
  favoured Step-5 fit.

## Change tally

| Category | Count |
|----------|------:|
| Transmission value/σ updates (§A) | 15 |
| Fission value/σ updates (§B) | 2 |
| Deletions (§C) | 10 |
| New fission points (§D.1) | 23 |
| New transmission points (§D.2) | 3 |
| Fixed-input update (§E) | 1 |
| Net measurement count | 189 − 10 + 26 = **205** |
