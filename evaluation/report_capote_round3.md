# Round-3 TNC Fit (2026-06-23): Fission + (n,tot) Data Revision

*Comprehensive fission and total-cross-section data review by R. Capote and I. Durán (2026-06-23 discussions), built on the round-2 Step-5 favoured fit. Audit trail: [`references/emails/tnc_update_roberto_20260623/`](../references/emails/tnc_update_roberto_20260623/summary.md), with a datapoint-level before→after manifest in [`data_change_manifest.md`](../references/emails/tnc_update_roberto_20260623/data_change_manifest.md).*

This round applies a large fission-data extension (new absolute and ratio measurements across all four fissile nuclides), value/uncertainty updates to the total-cross-section (transmission) entries from local EXFOR re-fits, removal of identified outliers and duplicates, and an update of the U-234(n,γ) value to the B80 evaluation. The **current SCA/SCR classification of every existing entry is kept** per the instruction; only values and uncertainties change on updated entries. The fit configuration is identical to round-2 Step 5 (30 free parameters; Lounsbury alpha included; the four Lemmel SCA−SCR points retained, so all four SCR parameters remain in the fit).

Every interpretation that departs from a literal reading of the source tables is recorded in the [decisions appendix](#decisions-flags-f1f13) (flags F1–F13) for review.

## Overview

| Fit | N meas | N free | DOF | χ² | χ²/DOF | Converged |
|-----|-------:|-------:|----:|---:|-------:|:---------:|
| Round-2 Step 5 (base) | 127 | 30 | 97 | 80.55 | 0.830 | Yes |
| **Round 3 (2026-06-23)** | **143** | 30 | 113 | **73.42** | 0.650 | Yes |

Full database: 205 measurements (189 base − 10 deletions + 26 additions); 143 enter the microscopic fit after the Maxwellian filter. χ² *decreases* despite the net +16 measurements, because the deletions remove high-residual outliers while the new fission data are mutually consistent.

## Change Summary

| Category | Count | Serial numbers |
|----------|------:|----------------|
| Transmission value/σ updates | 15 | 63, 64, 65, 73, 74, 75, 76, 78, 85, 87, 186, 187, 188, 189, 193 |
| Fission value/σ updates | 2 | 98, 108 |
| U-234(n,γ) → 100.9 b (B80) | 1 | 5 |
| Deletions (outliers/duplicates) | 10 | 71, 79, 86, 103, 104, 106, 107, 190, 191, 194 |
| Additions (23 fission + 3 transmission) | 26 | 195–220 |

## Fitted Quantities (Round 3 vs Step-5 base)

Shifts exceeding 0.1% relative to the Step-5 base are highlighted in **bold**. K1 = F3ETA = NUB·FF − FA is the central derived quantity.

| Quantity | Step 5 | Round 3 | Unc (%) | Δ vs Step 5 (%) |
|----------|-------:|--------:|--------:|----------------:|
| **SCA 33** | 13.3908 | **13.3371** | 2.615 | **-0.401** |
| SCA 35 | 14.1213 | 14.1157 | 1.264 | -0.040 |
| **SCA 39** | 7.4728 | **7.4805** | 5.105 | **+0.104** |
| SCA 41 | 11.9951 | 11.9908 | 6.657 | -0.036 |
| **SCR 33** | 12.3795 | **12.3135** | 3.759 | **-0.533** |
| SCR 35 | 12.7734 | 12.7656 | 2.628 | -0.061 |
| **SCR 39** | 6.6595 | **6.7373** | 15.929 | **+1.169** |
| **SCR 41** | 10.8374 | **10.9254** | 24.836 | **+0.812** |
| **ABS 33** | 573.4874 | **574.9932** | 0.239 | **+0.263** |
| **ABS 35** | 683.4317 | **684.4781** | 0.144 | **+0.153** |
| **ABS 39** | 1020.6859 | **1022.0765** | 0.290 | **+0.136** |
| **ABS 41** | 1383.6990 | **1380.6308** | 0.436 | **-0.222** |
| **FIS 33** | 528.6252 | **530.3673** | 0.255 | **+0.330** |
| **FIS 35** | 584.3285 | **585.4044** | 0.141 | **+0.184** |
| **FIS 39** | 749.1488 | **750.0765** | 0.238 | **+0.124** |
| **FIS 41** | 1018.5549 | **1016.6900** | 0.514 | **-0.183** |
| **CA 33** | 44.8622 | **44.6259** | 1.785 | **-0.527** |
| CA 35 | 99.1032 | 99.0737 | 0.767 | -0.030 |
| **CA 39** | 271.5372 | **272.0000** | 0.884 | **+0.170** |
| **CA 41** | 365.1441 | **363.9407** | 1.251 | **-0.330** |
| **CAP 34** | 95.9000 | **100.9000** | 2.086 | **+5.214** |
| NUB 33 | 2.4914 | 2.4908 | 0.183 | -0.025 |
| NUB 35 | 2.4304 | 2.4301 | 0.168 | -0.013 |
| NUB 39 | 2.8824 | 2.8823 | 0.190 | -0.005 |
| NUB 41 | 2.9444 | 2.9440 | 0.206 | -0.013 |
| NUB 52 | 3.7668 | 3.7666 | 0.129 | -0.004 |
| ETA 33 | 2.2965 | 2.2975 | 0.197 | +0.042 |
| ETA 35 | 2.0780 | 2.0783 | 0.186 | +0.018 |
| ETA 39 | 2.1156 | 2.1152 | 0.261 | -0.017 |
| ETA 41 | 2.1674 | 2.1680 | 0.370 | +0.026 |
| **ALPHA 33** | 0.0849 | **0.0841** | 1.869 | **-0.853** |
| **ALPHA 35** | 0.1696 | **0.1692** | 0.811 | **-0.213** |
| ALPHA 39 | 0.3625 | 0.3626 | 0.919 | +0.047 |
| **ALPHA 41** | 0.3585 | **0.3580** | 1.466 | **-0.147** |
| WGA 33 | 1.0000 | 1.0000 | 0.100 | +0.000 |
| WGA 35 | 0.9820 | 0.9820 | 0.305 | +0.000 |
| WGA 39 | 1.0780 | 1.0780 | 0.928 | +0.000 |
| WGA 41 | 1.0460 | 1.0460 | 0.669 | +0.000 |
| WGF 33 | 0.9970 | 0.9970 | 0.100 | +0.000 |
| WGF 35 | 0.9810 | 0.9810 | 0.510 | +0.000 |
| WGF 39 | 1.0520 | 1.0520 | 1.331 | +0.000 |
| WGF 41 | 1.0520 | 1.0520 | 0.665 | +0.000 |
| **FA 33** | 573.4874 | **574.9932** | 0.259 | **+0.263** |
| **FA 35** | 671.1299 | **672.1575** | 0.338 | **+0.153** |
| **FA 39** | 1100.2995 | **1101.7984** | 0.972 | **+0.136** |
| **FA 41** | 1447.3492 | **1444.1398** | 0.799 | **-0.222** |
| **FF 33** | 527.0393 | **528.7762** | 0.274 | **+0.330** |
| **FF 35** | 573.2262 | **574.2817** | 0.529 | **+0.184** |
| **FF 39** | 788.1045 | **789.0804** | 1.352 | **+0.124** |
| **FF 41** | 1071.5198 | **1069.5579** | 0.841 | **-0.183** |
| F1ETA 33 | 2.2897 | 2.2906 | 0.243 | +0.042 |
| F1ETA 35 | 2.0758 | 2.0762 | 0.623 | +0.018 |
| F1ETA 39 | 2.0646 | 2.0642 | 1.643 | -0.017 |
| F1ETA 41 | 2.1798 | 2.1804 | 1.014 | +0.026 |
| **F2ETA 33** | 1313.0907 | **1317.0883** | 0.312 | **+0.304** |
| **F2ETA 35** | 1393.1589 | **1395.5398** | 0.552 | **+0.171** |
| **F2ETA 39** | 2271.6351 | **2274.3355** | 1.363 | **+0.119** |
| **F2ETA 41** | 3154.9640 | **3148.7864** | 0.841 | **-0.196** |
| **F3ETA 33** | 739.6033 | **742.0951** | 0.450 | **+0.337** |
| **F3ETA 35** | 722.0290 | **723.3823** | 1.085 | **+0.187** |
| **F3ETA 39** | 1171.3357 | **1172.5370** | 2.763 | **+0.103** |
| **F3ETA 41** | 1707.6149 | **1704.6467** | 1.541 | **-0.174** |

## K1 = F3ETA (Round 3)

| Nuclide | K1 = F3ETA | Unc (%) | Step-5 base | Δ% |
|---------|-----------:|--------:|------------:|---:|
| ²³³U | 742.10 | 0.450 | 739.60 | +0.337% |
| ²³⁵U | 723.38 | 1.085 | 722.03 | +0.187% |
| ²³⁹Pu | 1172.54 | 2.763 | 1171.34 | +0.103% |
| ²⁴¹Pu | 1704.65 | 1.541 | 1707.61 | -0.174% |

K1(²³⁵U) = 723.4, consistent with the value (~722.4) anticipated in the 2026-05-06 discussion using the same B81-derived Westcott factor. The new fission data shift all four K1 values by less than 0.4%.

## Full Input-Data List (absolute + relative uncertainty)

Per the 2026-06-23 "new TNC fit" request, the complete round-3 input database with **both absolute and relative uncertainty** for every measurement, for cross-checking against Axton and the 2026-06-23 changes. Status: **NEW** = added this round (Nos. 195–220); **UPD** = value/uncertainty updated this round; blank = unchanged from the Step-5 base. "In fit" marks the measurements that pass the microscopic filter and enter the GLS fit (others are Maxwellian-derived and used only for derived-quantity folding).

| No. | Author | Reference | Quantity | Value | σ (abs) | σ (%) | Status | In fit |
|----:|--------|-----------|----------|------:|--------:|------:|:------:|:------:|
| 1 | LAPONCHE | 1972(2) | (FA 39)/FA 35 | 1.6491 | 0.00961 | 0.583 |  |  |
| 2 | CORNISH | 1956(2) | CAP 39 | 312 | 14 | 4.490 |  |  |
| 3 | HALPERIN | 1963(2) | CAP 33 | 49.53 | 3.17 | 6.400 |  |  |
| 4 | CABELL | 1971(2) | FLEM | 0.09827 | 0.00329 | 3.348 |  |  |
| 5 | LEMMEL | 1982(1) | CAP 34 | 100.9 | 2.1 | 2.086 | UPD | yes |
| 6 | POPOVIC | 1953(2) | FF 35 | 571.1 | 12.6 | 2.211 |  |  |
| 7 | POPOVIC | 1955(2) | 33 FFH 33 | 844.72 | 26.6 | 3.147 |  |  |
| 8 | KEITH | 1968(2) | 33 FFH 33 | 834.15 | 11.4 | 1.371 |  |  |
| 9 | KEITH | 1968(2) | 33 FFH 35 | 889.01 | 12.9 | 1.446 |  |  |
| 10 | KEITH | 1968(2) | 39 FFH 39 | 1884.4 | 17.9 | 0.949 |  |  |
| 11 | BIGHAM | 1975(2) | 33 FFH 33 | 835.2 | 5.5 | 0.659 |  |  |
| 12 | JAFFEY | 1955(2) | (FF 41)/FF 39 | 1.3552 | 0.0193 | 1.424 |  |  |
| 13 | WHITE | 1967(2) | (FF 39)/FF 35 | 1.3583 | 0.0291 | 2.139 |  |  |
| 14 | WHITE | 1967(2) | (FF 41)/FF 35 | 1.8806 | 0.0527 | 2.802 |  |  |
| 15 | VIDAL | 1970(2) | (FF 33)/FF 35 | 0.932 | 0.009 | 0.966 |  |  |
| 16 | BIGHAM | 1975(2) | (FF 33)/FF 35 | 0.9323 | 0.0042 | 0.450 |  |  |
| 17 | SWEET | 1973(2) | (39 FFH 39)/34 FFH 35 | 1.3544 | 0.0192 | 1.414 |  |  |
| 18 | BIGHAM | 1975(2) | (39 FFH 39)/33 FFH 33 | 2.2796 | 0.0153 | 0.669 |  |  |
| 19 | BIGHAM | 1975(2) | (39 FFH 39)/33 FFH 35 | 2.1239 | 0.0151 | 0.709 |  |  |
| 20 | BIGHAM | 1975(2) | F1BIG | 0.5527 | 0.00311 | 0.562 |  |  |
| 21 | LOUNSBURY | 1970(1) | (CA 33)/FIS 33 | 0.0861 | 0.00193 | 2.239 |  | yes |
| 22 | LOUNSBURY | 1970(1) | (CA 35)/FIS 35 | 0.1697 | 0.0029 | 1.709 |  | yes |
| 23 | LOUNSBURY | 1970(1) | (CA 39)/FIS 39 | 0.3555 | 0.0057 | 1.603 |  | yes |
| 24 | INGHRAM | 1956(2) | (CAP 33)/FF 33 | 0.09403 | 0.003 | 3.190 |  |  |
| 25 | CORNISH | 1960(2) | (CAP 35)/FF 35 | 0.188 | 0.014 | 7.447 |  |  |
| 26 | OKAZAKI | 1964(2) | (CAP 33)/FF 33 | 0.0902 | 0.00156 | 1.731 |  |  |
| 27 | OKAZAKI | A1964(2) | (CAP 35)/FF 35 | 0.1705 | 0.0018 | 1.056 |  |  |
| 28 | LISMAN | 1967(2) | (CAP 33)/FF 33 | 0.093 | 0.00198 | 2.128 |  |  |
| 29 | LISMAN | 1967(2) | (CAP 35)/FF 35 | 0.1712 | 0.00152 | 0.886 |  |  |
| 30 | CONWAY | 1967(2) | (CAP 33)/FF 33 | 0.0851 | 0.0042 | 4.935 |  |  |
| 31 | CONWAY | 1967(2) | (CAP 35)/FF 35 | 0.1705 | 0.0072 | 4.223 |  |  |
| 32 | DURHAM | 1967(2) | (CAP 35)/FF 35 | 0.1746 | 0.002 | 1.145 |  |  |
| 33 | DURHAM | 1967(2) | (CAP 39)/FF 39 | 0.3882 | 0.0059 | 1.520 |  |  |
| 34 | CABELL | 1971(2) | (CAP 33)/FF 33 | 0.0857 | 0.00261 | 3.048 |  |  |
| 35 | CABELL | 1971(2) | (CAP 35)/FF 35 | 0.1696 | 0.00298 | 1.759 |  |  |
| 36 | CABELL | 1971(2) | (CAP 39)/FF 39 | 0.3812 | 0.0183 | 4.808 |  |  |
| 37 | DEBOISBLANC | 1961(2) | (F1ETA 33)/F1ETA 35 | 1.115 | 0.00799 | 0.717 |  |  |
| 38 | FAST | 1960(2) | (F1ETA 39)/F1ETA 35 | 0.997 | 0.0162 | 1.625 |  |  |
| 39 | FAST | 1960(2) | (F1ETA 41)/F1ETA 35 | 1.0528 | 0.0174 | 1.653 |  |  |
| 40 | CABELL | 1965(2) | (F3ETA 33)/F3ETA 35 | 1.0461 | 0.0247 | 2.357 |  |  |
| 41 | CABELL | 1965(2) | (F3ETA 39)/F3ETA 35 | 1.6079 | 0.0386 | 2.403 |  |  |
| 42 | CABELL | 1965(2) | (F3ETA 33)/F3ETA 35 | 1.0309 | 0.0404 | 3.919 |  |  |
| 43 | CABELL | 1965(2) | (F3ETA 39)/F3ETA 35 | 1.6223 | 0.0675 | 4.161 |  |  |
| 44 | GWIN | 1962(2) | (F2ETA 33)/F2ETA 35 | 0.953 | 0.0135 | 1.414 |  |  |
| 45 | GWIN | 1962(2) | (F2ETA 39)/F2ETA 35 | 1.631 | 0.0231 | 1.414 |  |  |
| 46 | GWIN | A1962(2) | F3ETA 33 | 744.7 | 4 | 0.537 |  |  |
| 47 | GWIN | A1962(2) | F3ETA 35 | 722.7 | 3.9 | 0.540 |  |  |
| 48 | LAPONCHE | 1972(2) | (F3ETA 39)/F3ETA 35 | 1.6058 | 0.0178 | 1.107 |  |  |
| 49 | LAPONCHE | 1972(2) | (F3ETA 41)/F3ETA 39 | 1.4687 | 0.0526 | 3.580 |  |  |
| 50 | ALIKHANOV | 1956(2) | (F3ETA 33)/F3ETA 35 | 1.039 | 0.037 | 3.561 |  |  |
| 51 | ALIKHANOV | 1956(2) | (F3ETA 39)/F3ETA 35 | 1.565 | 0.105 | 6.709 |  |  |
| 52 | MUELHAUS | 1959(2) | (F3ETA 33)/F3ETA 35 | 1.0221 | 0.0185 | 1.811 |  |  |
| 53 | MUELHAUS | 1959(2) | (F3ETA 39)/F3ETA 35 | 1.6043 | 0.0364 | 2.271 |  |  |
| 54 | DIVADEENAM | 1984(2) | WGA 33 | 1 | 0.001 | 0.100 |  | yes |
| 55 | DIVADEENAM | 1984(2) | WGA 35 | 0.982 | 0.003 | 0.305 |  | yes |
| 56 | DIVADEENAM | 1984(2) | WGA 39 | 1.078 | 0.01 | 0.928 |  | yes |
| 57 | DIVADEENAM | 1984(2) | WGA 41 | 1.046 | 0.007 | 0.669 |  | yes |
| 58 | DIVADEENAM | 1984(2) | WGF 33 | 0.997 | 0.001 | 0.100 |  | yes |
| 59 | DERUYTTER | 1985(3) | WGF 35 | 0.981 | 0.005 | 0.510 |  | yes |
| 60 | DIVADEENAM | 1984(2) | WGF 39 | 1.052 | 0.014 | 1.331 |  | yes |
| 61 | DIVADEENAM | 1984(2) | WGF 41 | 1.052 | 0.007 | 0.665 |  | yes |
| 62 | EGELSTAFF | 1954(1) | (ABS 35)+SCR 35 | 724.33 | 26 | 3.590 |  | yes |
| 63 | MELKONIAN | 1953(1) | (ABS 35)+SCR 35 | 699 | 10 | 1.431 | UPD | yes |
| 64 | PALEVSKY | 1954(1) | (ABS 35)+SCR 35 | 700 | 5 | 0.714 | UPD | yes |
| 65 | NITIKIN | 1955(1) | (ABS 35)+SCA 35 | 707 | 10 | 1.414 | UPD | yes |
| 66 | SIMPSON | 1960(1) | (ABS 35)+SCR 35 | 690.32 | 9.6 | 1.391 |  | yes |
| 67 | SAFFORD | 1959(1) | (ABS 35)+SCA 35 | 696.32 | 2.5 | 0.359 |  | yes |
| 68 | SAFFORD | 1959(1) | (ABS 35)+SCA 35 | 698.32 | 5.1 | 0.730 |  | yes |
| 69 | BLOCK | 1960(1) | (ABS 35)+SCR 35 | 693.32 | 5 | 0.721 |  | yes |
| 70 | SAPLAKOGLU | 1961(1) | (ABS 35)+SCR 35 | 696.32 | 2.5 | 0.359 |  | yes |
| 72 | ZIMMERMAN | 1955(1) | (ABS 39)+SCR 39 | 1022.5 | 13 | 1.271 |  | yes |
| 73 | NITIKIN | 1955(1) | (ABS 39)+SCA 39 | 1030 | 20 | 1.942 | UPD | yes |
| 74 | BOLLINGER | 1958(1) | (ABS 39)+SCR 39 | 1029 | 10 | 0.972 | UPD | yes |
| 75 | PATTENDEN | A1956(1) | (ABS 39)+SCR 39 | 1039 | 25 | 2.406 | UPD | yes |
| 76 | SAFFORD | 1961(1) | (ABS 39)+SCR 39 | 1030 | 15 | 1.456 | UPD | yes |
| 77 | MUETHER | 1954(1) | (ABS 33)+SCR 33 | 597.27 | 14 | 2.344 |  | yes |
| 78 | NITIKIN | 1955(1) | (ABS 33)+SCA 33 | 580 | 20 | 3.448 | UPD | yes |
| 80 | GREEN | 1957(1) | ABS 33 | 574.26 | 20 | 3.483 |  | yes |
| 81 | SIMPSON | 1960(1) | (ABS 33)+SCR 33 | 587.27 | 4.7 | 0.800 |  | yes |
| 82 | SAFFORD | 1960(1) | (ABS 33)+SCA 33 | 585.77 | 5.8 | 0.990 |  | yes |
| 83 | SAFFORD | 1960(1) | (ABS 33)+SCA 33 | 585.67 | 2.4 | 0.410 |  | yes |
| 84 | BLOCK | 1960(1) | (ABS 33)+SCR 33 | 587.27 | 3 | 0.511 |  | yes |
| 85 | SIMPSON | 1961(1) | (ABS 41)+SCA 41 | 1385 | 15 | 1.083 | UPD | yes |
| 87 | SMITH | 1984(3) | (ABS 41)+SCR 41 | 1395 | 10 | 0.717 | UPD | yes |
| 89 | LEMMEL | 1982(1) | (SCA 33)-SCR 33 | 1.2006 | 0.6 | 49.977 |  | yes |
| 91 | LEMMEL | 1982(1) | (SCA 35)-SCR 35 | 1.7008 | 0.75 | 44.097 |  | yes |
| 93 | LEMMEL | 1982(1) | (SCA 39)-SCR 39 | 0.80037 | 1.1 | 137.437 |  | yes |
| 95 | LEMMEL | 1982(1) | (SCA 41)-SCR 41 | 1.2006 | 3 | 249.886 |  | yes |
| 96 | DERUYTTER | 1974(1) | 39 FH1 39 | 1812.1 | 8.23 | 0.454 |  | yes |
| 97 | DERUYTTER | 1973(1) | 34 FH1 35 | 1439.1 | 6.4 | 0.445 |  | yes |
| 98 | DERUYTTER | 1961(1) | FIS 35 | 587 | 5.87 | 1.000 | UPD | yes |
| 99 | WHITE | 1967(1) | (FIS 39)/FIS 35 | 1.253 | 0.0226 | 1.800 |  | yes |
| 100 | WHITE | 1967(1) | (FIS 41)/FIS 35 | 1.763 | 0.0485 | 2.751 |  | yes |
| 101 | BERCEANO | 1977(1) | 34 FH1 35 | 1443 | 11.4 | 0.791 |  | yes |
| 102 | RAFFLE | 1959(1) | 33 FH1 33 | 834.26 | 27.5 | 3.302 |  | yes |
| 105 | RAFFLE | 1956(1) | (FIS 41)/FIS 39 | 1.3319 | 0.0799 | 6.000 |  | yes |
| 108 | WATANABE | 1964(1) | FIS 41 | 1019.3 | 24.5 | 2.400 | UPD | yes |
| 109 | FRAYSSE | 1965(1) | (39 FH1 39)/34 FH1 35 | 1.2425 | 0.0193 | 1.557 |  | yes |
| 110 | BORCEA | 1970(1) | (39 FH1 39)/34 FH1 35 | 1.2578 | 0.0204 | 1.624 |  | yes |
| 111 | BOLDEMAN | 1977(1) | NUB 52 | 3.7549 | 0.0162 | 0.431 |  | yes |
| 112 | BOLDEMAN | 1980(1) | (NUB 33)/NUB 52 | 0.66049 | 0.00168 | 0.255 |  | yes |
| 113 | BOLDEMAN | 1980(1) | (NUB 35)/NUB 52 | 0.64318 | 0.00164 | 0.255 |  | yes |
| 114 | BOLDEMAN | 1980(1) | (NUB 39)/NUB 52 | 0.76614 | 0.00186 | 0.243 |  | yes |
| 115 | BOLDEMAN | 1980(1) | (NUB 41)/NUB 52 | 0.78028 | 0.0018 | 0.231 |  | yes |
| 116 | SPENCER | 1982(1) | NUB 52 | 3.7831 | 0.00836 | 0.221 |  | yes |
| 117 | GWIN | 1984(1) | (NUB 33)/NUB 52 | 0.65996 | 0.00224 | 0.340 |  | yes |
| 118 | GWIN | 1984(1) | (NUB 35)/NUB 52 | 0.64701 | 0.00171 | 0.265 |  | yes |
| 119 | GWIN | 1984(1) | (NUB 39)/NUB 52 | 0.7653 | 0.00197 | 0.257 |  | yes |
| 120 | GWIN | 1984(1) | (NUB 41)/NUB 52 | 0.78423 | 0.00227 | 0.290 |  | yes |
| 121 | HOPKINS | 1963(1) | NUB 52 | 3.7767 | 0.0316 | 0.838 |  | yes |
| 122 | HOPKINS | 1963(1) | (NUB 33)/NUB 52 | 0.65483 | 0.00586 | 0.895 |  | yes |
| 123 | HOPKINS | 1963(1) | (NUB 35)/NUB 52 | 0.64587 | 0.00541 | 0.838 |  | yes |
| 124 | HOPKINS | 1963(1) | (NUB 39)/NUB 52 | 0.75047 | 0.00747 | 0.996 |  | yes |
| 125 | ASPLUND | 1963(1) | NUB 52 | 3.791 | 0.0404 | 1.066 |  | yes |
| 126 | CONDE | 1965(1) | (NUB 35)/NUB 52 | 0.64222 | 0.00544 | 0.847 |  | yes |
| 127 | WHITE | 1968(1) | NUB 52 | 3.8194 | 0.0395 | 1.033 |  | yes |
| 128 | AXTON | A1985(3) | NUB 52 | 3.7547 | 0.0113 | 0.300 |  | yes |
| 129 | COLV/AXT | 1966(1) | NUB 52 | 3.7299 | 0.0301 | 0.806 |  | yes |
| 130 | COLV/ULL | 1965(1) | NUB 52 | 3.7405 | 0.0164 | 0.438 |  | yes |
| 131 | COLVIN | 1965(1) | (NUB 33)/NUB 35 | 1.0204 | 0.00599 | 0.587 |  | yes |
| 132 | COLVIN | 1965(1) | (NUB 39)/NUB 35 | 1.1835 | 0.00799 | 0.675 |  | yes |
| 133 | COLVIN | 1965(1) | (NUB 41)/NUB 35 | 1.2102 | 0.011 | 0.905 |  | yes |
| 134 | COLVIN | 1965(1) | (NUB 35)/NUB 52 | 0.64334 | 0.00304 | 0.472 |  | yes |
| 135 | MATHER | 1964(1) | (NUB 33)/NUB 52 | 0.67093 | 0.00829 | 1.235 |  | yes |
| 136 | MATHER | 1964(1) | (NUB 35)/NUB 52 | 0.64145 | 0.00344 | 0.537 |  | yes |
| 137 | MATHER | 1964(1) | (NUB 39)/NUB 52 | 0.77307 | 0.00867 | 1.122 |  | yes |
| 138 | ALEK-ROV | 1981(1) | NUB 52 | 3.7618 | 0.0182 | 0.483 |  | yes |
| 139 | SMITH | 1984(1) | NUB 52 | 3.7678 | 0.0114 | 0.303 |  | yes |
| 140 | EDWARDS | 1982(1) | NUB 52 | 3.7641 | 0.0268 | 0.711 |  | yes |
| 141 | BOZ-NESH | 1977(1) | NUB 52 | 3.7475 | 0.0217 | 0.580 |  | yes |
| 142 | DEVOLPI | 1972(1) | NUB 52 | 3.7507 | 0.0174 | 0.463 |  | yes |
| 143 | ZHANG | 1981(1) | NUB 52 | 3.7534 | 0.0184 | 0.490 |  | yes |
| 144 | SPIEGEL | 1981(1) | NUB 52 | 3.7828 | 0.0287 | 0.759 |  | yes |
| 145 | SMITH | 1984(1) | ETA 33 | 2.2972 | 0.00804 | 0.350 |  | yes |
| 146 | SMITH | 1984(1) | ETA 35 | 2.0842 | 0.00805 | 0.386 |  | yes |
| 147 | SMITH | 1984(1) | ETA 39 | 2.112 | 0.00803 | 0.380 |  | yes |
| 148 | SMITH | 1984(1) | ETA 41 | 2.169 | 0.00909 | 0.419 |  | yes |
| 149 | MACKLIN | 1960(1) | ETA 33 | 2.3121 | 0.00867 | 0.375 |  | yes |
| 150 | MACKLIN | 1960(1) | ETA 35 | 2.0757 | 0.00793 | 0.382 |  | yes |
| 151 | MACKLIN | 1962(1) | ETA 39 | 2.1163 | 0.018 | 0.851 |  | yes |
| 152 | CABELL | 1968(1) | F1CAB | 273.84 | 13.6 | 4.960 |  |  |
| 153 | CABELL | 1968(1) | F2CAB | 1171.9 | 37.2 | 3.176 |  |  |
| 154 | CABELL | 1968(1) | F3CAB | 3.269 | 0.0907 | 2.776 |  |  |
| 155 | CABELL | 1968(1) | F4CAB | 303.39 | 17.7 | 5.836 |  |  |
| 156 | CABELL | 1968(1) | F5CAB | 3.871 | 0.062 | 1.602 |  |  |
| 157 | BNL 325 | 1973(1) | CA 40 | 289.5 | 1.4 | 0.484 |  |  |
| 158 | BNL 325 | 1973(1) | CA 42 | 18.5 | 0.4 | 2.162 |  |  |
| 159 | BNL325 + WESTCOTT | 1973(1) | GC116 39 | 1.3265 | 0.0803 | 6.056 |  |  |
| 160 | WESTCOTT | 1960(1) | GC116 40 | 1.086 | 0.0217 | 2.000 |  |  |
| 161 | BNL325 + WESTCOTT | 1973(1) | GC116 41 | 1.1085 | 0.0272 | 2.457 |  |  |
| 162 | WESTCOTT | 1960(1) | GC116 42 | 1.1335 | 0.0303 | 2.670 |  |  |
| 163 | BNL325 + WESTCOTT | 1973(1) | GA116 39 | 1.1846 | 0.018 | 1.521 |  |  |
| 164 | WESTCOTT | 1960(1) | GA116 41 | 1.1073 | 0.00907 | 0.819 |  |  |
| 165 | LLOYD | 1982(1) | (ETA 39)/FIS 39 | 0.002782 | 5.76e-05 | 2.072 |  | yes |
| 166 | SPENCER | 1985(3) | (ABS 35)+SCR 35 | 690 | 6.9 | 1.000 |  | yes |
| 167 | SPENCER | 1985(3) | (ABS 39)+SCR 39 | 1025.8 | 10.3 | 1.000 |  | yes |
| 168 | ADAMCHUK | At. En. (R), 65, 434 (1988) | (CA 35)/FIS 35 | 0.169 | 0.00355 | 2.100 |  | yes |
| 169 | WALLNER | PRL 112, 192501 (2014) | CA 35 | 99.5 | 1.99 | 2.000 |  | yes |
| 170 | WALLNER | PRL 112, 192501 (2014) | CA 35 | 101.5 | 2.03 | 2.000 |  | yes |
| 171 | ATOYI | Acta Cryst. 20, 587 (1966) | SCR 35 | 12.3 | 0.5 | 4.065 |  | yes |
| 172 | WILLIS | PRS A274, 122 (1963) | SCR 35 | 12.8 | 1 | 7.812 |  | yes |
| 174 | ARIF | PR A35, 2810 (1987) | SCA 35 | 14 | 0.2 | 1.429 |  | yes |
| 175 | CEULEMANS | Divadeenam compilation 1970 | SCA 35 | 14.6 | 0.5 | 3.425 |  | yes |
| 176 | BAYVAS | Divadeenam compilation 1980 | SCA 35 | 14.3 | 1.5 | 10.489 |  | yes |
| 177 | FOOTE | Divadeenam compilation 1958 | SCR 35 | 15 | 1 | 6.667 |  | yes |
| 178 | OLEKSA | Phys Rev 109, 1645 (1958) | SCA 33 | 13.4 | 0.5 | 3.731 |  | yes |
| 179 | GREEN | JNM 51, 281 (1974) | SCR 33 | 12.6 | 0.7 | 5.556 |  | yes |
| 180 | VERTEBNYI | Divadeenam compilation 1974 | SCA 33 | 13.2 | 0.6 | 4.545 |  | yes |
| 181 | GREEN | JNM 34, 281 (1970) | SCA 39 | 7.7 | 0.6 | 7.792 |  | yes |
| 182 | ATOYI | Acta Cryst. 20, 587 (1966) | SCA 39 | 7.3 | 0.7 | 9.589 |  | yes |
| 183 | ROOF | Acta Cryst. 15, 351 (1962) | SCA 39 | 7.3 | 0.7 | 9.589 |  | yes |
| 184 | PRONYAEV | Internal analysis 2026-04-17 | SCA 41 | 12 | 0.8 | 6.667 |  | yes |
| 185 | ANTONOV | Capote 2026-05-04 list | (ABS 35)+SCA 35 | 695 | 15 | 2.158 |  | yes |
| 186 | GERASIMOV | Capote 2026-05-04 list | (ABS 35)+SCR 35 | 692 | 8 | 1.156 | UPD | yes |
| 187 | LEONARD | Capote 2026-05-04 list (1954) | (ABS 35)+SCR 35 | 701 | 14 | 1.997 | UPD | yes |
| 188 | PSCHENICHNYI | Capote 2026-05-04 list (1978) | (ABS 33)+SCR 33 | 588 | 9 | 1.531 | UPD | yes |
| 189 | HARVEY-1 | Capote 2026-05-04 list (1979) | (ABS 33)+SCR 33 | 588 | 9 | 1.531 | UPD | yes |
| 192 | ANDERSON | Capote 2026-05-04 list (1945) | (ABS 39)+SCR 39 | 1045 | 25 | 2.392 |  | yes |
| 193 | PALEVSKY | Capote 2026-05-04 list (1955) | (ABS 39)+SCR 39 | 1030 | 10 | 0.971 | UPD | yes |
| 195 | TOVESSON | 2010 (X4 14271) | (FIS 39)/FIS 35 | 1.286 | 0.0257 | 2.000 | NEW | yes |
| 196 | GWIN | 1984 (X4 12905) | FIS 39 | 750.2 | 12.8 | 1.700 | NEW | yes |
| 197 | ZHURAVLEV | 1976 (X4 40476) | (FIS 39)/FIS 35 | 1.274 | 0.0255 | 2.000 | NEW | yes |
| 198 | DERUYTTER | 1970 (X4 20143) | FIS 39 | 749.3 | 3.75 | 0.500 | NEW | yes |
| 199 | DERUYTTER | 1972 (X4 20132) | FIS 39 | 748.8 | 7.49 | 1.000 | NEW | yes |
| 200 | STAVISSKII | 1965 (X4 41635) | FIS 39 | 748 | 7.48 | 1.000 | NEW | yes |
| 201 | EGELSTAFF | 1955 (X4 23526) | (FIS 41)/FIS 39 | 1.33 | 0.0399 | 3.000 | NEW | yes |
| 202 | JAFFEY | 1955 (X4 12521) | (FIS 41)/FIS 39 | 1.362 | 0.0232 | 1.700 | NEW | yes |
| 203 | JAMES | 1965 (X4 20940) | FIS 41 | 1014.4 | 20.3 | 2.000 | NEW | yes |
| 204 | WESTON | 1978 (X4 10768) | FIS 41 | 1018.8 | 27.5 | 2.700 | NEW | yes |
| 205 | TOVESSON | 2010 (X4 14271) | (FIS 41)/FIS 35 | 1.73 | 0.0346 | 2.000 | NEW | yes |
| 206 | BERCEANU | 1982 (X4 30750) | (FIS 33)/FIS 35 | 0.892 | 0.0178 | 2.000 | NEW | yes |
| 207 | ZHURAVLEV | 1976 (X4 40476) | (FIS 33)/FIS 35 | 0.908 | 0.0136 | 1.500 | NEW | yes |
| 208 | WENHAI | 1975 (X4 32624) | (FIS 33)/FIS 35 | 0.916 | 0.00916 | 1.000 | NEW | yes |
| 209 | DERUYTTER | 1974 (X4 20411) | FIS 33 | 533.2 | 2.67 | 0.500 | NEW | yes |
| 210 | DERUYTTER | 1971 (X4 20131) | FIS 35 | 586.6 | 4.69 | 0.800 | NEW | yes |
| 211 | GWIN | 1984 (X4 12905) | FIS 35 | 586.1 | 1.76 | 0.300 | NEW | yes |
| 212 | BORCEA | 1973 (X4 30140) | FIS 35 | 586.3 | 7.62 | 1.300 | NEW | yes |
| 213 | AMADUCCI | 2022 (X4 23453) | FIS 35 | 586.2 | 2.93 | 0.500 | NEW | yes |
| 214 | DERUYTTER | 1970 (X4 20143) | FIS 35 | 587.6 | 2.35 | 0.400 | NEW | yes |
| 215 | YE | 1964 (X4 32827) | FIS 35 | 582 | 8.73 | 1.500 | NEW | yes |
| 216 | FRAYSSE | 1965 (X4 20071) | FIS 35 | 584.3 | 8.76 | 1.500 | NEW | yes |
| 217 | SAFFORD | 1959 (X4 12439) | FIS 35 | 580 | 6.96 | 1.200 | NEW | yes |
| 218 | MOORE | 1960 (X4 12341) | (ABS 33)+SCA 33 | 587 | 18 | 3.066 | NEW | yes |
| 219 | LYNN | 1955 (X4 21027) | (ABS 35)+SCR 35 | 704 | 14 | 1.989 | NEW | yes |
| 220 | YOUNG | 1968 (X4 10406) | (ABS 41)+SCR 41 | 1390 | 10 | 0.719 | NEW | yes |

## Decisions (flags F1–F13)

Each open interpretation point from the 2026-06-23 review, resolved with the most plausible reading and recorded here for confirmation by the working group. Full context in the [manifest §F](../references/emails/tnc_update_roberto_20260623/data_change_manifest.md).

**F1 — Pattenden 56 Pu-239 uncertainty.** The fission table gives "1039(5%)" while the dedicated Pu-239 (n,tot) table gives "1039(25)" (≈2.41%). The later, dedicated (n,tot) revision with the local plot-fit is taken as authoritative → **1039 ± 25 b** (No. 75).

**F2 — Leonard 56 Pu-239 (No. 194) — keep or delete.** The Pu-239 (n,tot) table shows an adjusted 1042(25), but the fission email and the local analysis both flag it an outlier → **deleted**.

**F3 — Lynn 55 U-235 (n,tot) — add or not.** Listed as a "NEW set" 704(14) in the consolidated U-235 (n,tot) table, though the local notes advised against adding it (old; large local-vs-integral discrepancy). The consolidated table governs → **added** (No. 219); dissent recorded. Worth confirming.

**F4 — Bollinger 52 U-233 (n,tot) — add or not.** Present in the local working notes but omitted from the consolidated U-233 (n,tot) list (only Moore carried over as NEW DATA) → **not added**.

**F5 — Deruytter 74 FIS(²³³U) = 533.2 — missing uncertainty.** No uncertainty quoted. Assigned **σ = 0.5%** (No. 209), matching the precision of the other Deruytter absolute fission values. **Provisional — confirm the intended value.**

**F6 — U-234(n,γ) = 100.9 b — missing uncertainty.** Only the central value (B80) was given. The existing relative uncertainty **2.086%** is kept (No. 5); remains a fixed input.

**F7 — Watanabe-FIS41 / Deruytter61-FIS35 — update or new.** Both match existing entries by author/year/quantity/EXFOR id and are tagged as re-fits → treated as **updates** to Nos. 108 / 98, not new points. The systematic code on No. 98 (B) is retained.

**F8 — Berceanu 77 (No. 101) "CONTAINS BORCEA 70".** Read as a double-counting warning → **No. 101 kept as-is; Borcea 70 not added separately.**

**F9 — Berceanu U-235 FIS35 589.3 "replace by Borcea 73".** Interpreted literally → **Berceanu 589.3 dropped; Borcea 73 = 586.3 (1.3%) added** (No. 212).

**F10 — Harvey 79 U-233 numbering.** The survivor is labelled "data #189" while the quoted old value (592) matches No. 190. Net intent is one Harvey at 588(9) → **No. 189 set to 588(9), No. 190 deleted** (the survivor is overwritten, so the old 585-vs-592 distinction is immaterial).

**F11 — Gerasimov 62 U-235 uncertainty.** Fission email "692(1.4%)" vs U-235 (n,tot) email "692(8)" (≈1.16%). The dedicated (n,tot) absolute value is used → **692 ± 8 b** (No. 186).

**F12 — Shape measurements.** Bowman, Melkonian, Schrack, Wagemans 79/88 and Mihailescu are shape (relative-to-thermal) measurements → **not added** as absolute points. (This Melkonian ≠ the U-235 (n,tot) entry No. 63, which is retained.)

**F13 — Classification of the new transmission points.** "Keep current SCA/SCR" does not cover brand-new entries. Young (No. 220) is explicitly ABS+SCR. **Moore (No. 218) assigned ABS+SCA** (consistent with the local U-233 working assumption) and **Lynn (No. 219) assigned ABS+SCR** (era convention) — both provisional. Note: this only affects the favoured Step-5-style fit; under a Step-6-style "all ABS+SCR → ABS+SCA" conversion the choice is moot.

## Files Produced (Round 3)

  * `axton1986/axton1986_data_capote_round3.json` — round-3 database (205 measurements), regenerated by `evaluation/build_round3_json.py`
  * `evaluation/build_round3_json.py` — auditable generator (applies the manifest to the Step-5 base)
  * `evaluation/benchmark_capote_round3.py` — round-3 fit benchmark
  * `evaluation/build_round3_report.py` — this report generator
  * `evaluation/round3_summary.json` — machine-readable round-3 results (+ Step-5 reference)
  * `references/emails/tnc_update_roberto_20260623/` — paraphrased discussions + datapoint-level manifest with decision register
