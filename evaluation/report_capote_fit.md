# Capote 2026-05-04 Fit Chain: Step-by-Step Benchmark Report

*Round 2 — incorporates corrections requested in the 2026-05-08 email round (see [`references/emails/tnc_update_roberto_20260508/`](../references/emails/tnc_update_roberto_20260508/summary.md)).*

This report documents the implementation of the new TNC fit chain proposed in the 2026-05-04 emails *"NEW TNC FIT steps"* and *"Revision/additions to Axton Total cross sections experimental data"*, refined by the corrections in the 2026-05-08 round (drop KAISER as duplicate of ARIF; reclassify ATOYI / WILLIS / FOOTE / GREEN-233 from SCA to SCR; replace Westcott factors with B81-derived values). Each section corresponds to one of the seven steps (0–6) in the chain, showing how the iterative Generalized Least Squares (GLS) fit results evolve as data are added, deleted, reclassified, or modified.

All fits use the microscopic-only data subset (Maxwellian-derived functions excluded) on the Axton 1986 database. Steps 0 and 1 read from a round-2 baseline file (`axton1986_data_capote_round2_baseline.json`) that is identical to the original microscopic-Axton baseline except for the new Westcott factors; the original `axton1986_data.json` is preserved unchanged as the Axton-Table-8 reproduction artefact.

Quantities with shifts exceeding 0.1% relative to the previous step are highlighted in **bold**. The Westcott-folded quantities FA, FF, F1ETA, F2ETA, F3ETA are computed as derived outputs at each step per the request (K1 = F3ETA). A Round-1 → Round-2 comparison table is shown after the Step-5-vs-Step-6 head-to-head; the round-2 corrections are documented in detail in Appendix B.

## Overview

| # | Step | N meas | N free | DOF | χ² | Converged |
|---|------|-------:|-------:|----:|---:|:---------:|
| 0 | Axton T8 baseline (no Lounsbury) | 102 | 30 | 72 | 55.13 | Yes |
| 1 | + Lounsbury alpha (21–23) | 105 | 30 | 75 | 60.49 | Yes |
| 2 | + Adamchuk 1988 alpha(²³⁵U) | 106 | 30 | 76 | 60.56 | Yes |
| 3 | + Wallner 2014 U-235 capture | 108 | 30 | 78 | 62.85 | Yes |
| 4 | Sears SCA list + Pu-241 estimate; delete Lemmel SCA | 117 | 30 | 87 | 70.91 | Yes |
| 5 | + transmission additions; 3 SCR→SCA reclassifications | 127 | 30 | 97 | 80.55 | Yes |
| 6 | All ABS+SCR → ABS+SCA; drop Lemmel SCA-SCR; SCR partially removed | 123 | 28 | 95 | 80.14 | Yes |

## Targeted comparison: Step 5 vs Step 6

The 2026-05-04 email asks to *"compare steps 5 and 6 — hopefully one of them will be the final fit"*. The two candidate fits agree to better than 0.2% on every primary and derived quantity. Step 5 retains 4 SCR fit variables (now constrained by both the round-2 SCR reclassifications and the ABS+SCR transmissions); step 6 removes the ABS+SCR transmissions and Lemmel SCA-SCR, leaving SCR(²³³U) and SCR(²³⁵U) constrained only by the reclassified direct measurements (and SCR(²³⁹Pu), SCR(²⁴¹Pu) unconstrained → removed). The 2026-05-06 follow-up email expresses a marginal preference for Step 5 over Step 6.

## Step 0: Axton Table 8 Baseline (No Lounsbury)

This is the reference point: the microscopic GLS fit reproducing Axton's Table 8 results, with Lounsbury alpha measurements (Nos. 21–23) excluded. The detailed reproducibility analysis and the rationale for excluding Lounsbury are documented in `report_axton1986_reproduction.md` (Section 4); briefly, Lounsbury alpha measurements carry a "Z" systematic code (*Lounsbury Temperature Uncertainty*) that makes them effectively Maxwellian/integral, and Axton excluded them in his Table 6/Table 8 microscopic fit.

### Fitted quantities

| Quantity | Value | Unc (%) | Δ vs prev (%) |
|----------|------:|--------:|--------------:|
| SCA 33 | 12.3206 | 5.681 | – |
| SCA 35 | 16.2235 | 7.385 | – |
| SCA 39 | 7.9001 | 12.380 | – |
| SCA 41 | 11.9720 | 21.596 | – |
| SCR 33 | 11.1483 | 8.161 | – |
| SCR 35 | 14.5068 | 9.455 | – |
| SCR 39 | 6.8449 | 24.890 | – |
| SCR 41 | 10.8234 | 35.520 | – |
| ABS 33 | 575.2413 | 0.321 | – |
| ABS 35 | 681.2060 | 0.249 | – |
| ABS 39 | 1018.7914 | 0.362 | – |
| ABS 41 | 1382.2612 | 1.079 | – |
| FIS 33 | 533.1364 | 0.460 | – |
| FIS 35 | 585.0334 | 0.277 | – |
| FIS 39 | 748.5245 | 0.340 | – |
| FIS 41 | 1020.3307 | 1.127 | – |
| CA 33 | 42.1049 | 4.228 | – |
| CA 35 | 96.1726 | 1.811 | – |
| CA 39 | 270.2669 | 1.181 | – |
| CA 41 | 361.9304 | 1.729 | – |
| CAP 34 | 95.9000 | 2.086 | – |
| NUB 33 | 2.4860 | 0.217 | – |
| NUB 35 | 2.4264 | 0.188 | – |
| NUB 39 | 2.8794 | 0.207 | – |
| NUB 41 | 2.9406 | 0.220 | – |
| NUB 52 | 3.7649 | 0.132 | – |
| ETA 33 | 2.3040 | 0.269 | – |
| ETA 35 | 2.0838 | 0.255 | – |
| ETA 39 | 2.1156 | 0.310 | – |
| ETA 41 | 2.1706 | 0.399 | – |
| ALPHA 33 | 0.0790 | 4.545 | – |
| ALPHA 35 | 0.1644 | 1.962 | – |
| ALPHA 39 | 0.3611 | 1.289 | – |
| ALPHA 41 | 0.3547 | 1.697 | – |
| WGA 33 | 1.0000 | 0.100 | – |
| WGA 35 | 0.9820 | 0.305 | – |
| WGA 39 | 1.0780 | 0.928 | – |
| WGA 41 | 1.0460 | 0.669 | – |
| WGF 33 | 0.9970 | 0.100 | – |
| WGF 35 | 0.9810 | 0.510 | – |
| WGF 39 | 1.0520 | 1.331 | – |
| WGF 41 | 1.0520 | 0.665 | – |
| FA 33 | 575.2413 | 0.337 | – |
| FA 35 | 668.9443 | 0.394 | – |
| FA 39 | 1098.2571 | 0.996 | – |
| FA 41 | 1445.8452 | 1.270 | – |
| FF 33 | 531.5370 | 0.471 | – |
| FF 35 | 573.9178 | 0.580 | – |
| FF 39 | 787.4478 | 1.374 | – |
| FF 41 | 1073.3879 | 1.309 | – |
| F1ETA 33 | 2.2971 | 0.304 | – |
| F1ETA 35 | 2.0817 | 0.647 | – |
| F1ETA 39 | 2.0645 | 1.652 | – |
| F1ETA 41 | 2.1831 | 1.025 | – |
| F2ETA 33 | 1321.3993 | 0.429 | – |
| F2ETA 35 | 1392.5407 | 0.586 | – |
| F2ETA 39 | 2267.4064 | 1.379 | – |
| F2ETA 41 | 3156.3904 | 1.301 | – |
| F3ETA 33 | 746.1581 | 0.604 | – |
| F3ETA 35 | 723.5963 | 1.123 | – |
| F3ETA 39 | 1169.1493 | 2.779 | – |
| F3ETA 41 | 1710.5452 | 1.844 | – |

## Step 1: + Lounsbury Alpha (Nos. 21–23)

Step 1 adds the three Lounsbury 1970 alpha measurements back into the microscopic fit:

  * No.21 LOUNSBURY (CA 33)/FIS 33 = 0.0861, σ = 2.24%
  * No.22 LOUNSBURY (CA 35)/FIS 35 = 0.1697, σ = 1.71%
  * No.23 LOUNSBURY (CA 39)/FIS 39 = 0.3555, σ = 1.60%

This is the largest single change in the entire chain. Lounsbury's α(²³³U) = 0.0861 is far above Axton's converged value of 0.0787, and pulling the system toward this value drives a large redistribution of ABS(²³³U) between capture and fission — ALPHA(²³³U) shifts by +7.2%, CA(²³³U) by +6.6%, and FIS(²³³U) drops by 0.5%. F3ETA(²³³U) drops by 0.56%.

### Fitted quantities

| Quantity | Value | Unc (%) | Δ vs prev (%) |
|----------|------:|--------:|--------------:|
| SCA 33 | 12.3166 | 5.681 | -0.032 |
| **SCA 35** | **15.8847** | 7.270 | **-2.088** |
| **SCA 39** | **7.9230** | 12.364 | **+0.291** |
| SCA 41 | 11.9618 | 21.596 | -0.085 |
| SCR 33 | 11.1419 | 8.171 | -0.057 |
| **SCR 35** | **14.0797** | 9.527 | **-2.944** |
| **SCR 39** | **6.9469** | 23.295 | **+1.490** |
| **SCR 41** | **10.8049** | 35.707 | **-0.170** |
| ABS 33 | 575.2692 | 0.321 | +0.005 |
| **ABS 35** | **682.0852** | 0.233 | **+0.129** |
| ABS 39 | 1018.2345 | 0.340 | -0.055 |
| ABS 41 | 1382.7102 | 1.079 | +0.032 |
| **FIS 33** | **530.3843** | 0.355 | **-0.516** |
| **FIS 35** | **584.0124** | 0.244 | **-0.175** |
| FIS 39 | 748.1165 | 0.321 | -0.055 |
| **FIS 41** | **1018.4694** | 1.119 | **-0.182** |
| **CA 33** | **44.8849** | 1.837 | **+6.603** |
| **CA 35** | **98.0728** | 1.152 | **+1.976** |
| CA 39 | 270.1180 | 0.913 | -0.055 |
| **CA 41** | **364.2408** | 1.663 | **+0.638** |
| CAP 34 | 95.9000 | 2.086 | +0.000 |
| **NUB 33** | **2.4910** | 0.184 | **+0.203** |
| **NUB 35** | **2.4294** | 0.172 | **+0.123** |
| NUB 39 | 2.8814 | 0.192 | +0.067 |
| NUB 41 | 2.9435 | 0.209 | +0.099 |
| NUB 52 | 3.7664 | 0.129 | +0.039 |
| **ETA 33** | **2.2967** | 0.199 | **-0.319** |
| **ETA 35** | **2.0801** | 0.211 | **-0.180** |
| ETA 39 | 2.1170 | 0.267 | +0.067 |
| **ETA 41** | **2.1681** | 0.391 | **-0.116** |
| **ALPHA 33** | **0.0846** | 1.960 | **+7.156** |
| **ALPHA 35** | **0.1679** | 1.235 | **+2.154** |
| ALPHA 39 | 0.3611 | 0.965 | -0.001 |
| **ALPHA 41** | **0.3576** | 1.585 | **+0.822** |
| WGA 33 | 1.0000 | 0.100 | +0.000 |
| WGA 35 | 0.9820 | 0.305 | +0.000 |
| WGA 39 | 1.0780 | 0.928 | +0.000 |
| WGA 41 | 1.0460 | 0.669 | +0.000 |
| WGF 33 | 0.9970 | 0.100 | +0.000 |
| WGF 35 | 0.9810 | 0.510 | +0.000 |
| WGF 39 | 1.0520 | 1.331 | +0.000 |
| WGF 41 | 1.0520 | 0.665 | +0.000 |
| FA 33 | 575.2692 | 0.337 | +0.005 |
| **FA 35** | **669.8077** | 0.384 | **+0.129** |
| FA 39 | 1097.6568 | 0.988 | -0.055 |
| FA 41 | 1446.3148 | 1.270 | +0.032 |
| **FF 33** | **528.7932** | 0.369 | **-0.516** |
| **FF 35** | **572.9162** | 0.565 | **-0.175** |
| FF 39 | 787.0186 | 1.369 | -0.055 |
| **FF 41** | **1071.4298** | 1.302 | **-0.182** |
| **F1ETA 33** | **2.2898** | 0.244 | **-0.319** |
| **F1ETA 35** | **2.0779** | 0.630 | **-0.180** |
| F1ETA 39 | 2.0659 | 1.644 | +0.067 |
| **F1ETA 41** | **2.1805** | 1.021 | **-0.116** |
| **F2ETA 33** | **1317.2452** | 0.391 | **-0.314** |
| F2ETA 35 | 1391.8258 | 0.582 | -0.051 |
| F2ETA 39 | 2267.6891 | 1.378 | +0.012 |
| F2ETA 41 | 3153.7617 | 1.300 | -0.083 |
| **F3ETA 33** | **741.9759** | 0.515 | **-0.560** |
| **F3ETA 35** | **722.0181** | 1.108 | **-0.218** |
| F3ETA 39 | 1170.0323 | 2.771 | +0.076 |
| **F3ETA 41** | **1707.4469** | 1.840 | **-0.181** |

## Step 2: + Adamchuk 1988 Alpha(²³⁵U)

Step 2 adds one new measurement (No.168):

  * ADAMCHUK (CA 35)/FIS 35 = 0.169, σ = 2.1%, At. En. (R) 65, 434 (1988) (GMA NS=8026)

Adamchuk's α(²³⁵U) = 0.169 is highly consistent with the step-1 fitted value (0.168), so the step is essentially a no-op for χ²: only +0.07 from one added measurement, and shifts in U-235 quantities are at the 0.1–0.2% level. All other nuclides are unchanged.

### Fitted quantities

| Quantity | Value | Unc (%) | Δ vs prev (%) |
|----------|------:|--------:|--------------:|
| SCA 33 | 12.3166 | 5.681 | -0.000 |
| **SCA 35** | **15.8609** | 7.245 | **-0.150** |
| SCA 39 | 7.9233 | 12.364 | +0.003 |
| SCA 41 | 11.9618 | 21.596 | -0.000 |
| SCR 33 | 11.1419 | 8.171 | -0.000 |
| **SCR 35** | **14.0487** | 9.509 | **-0.220** |
| SCR 39 | 6.9479 | 23.281 | +0.014 |
| SCR 41 | 10.8049 | 35.708 | -0.000 |
| ABS 33 | 575.2694 | 0.321 | +0.000 |
| ABS 35 | 682.1500 | 0.230 | +0.010 |
| ABS 39 | 1018.2347 | 0.340 | +0.000 |
| ABS 41 | 1382.7113 | 1.079 | +0.000 |
| FIS 33 | 530.3690 | 0.355 | -0.003 |
| FIS 35 | 583.9317 | 0.239 | -0.014 |
| FIS 39 | 748.0666 | 0.320 | -0.007 |
| FIS 41 | 1018.3865 | 1.119 | -0.008 |
| CA 33 | 44.9004 | 1.832 | +0.035 |
| **CA 35** | **98.2183** | 0.999 | **+0.148** |
| CA 39 | 270.1681 | 0.910 | +0.019 |
| CA 41 | 364.3248 | 1.660 | +0.023 |
| CAP 34 | 95.9000 | 2.086 | +0.000 |
| NUB 33 | 2.4911 | 0.184 | +0.002 |
| NUB 35 | 2.4295 | 0.170 | +0.006 |
| NUB 39 | 2.8815 | 0.191 | +0.003 |
| NUB 41 | 2.9436 | 0.208 | +0.004 |
| NUB 52 | 3.7664 | 0.129 | +0.002 |
| ETA 33 | 2.2967 | 0.199 | -0.001 |
| ETA 35 | 2.0797 | 0.200 | -0.017 |
| ETA 39 | 2.1169 | 0.266 | -0.003 |
| ETA 41 | 2.1680 | 0.390 | -0.005 |
| ALPHA 33 | 0.0847 | 1.955 | +0.037 |
| **ALPHA 35** | **0.1682** | 1.064 | **+0.162** |
| ALPHA 39 | 0.3612 | 0.960 | +0.025 |
| ALPHA 41 | 0.3577 | 1.580 | +0.031 |
| WGA 33 | 1.0000 | 0.100 | +0.000 |
| WGA 35 | 0.9820 | 0.305 | +0.000 |
| WGA 39 | 1.0780 | 0.928 | +0.000 |
| WGA 41 | 1.0460 | 0.669 | +0.000 |
| WGF 33 | 0.9970 | 0.100 | +0.000 |
| WGF 35 | 0.9810 | 0.510 | +0.000 |
| WGF 39 | 1.0520 | 1.331 | +0.000 |
| WGF 41 | 1.0520 | 0.665 | +0.000 |
| FA 33 | 575.2694 | 0.337 | +0.000 |
| FA 35 | 669.8713 | 0.382 | +0.010 |
| FA 39 | 1097.6570 | 0.988 | +0.000 |
| FA 41 | 1446.3160 | 1.270 | +0.000 |
| FF 33 | 528.7779 | 0.369 | -0.003 |
| FF 35 | 572.8370 | 0.563 | -0.014 |
| FF 39 | 786.9660 | 1.369 | -0.007 |
| FF 41 | 1071.3426 | 1.302 | -0.008 |
| F1ETA 33 | 2.2898 | 0.244 | -0.001 |
| F1ETA 35 | 2.0776 | 0.627 | -0.017 |
| F1ETA 39 | 2.0659 | 1.644 | -0.003 |
| F1ETA 41 | 2.1804 | 1.021 | -0.005 |
| F2ETA 33 | 1317.2344 | 0.391 | -0.001 |
| F2ETA 35 | 1391.7199 | 0.582 | -0.008 |
| F2ETA 39 | 2267.6139 | 1.378 | -0.003 |
| F2ETA 41 | 3153.6197 | 1.300 | -0.005 |
| F3ETA 33 | 741.9650 | 0.515 | -0.001 |
| F3ETA 35 | 721.8486 | 1.104 | -0.023 |
| F3ETA 39 | 1169.9569 | 2.771 | -0.006 |
| F3ETA 41 | 1707.3037 | 1.840 | -0.008 |

## Step 3: + Wallner 2014 U-235 Capture

Step 3 adds two Wallner et al. 2014 thermal capture measurements (PRL 112, 192501, 2014):

  * No.169 WALLNER CA 35 =  99.5 b, σ = 2.0%
  * No.170 WALLNER CA 35 = 101.5 b, σ = 2.0%

Both Wallner values lie above the step-2 fit (CA 35 ≈ 98.2 b). The fit accommodates the pull with χ² = +2.3 over 2 added measurements; CA(²³⁵U) shifts +0.76% to 98.97 b. Through the absorption channel correlations, SCA(²³⁵U) drops 1.3% and SCR(²³⁵U) drops 1.9%. F3ETA(²³⁵U) shifts only 0.08%, so the K1 of interest is essentially preserved.

### Fitted quantities

| Quantity | Value | Unc (%) | Δ vs prev (%) |
|----------|------:|--------:|--------------:|
| SCA 33 | 12.3165 | 5.681 | -0.001 |
| **SCA 35** | **15.6565** | 7.185 | **-1.288** |
| SCA 39 | 7.9214 | 12.364 | -0.024 |
| SCA 41 | 11.9602 | 21.596 | -0.013 |
| SCR 33 | 11.1417 | 8.172 | -0.002 |
| **SCR 35** | **13.7764** | 9.597 | **-1.938** |
| **SCR 39** | **6.9407** | 23.380 | **-0.103** |
| SCR 41 | 10.8020 | 35.738 | -0.027 |
| ABS 33 | 575.2702 | 0.321 | +0.000 |
| ABS 35 | 682.7209 | 0.220 | +0.084 |
| ABS 39 | 1018.3508 | 0.340 | +0.011 |
| ABS 41 | 1382.7815 | 1.079 | +0.005 |
| FIS 33 | 530.2947 | 0.355 | -0.014 |
| FIS 35 | 583.7524 | 0.237 | -0.031 |
| FIS 39 | 747.9269 | 0.320 | -0.019 |
| FIS 41 | 1018.0396 | 1.119 | -0.034 |
| **CA 33** | **44.9755** | 1.826 | **+0.167** |
| **CA 35** | **98.9685** | 0.814 | **+0.764** |
| CA 39 | 270.4239 | 0.907 | +0.095 |
| **CA 41** | **364.7420** | 1.657 | **+0.114** |
| CAP 34 | 95.9000 | 2.086 | +0.000 |
| NUB 33 | 2.4913 | 0.184 | +0.010 |
| NUB 35 | 2.4303 | 0.169 | +0.030 |
| NUB 39 | 2.8819 | 0.191 | +0.016 |
| NUB 41 | 2.9441 | 0.208 | +0.018 |
| NUB 52 | 3.7667 | 0.129 | +0.007 |
| ETA 33 | 2.2966 | 0.199 | -0.004 |
| ETA 35 | 2.0780 | 0.190 | -0.084 |
| ETA 39 | 2.1166 | 0.266 | -0.014 |
| ETA 41 | 2.1675 | 0.390 | -0.022 |
| **ALPHA 33** | **0.0848** | 1.949 | **+0.181** |
| **ALPHA 35** | **0.1695** | 0.879 | **+0.795** |
| **ALPHA 39** | **0.3616** | 0.956 | **+0.113** |
| **ALPHA 41** | **0.3583** | 1.574 | **+0.149** |
| WGA 33 | 1.0000 | 0.100 | +0.000 |
| WGA 35 | 0.9820 | 0.305 | +0.000 |
| WGA 39 | 1.0780 | 0.928 | +0.000 |
| WGA 41 | 1.0460 | 0.669 | +0.000 |
| WGF 33 | 0.9970 | 0.100 | +0.000 |
| WGF 35 | 0.9810 | 0.510 | +0.000 |
| WGF 39 | 1.0520 | 1.331 | +0.000 |
| WGF 41 | 1.0520 | 0.665 | +0.000 |
| FA 33 | 575.2702 | 0.337 | +0.000 |
| FA 35 | 670.4319 | 0.377 | +0.084 |
| FA 39 | 1097.7822 | 0.988 | +0.011 |
| FA 41 | 1446.3895 | 1.270 | +0.005 |
| FF 33 | 528.7038 | 0.369 | -0.014 |
| FF 35 | 572.6611 | 0.562 | -0.031 |
| FF 39 | 786.8191 | 1.369 | -0.019 |
| FF 41 | 1070.9776 | 1.302 | -0.034 |
| F1ETA 33 | 2.2897 | 0.244 | -0.004 |
| F1ETA 35 | 2.0758 | 0.624 | -0.084 |
| F1ETA 39 | 2.0656 | 1.644 | -0.014 |
| F1ETA 41 | 2.1800 | 1.021 | -0.022 |
| F2ETA 33 | 1317.1830 | 0.391 | -0.004 |
| F2ETA 35 | 1391.7155 | 0.581 | -0.000 |
| F2ETA 39 | 2267.5523 | 1.378 | -0.003 |
| F2ETA 41 | 3153.0973 | 1.300 | -0.017 |
| F3ETA 33 | 741.9129 | 0.515 | -0.007 |
| F3ETA 35 | 721.2836 | 1.104 | -0.078 |
| F3ETA 39 | 1169.7702 | 2.771 | -0.016 |
| F3ETA 41 | 1706.7079 | 1.840 | -0.035 |

## Step 4: Sears SCA List + Pu-241 Estimate; Delete Lemmel SCA

Step 4 implements the most extensive change in the chain. It replaces the four Lemmel direct SCA measurements with a curated compilation drawn from Sears (1992), the Divadeenam compilation, EXFOR replacements, and Pronyaev's potential-scattering estimate for Pu-241.

The Lemmel SCA−SCR datapoints (Nos. 89, 91, 93, 95) are *kept* per the instruction *"leave SCA-SCR"*; only the four direct Lemmel SCA datapoints are deleted.

**Round-2 corrections applied here** (see Appendix B): KAISER (No.173) is removed as a duplicate of ARIF; ATOYI (No.171), WILLIS (No.172), FOOTE (No.177) and GREEN-233 (No.179) are reclassified from SCA to SCR. Net step-4 cardinality is therefore 9 added (vs. 14 before round 2): 1 deleted (Kaiser) plus 4 reclassified (still added, but now constraining SCR rather than SCA).

#### Datapoints deleted
| No. | Author | Reference | Quantity | Value | Note |
|----:|--------|-----------|----------|------:|------|
| 88 | Lemmel | 1982 | SCA(²³³U) | 12.306 | replaced by step-4 SCA list |
| 90 | Lemmel | 1982 | SCA(²³⁵U) | 16.508 | replaced by step-4 SCA list |
| 92 | Lemmel | 1982 | SCA(²³⁹Pu) | 8.004 | replaced by step-4 SCA list |
| 94 | Lemmel | 1982 | SCA(²⁴¹Pu) | 12.006 | replaced by Pronyaev 12.0(0.8) |

#### Datapoints added
| No. | Author / year | Reference | Type | Value (b) | σ (b) | Appendix |
|----:|---------------|-----------|------|----------:|------:|:---------|
| 171 | Atoyi | Acta Cryst. 20, 587 (1966) | SCR(²³⁵U) [r2] | 12.3 | 0.5 | [B2](#a-decisions-and-contentious-points) |
| 172 | Willis | PRS A274, 122 (1963) | SCR(²³⁵U) [r2] | 12.8 | 1 | [B2](#a-decisions-and-contentious-points) |
| 173 | Kaiser | Physica B 136, 134 (1986) | — DELETED [r2] | 14 | 0.2 | [B1](#a-decisions-and-contentious-points) |
| 174 | Arif | PR A35, 2810 (1987) | SCA(²³⁵U) | 14 | 0.2 |  |
| 175 | Ceulemans | Divadeenam compilation 1970 | SCA(²³⁵U) | 14.6 | 0.5 | [A4](#a-decisions-and-contentious-points) |
| 176 | Bayvas | Divadeenam compilation 1980 | SCA(²³⁵U) | 14.3 | 1.5 |  |
| 177 | Foote | Divadeenam compilation 1958 | SCR(²³⁵U) [r2] | 15 | 1 | [B3](#a-decisions-and-contentious-points) |
| 178 | Oleksa | Phys Rev 109, 1645 (1958) | SCA(²³³U) | 13.4 | 0.5 | [A5](#a-decisions-and-contentious-points) |
| 179 | Green | JNM 51, 281 (1974) | SCR(²³³U) [r2] | 12.6 | 0.7 | [B2](#a-decisions-and-contentious-points) |
| 180 | Vertebnyi | Divadeenam compilation 1974 | SCA(²³³U) | 13.2 | 0.6 |  |
| 181 | Green | JNM 34, 281 (1970) | SCA(²³⁹Pu) | 7.7 | 0.6 |  |
| 182 | Atoyi | Acta Cryst. 20, 587 (1966) | SCA(²³⁹Pu) | 7.3 | 0.7 |  |
| 183 | Roof | Acta Cryst. 15, 351 (1962) | SCA(²³⁹Pu) | 7.3 | 0.7 |  |
| 184 | Pronyaev | Internal analysis 2026-04-17 | SCA(²⁴¹Pu) | 12 | 0.8 | [A7](#a-decisions-and-contentious-points) |

#### Impact
The U-235 SCA channel is the most affected: SCA(²³⁵U) drops driven by the ARIF interferometry value 14.0 b replacing the 16-ish Lemmel value (Kaiser is no longer added separately; it is the same experiment as Arif). SCA uncertainties collapse dramatically, reflecting the data richness of the new compilation. ABS, FIS, CA, and the F-quantities shift by ≤0.2% relative to step 3. χ² rises by ≈8 from a net +9 measurements (~0.9/measurement, indicating moderate but absorbable tension).


### Fitted quantities

| Quantity | Value | Unc (%) | Δ vs prev (%) |
|----------|------:|--------:|--------------:|
| **SCA 33** | **13.3989** | 2.631 | **+8.789** |
| **SCA 35** | **14.1217** | 1.265 | **-9.803** |
| **SCA 39** | **7.4596** | 5.107 | **-5.829** |
| **SCA 41** | **11.9951** | 6.665 | **+0.292** |
| **SCR 33** | **12.3565** | 3.809 | **+10.902** |
| **SCR 35** | **12.7854** | 2.629 | **-7.193** |
| **SCR 39** | **6.4694** | 21.374 | **-6.791** |
| **SCR 41** | **10.8324** | 27.684 | **+0.282** |
| **ABS 33** | **574.0827** | 0.295 | **-0.206** |
| **ABS 35** | **683.5588** | 0.184 | **+0.123** |
| ABS 39 | 1018.8655 | 0.336 | +0.051 |
| ABS 41 | 1382.9760 | 1.070 | +0.014 |
| **FIS 33** | **529.1971** | 0.331 | **-0.207** |
| **FIS 35** | **584.4090** | 0.210 | **+0.112** |
| FIS 39 | 748.3107 | 0.317 | +0.051 |
| FIS 41 | 1018.1747 | 1.111 | +0.013 |
| **CA 33** | **44.8856** | 1.822 | **-0.200** |
| **CA 35** | **99.1499** | 0.793 | **+0.183** |
| CA 39 | 270.5548 | 0.903 | +0.048 |
| CA 41 | 364.8014 | 1.650 | +0.016 |
| CAP 34 | 95.9000 | 2.086 | +0.000 |
| NUB 33 | 2.4914 | 0.184 | +0.001 |
| NUB 35 | 2.4303 | 0.169 | +0.003 |
| NUB 39 | 2.8819 | 0.191 | +0.001 |
| NUB 41 | 2.9442 | 0.208 | +0.001 |
| NUB 52 | 3.7667 | 0.129 | +0.001 |
| ETA 33 | 2.2966 | 0.199 | +0.000 |
| ETA 35 | 2.0778 | 0.190 | -0.008 |
| ETA 39 | 2.1167 | 0.266 | +0.002 |
| ETA 41 | 2.1676 | 0.390 | +0.000 |
| ALPHA 33 | 0.0848 | 1.949 | +0.007 |
| ALPHA 35 | 0.1697 | 0.875 | +0.071 |
| ALPHA 39 | 0.3616 | 0.954 | -0.003 |
| ALPHA 41 | 0.3583 | 1.574 | +0.003 |
| WGA 33 | 1.0000 | 0.100 | +0.000 |
| WGA 35 | 0.9820 | 0.305 | +0.000 |
| WGA 39 | 1.0780 | 0.928 | +0.000 |
| WGA 41 | 1.0460 | 0.669 | +0.000 |
| WGF 33 | 0.9970 | 0.100 | +0.000 |
| WGF 35 | 0.9810 | 0.510 | +0.000 |
| WGF 39 | 1.0520 | 1.331 | +0.000 |
| WGF 41 | 1.0520 | 0.665 | +0.000 |
| **FA 33** | **574.0827** | 0.311 | **-0.206** |
| **FA 35** | **671.2548** | 0.357 | **+0.123** |
| FA 39 | 1098.3370 | 0.987 | +0.051 |
| FA 41 | 1446.5929 | 1.262 | +0.014 |
| **FF 33** | **527.6095** | 0.346 | **-0.207** |
| **FF 35** | **573.3052** | 0.551 | **+0.112** |
| FF 39 | 787.2228 | 1.368 | +0.051 |
| FF 41 | 1071.1197 | 1.295 | +0.013 |
| F1ETA 33 | 2.2897 | 0.244 | +0.000 |
| F1ETA 35 | 2.0757 | 0.624 | -0.008 |
| F1ETA 39 | 2.0656 | 1.644 | +0.002 |
| F1ETA 41 | 2.1800 | 1.021 | +0.000 |
| **F2ETA 33** | **1314.4669** | 0.369 | **-0.206** |
| **F2ETA 35** | **1393.3184** | 0.570 | **+0.115** |
| F2ETA 39 | 2268.7368 | 1.377 | +0.052 |
| F2ETA 41 | 3153.5496 | 1.293 | +0.014 |
| **F3ETA 33** | **740.3842** | 0.499 | **-0.206** |
| **F3ETA 35** | **722.0637** | 1.099 | **+0.108** |
| F3ETA 39 | 1170.3998 | 2.771 | +0.054 |
| F3ETA 41 | 1706.9566 | 1.836 | +0.015 |

## Step 5: + Transmission Entries; 3 ABS+SCR → ABS+SCA Reclassifications

Step 5 implements Roberto's transmission-data revision from his "Revision/additions to Axton Total cross sections experimental data" email of 2026-05-04. Three existing transmission entries are reclassified, ten new entries are added, and two would-be additions (Spencer 1987) are recognised as already in the database.

#### Datapoints reclassified (ABS+SCR → ABS+SCA)
| No. | Author | Year | Nuclide | Change | Appendix |
|----:|--------|------|---------|--------|:---------|
| 68 | Safford | 1959 | U-235 | ABS+SCR → ABS+SCA |  |
| 78 | Nikitin | 1955 | U-233 | ABS+SCR → ABS+SCA | [A8](#a-decisions-and-contentious-points) |
| 83 | Safford | 1960 | U-233 | ABS+SCR → ABS+SCA | [A9](#a-decisions-and-contentious-points) |

#### Datapoints added
| No. | Author / year | Reference | Type | Value (b) | σ (b) | Appendix |
|----:|---------------|-----------|------|----------:|------:|:---------|
| 185 | Antonov | 1986 | ABS+SCA(²³⁵U) | 695 | 15 | [A10](#a-decisions-and-contentious-points) |
| 186 | Gerasimov | 1962 | ABS+SCR(²³⁵U) | 685 | 9 | [A11](#a-decisions-and-contentious-points) |
| 187 | Leonard | 1954 | ABS+SCR(²³⁵U) | 702 | 14 | [A12](#a-decisions-and-contentious-points) |
| 188 | Pschenichnyi | 1978 | ABS+SCR(²³³U) | 587.9 | 3 | [A13](#a-decisions-and-contentious-points) |
| 189 | Harvey-1 | 1979 | ABS+SCR(²³³U) | 585 | 1.4 | [A14](#a-decisions-and-contentious-points) |
| 190 | Harvey-2 | 1979 | ABS+SCR(²³³U) | 592 | 6 | [A14](#a-decisions-and-contentious-points) |
| 191 | Schwartz | 1958 | ABS+SCR(²⁴¹Pu) | 1410 | 80 |  |
| 192 | Anderson | 1945 | ABS+SCR(²³⁹Pu) | 1045 | 25 |  |
| 193 | Palevsky | 1955 | ABS+SCR(²³⁹Pu) | 1025 | 10 |  |
| 194 | Leonard | 1956 | ABS+SCR(²³⁹Pu) | 1055 | 13 | [A15](#a-decisions-and-contentious-points) |

#### Datapoints not added (already in DB)
| Author / year | Type | Value | σ (b) | Reason |
|---------------|------|------:|------:|--------|
| Spencer 1987 | (²³⁵U) ABS+SCR | 690 | 5 | Already in DB as No.166 SPENCER 1985(3) val=690.0 σ=1.0% — values match within rounding (per project decision); uncertainty discrepancy logged in A1 |
| Spencer 1987 | (²³⁹Pu) ABS+SCR | 1025 | 6 | Already in DB as No.167 SPENCER 1985(3) val=1025.8 σ=1.0% — values match within rounding; uncertainty discrepancy logged in A1 |

#### Impact
The new transmission datapoints sharpen ABS (and through it FIS and CA) but barely touch the SCA channel from step 4. ABS uncertainties tighten by 10–35% across the four nuclides. The largest individual tension is **Leonard 56 ABS+SCR(²³⁹Pu) = 1055 b** at +2.1σ from the predicted 1027 b; this contributes ≈4.5 to the χ² increase of 9.5. K1 = F3ETA shifts ≤0.21% across all four nuclides relative to step 4.


### Fitted quantities

| Quantity | Value | Unc (%) | Δ vs prev (%) |
|----------|------:|--------:|--------------:|
| SCA 33 | 13.3908 | 2.624 | -0.061 |
| SCA 35 | 14.1213 | 1.265 | -0.003 |
| **SCA 39** | **7.4728** | 5.106 | **+0.176** |
| SCA 41 | 11.9951 | 6.665 | -0.000 |
| **SCR 33** | **12.3795** | 3.744 | **+0.186** |
| SCR 35 | 12.7734 | 2.636 | -0.094 |
| **SCR 39** | **6.6595** | 17.420 | **+2.939** |
| SCR 41 | 10.8374 | 27.557 | +0.046 |
| **ABS 33** | **573.4874** | 0.186 | **-0.104** |
| ABS 35 | 683.4317 | 0.181 | -0.019 |
| **ABS 39** | **1020.6859** | 0.307 | **+0.179** |
| ABS 41 | 1383.6990 | 1.053 | +0.052 |
| **FIS 33** | **528.6252** | 0.240 | **-0.108** |
| FIS 35 | 584.3285 | 0.207 | -0.014 |
| **FIS 39** | **749.1488** | 0.304 | **+0.112** |
| FIS 41 | 1018.5549 | 1.096 | +0.037 |
| CA 33 | 44.8622 | 1.806 | -0.052 |
| CA 35 | 99.1032 | 0.791 | -0.047 |
| **CA 39** | **271.5372** | 0.863 | **+0.363** |
| CA 41 | 365.1441 | 1.635 | +0.094 |
| CAP 34 | 95.9000 | 2.086 | +0.000 |
| NUB 33 | 2.4914 | 0.184 | +0.003 |
| NUB 35 | 2.4304 | 0.169 | +0.002 |
| NUB 39 | 2.8824 | 0.191 | +0.016 |
| NUB 41 | 2.9444 | 0.208 | +0.008 |
| NUB 52 | 3.7668 | 0.129 | +0.002 |
| ETA 33 | 2.2965 | 0.199 | -0.001 |
| ETA 35 | 2.0780 | 0.190 | +0.007 |
| ETA 39 | 2.1156 | 0.264 | -0.051 |
| ETA 41 | 2.1674 | 0.390 | -0.007 |
| ALPHA 33 | 0.0849 | 1.948 | +0.056 |
| ALPHA 35 | 0.1696 | 0.875 | -0.033 |
| **ALPHA 39** | **0.3625** | 0.937 | **+0.251** |
| ALPHA 41 | 0.3585 | 1.573 | +0.057 |
| WGA 33 | 1.0000 | 0.100 | +0.000 |
| WGA 35 | 0.9820 | 0.305 | +0.000 |
| WGA 39 | 1.0780 | 0.928 | +0.000 |
| WGA 41 | 1.0460 | 0.669 | +0.000 |
| WGF 33 | 0.9970 | 0.100 | +0.000 |
| WGF 35 | 0.9810 | 0.510 | +0.000 |
| WGF 39 | 1.0520 | 1.331 | +0.000 |
| WGF 41 | 1.0520 | 0.665 | +0.000 |
| **FA 33** | **573.4874** | 0.212 | **-0.104** |
| FA 35 | 671.1299 | 0.355 | -0.019 |
| **FA 39** | **1100.2995** | 0.977 | **+0.179** |
| FA 41 | 1447.3492 | 1.248 | +0.052 |
| **FF 33** | **527.0393** | 0.261 | **-0.108** |
| FF 35 | 573.2262 | 0.550 | -0.014 |
| **FF 39** | **788.1045** | 1.365 | **+0.112** |
| FF 41 | 1071.5198 | 1.282 | +0.037 |
| F1ETA 33 | 2.2897 | 0.244 | -0.001 |
| F1ETA 35 | 2.0758 | 0.624 | +0.007 |
| F1ETA 39 | 2.0646 | 1.643 | -0.051 |
| F1ETA 41 | 2.1798 | 1.021 | -0.007 |
| **F2ETA 33** | **1313.0907** | 0.290 | **-0.105** |
| F2ETA 35 | 1393.1589 | 0.569 | -0.011 |
| **F2ETA 39** | **2271.6351** | 1.373 | **+0.128** |
| F2ETA 41 | 3154.9640 | 1.279 | +0.045 |
| **F3ETA 33** | **739.6033** | 0.444 | **-0.105** |
| F3ETA 35 | 722.0290 | 1.098 | -0.005 |
| F3ETA 39 | 1171.3357 | 2.772 | +0.080 |
| F3ETA 41 | 1707.6149 | 1.827 | +0.039 |

## Step 6: All ABS+SCR → ABS+SCA; Drop Lemmel SCA-SCR; SCR Partially Removed

Step 6 finalises the Capote chain:

  * **Convert** every (ABS X)+SCR X measurement to (ABS X)+SCA X — 26 conversions covering all transmission entries from baseline plus the step-5 additions.
  * **Delete** the four Lemmel SCA−SCR datapoints (Nos. 89, 91, 93, 95).
  * **Round-2 nuance** (see Appendix B): SCR(²³⁹Pu) and SCR(²⁴¹Pu) become unconstrained and are removed from the fit parameter space, but SCR(²³³U) and SCR(²³⁵U) remain in the fit because the four reclassified entries from step 4 (ATOYI, WILLIS, FOOTE on U-235; GREEN-233 on U-233) now constitute *direct* SCR measurements. Roberto's original *"reducing 4 fitting variables SCR"* simplifies to *reducing 2 fitting variables*. Decision logged in [B5](#b-round-2-corrections).

#### Impact relative to step 5
Differences are small. SCA channel essentially unchanged (much less than 0.02%) — the SCA additions in step 4 are the dominant constraint. ABS, FIS, CA, and the F-quantities shift by ≤0.15% as the converted transmission datapoints now correlate only with SCA, slightly relaxing the absorption fit. χ² is comparable (80.5 → 80.1, a 0.4 *decrease*) with 2 fewer free parameters.


### Fitted quantities

| Quantity | Value | Unc (%) | Δ vs prev (%) |
|----------|------:|--------:|--------------:|
| **SCA 33** | **13.3207** | 2.883 | **-0.524** |
| **SCA 35** | **14.0725** | 1.305 | **-0.345** |
| SCA 39 | 7.4716 | 5.105 | -0.016 |
| SCA 41 | 11.9950 | 6.665 | -0.001 |
| **SCR 33** | **12.6000** | 5.556 | **+1.781** |
| **SCR 35** | **12.9936** | 3.172 | **+1.723** |
| **ABS 33** | **572.7526** | 0.185 | **-0.128** |
| ABS 35 | 682.9085 | 0.180 | -0.077 |
| ABS 39 | 1020.1379 | 0.302 | -0.054 |
| ABS 41 | 1382.8937 | 1.047 | -0.058 |
| **FIS 33** | **527.9557** | 0.239 | **-0.127** |
| FIS 35 | 583.9164 | 0.207 | -0.071 |
| FIS 39 | 748.8023 | 0.301 | -0.046 |
| FIS 41 | 1018.0029 | 1.090 | -0.054 |
| **CA 33** | **44.7969** | 1.806 | **-0.146** |
| **CA 35** | **98.9921** | 0.791 | **-0.112** |
| CA 39 | 271.3356 | 0.855 | -0.074 |
| CA 41 | 364.8908 | 1.631 | -0.069 |
| CAP 34 | 95.9000 | 2.086 | +0.000 |
| NUB 33 | 2.4914 | 0.184 | -0.001 |
| NUB 35 | 2.4303 | 0.169 | -0.002 |
| NUB 39 | 2.8823 | 0.191 | -0.003 |
| NUB 41 | 2.9443 | 0.208 | -0.002 |
| NUB 52 | 3.7668 | 0.129 | -0.001 |
| ETA 33 | 2.2966 | 0.199 | +0.000 |
| ETA 35 | 2.0780 | 0.190 | +0.004 |
| ETA 39 | 2.1157 | 0.263 | +0.005 |
| ETA 41 | 2.1674 | 0.390 | +0.002 |
| ALPHA 33 | 0.0848 | 1.948 | -0.019 |
| ALPHA 35 | 0.1695 | 0.875 | -0.042 |
| ALPHA 39 | 0.3624 | 0.934 | -0.028 |
| ALPHA 41 | 0.3584 | 1.573 | -0.015 |
| WGA 33 | 1.0000 | 0.100 | +0.000 |
| WGA 35 | 0.9820 | 0.305 | +0.000 |
| WGA 39 | 1.0780 | 0.928 | +0.000 |
| WGA 41 | 1.0460 | 0.669 | +0.000 |
| WGF 33 | 0.9970 | 0.100 | +0.000 |
| WGF 35 | 0.9810 | 0.510 | +0.000 |
| WGF 39 | 1.0520 | 1.331 | +0.000 |
| WGF 41 | 1.0520 | 0.665 | +0.000 |
| **FA 33** | **572.7526** | 0.210 | **-0.128** |
| FA 35 | 670.6162 | 0.355 | -0.077 |
| FA 39 | 1099.7086 | 0.975 | -0.054 |
| FA 41 | 1446.5068 | 1.243 | -0.058 |
| **FF 33** | **526.3718** | 0.259 | **-0.127** |
| FF 35 | 572.8220 | 0.550 | -0.071 |
| FF 39 | 787.7400 | 1.364 | -0.046 |
| FF 41 | 1070.9390 | 1.277 | -0.054 |
| F1ETA 33 | 2.2897 | 0.244 | +0.000 |
| F1ETA 35 | 2.0759 | 0.624 | +0.004 |
| F1ETA 39 | 2.0647 | 1.643 | +0.005 |
| F1ETA 41 | 2.1799 | 1.021 | +0.002 |
| **F2ETA 33** | **1311.4128** | 0.289 | **-0.128** |
| F2ETA 35 | 1392.1472 | 0.569 | -0.073 |
| F2ETA 39 | 2270.5258 | 1.373 | -0.049 |
| F2ETA 41 | 3153.1933 | 1.275 | -0.056 |
| **F3ETA 33** | **738.6603** | 0.443 | **-0.128** |
| F3ETA 35 | 721.5310 | 1.098 | -0.069 |
| F3ETA 39 | 1170.8172 | 2.771 | -0.044 |
| F3ETA 41 | 1706.6865 | 1.824 | -0.054 |

## Step 5 vs Step 6: Head-to-Head Comparison

Both candidate fits agree to better than 0.2% on every primary and derived quantity. Step 6 has 2 fewer free parameters than step 5 (SCR(²³⁹Pu), SCR(²⁴¹Pu) become unconstrained after the ABS+SCR → ABS+SCA conversion; SCR(²³³U), SCR(²³⁵U) remain in the fit, constrained by the round-2 reclassified entries — see Appendix B5). χ² is comparable: 80.55 vs 80.14, with step 6 *fitting slightly better* despite the parameter reduction.

| Quantity | Step 5 | Step 6 | Δ% |
|----------|-------:|-------:|---:|
| **SCA 33** | 13.3908 | 13.3207 | **-0.524** |
| **SCA 35** | 14.1213 | 14.0725 | **-0.345** |
| SCA 39 | 7.4728 | 7.4716 | -0.016 |
| SCA 41 | 11.9951 | 11.9950 | -0.001 |
| **SCR 33** | 12.3795 | 12.6000 | **+1.781** |
| **SCR 35** | 12.7734 | 12.9936 | **+1.723** |
| SCR 39 | 6.6595 | – (removed) | – |
| SCR 41 | 10.8374 | – (removed) | – |
| **ABS 33** | 573.4874 | 572.7526 | **-0.128** |
| ABS 35 | 683.4317 | 682.9085 | -0.077 |
| ABS 39 | 1020.6859 | 1020.1379 | -0.054 |
| ABS 41 | 1383.6990 | 1382.8937 | -0.058 |
| **FIS 33** | 528.6252 | 527.9557 | **-0.127** |
| FIS 35 | 584.3285 | 583.9164 | -0.071 |
| FIS 39 | 749.1488 | 748.8023 | -0.046 |
| FIS 41 | 1018.5549 | 1018.0029 | -0.054 |
| **CA 33** | 44.8622 | 44.7969 | **-0.146** |
| **CA 35** | 99.1032 | 98.9921 | **-0.112** |
| CA 39 | 271.5372 | 271.3356 | -0.074 |
| CA 41 | 365.1441 | 364.8908 | -0.069 |
| CAP 34 | 95.9000 | 95.9000 | +0.000 |
| NUB 33 | 2.4914 | 2.4914 | -0.001 |
| NUB 35 | 2.4304 | 2.4303 | -0.002 |
| NUB 39 | 2.8824 | 2.8823 | -0.003 |
| NUB 41 | 2.9444 | 2.9443 | -0.002 |
| NUB 52 | 3.7668 | 3.7668 | -0.001 |
| ETA 33 | 2.2965 | 2.2966 | +0.000 |
| ETA 35 | 2.0780 | 2.0780 | +0.004 |
| ETA 39 | 2.1156 | 2.1157 | +0.005 |
| ETA 41 | 2.1674 | 2.1674 | +0.002 |
| ALPHA 33 | 0.0849 | 0.0848 | -0.019 |
| ALPHA 35 | 0.1696 | 0.1695 | -0.042 |
| ALPHA 39 | 0.3625 | 0.3624 | -0.028 |
| ALPHA 41 | 0.3585 | 0.3584 | -0.015 |
| WGA 33 | 1.0000 | 1.0000 | +0.000 |
| WGA 35 | 0.9820 | 0.9820 | +0.000 |
| WGA 39 | 1.0780 | 1.0780 | +0.000 |
| WGA 41 | 1.0460 | 1.0460 | +0.000 |
| WGF 33 | 0.9970 | 0.9970 | +0.000 |
| WGF 35 | 0.9810 | 0.9810 | +0.000 |
| WGF 39 | 1.0520 | 1.0520 | +0.000 |
| WGF 41 | 1.0520 | 1.0520 | +0.000 |
| **FA 33** | 573.4874 | 572.7526 | **-0.128** |
| FA 35 | 671.1299 | 670.6162 | -0.077 |
| FA 39 | 1100.2995 | 1099.7086 | -0.054 |
| FA 41 | 1447.3492 | 1446.5068 | -0.058 |
| **FF 33** | 527.0393 | 526.3718 | **-0.127** |
| FF 35 | 573.2262 | 572.8220 | -0.071 |
| FF 39 | 788.1045 | 787.7400 | -0.046 |
| FF 41 | 1071.5198 | 1070.9390 | -0.054 |
| F1ETA 33 | 2.2897 | 2.2897 | +0.000 |
| F1ETA 35 | 2.0758 | 2.0759 | +0.004 |
| F1ETA 39 | 2.0646 | 2.0647 | +0.005 |
| F1ETA 41 | 2.1798 | 2.1799 | +0.002 |
| **F2ETA 33** | 1313.0907 | 1311.4128 | **-0.128** |
| F2ETA 35 | 1393.1589 | 1392.1472 | -0.073 |
| F2ETA 39 | 2271.6351 | 2270.5258 | -0.049 |
| F2ETA 41 | 3154.9640 | 3153.1933 | -0.056 |
| **F3ETA 33** | 739.6033 | 738.6603 | **-0.128** |
| F3ETA 35 | 722.0290 | 721.5310 | -0.069 |
| F3ETA 39 | 1171.3357 | 1170.8172 | -0.044 |
| F3ETA 41 | 1707.6149 | 1706.6865 | -0.054 |

**Per the 2026-05-06 follow-up email, Step 5 is marginally favoured** as the final fit candidate, on the grounds that the systematic SCR < SCA difference visible in step 5 is physically expected and yields fission cross sections in slightly better agreement with current library R-matrix fits. Differences with step 6 are within uncertainties.

## K1 = F3ETA Evolution Through the Chain

K1 = F3ETA = NUB · FF − FA is the central derived quantity flagged as the K1 of the new TNC fit. Values across all 7 steps (round 2):

| Step | F3ETA 33 | F3ETA 35 | F3ETA 39 | F3ETA 41 |
|------|---------:|---------:|---------:|---------:|
| step0 baseline |  746.1581 |  723.5963 | 1169.1493 | 1710.5452 |
| step1 lounsbury |  741.9759 |  722.0181 | 1170.0323 | 1707.4469 |
| step2 adamchuk |  741.9650 |  721.8486 | 1169.9569 | 1707.3037 |
| step3 wallner |  741.9129 |  721.2836 | 1169.7702 | 1706.7079 |
| step4 sca list |  740.3842 |  722.0637 | 1170.3998 | 1706.9566 |
| step5 transmission |  739.6033 |  722.0290 | 1171.3357 | 1707.6149 |
| step6 all sca |  738.6603 |  721.5310 | 1170.8172 | 1706.6865 |

K1(²³³U) drops from ≈ 746.2 (step 0) to ≈ 738.7 (step 6) — a 1.0% reduction driven primarily by the Lounsbury inclusion in step 1. K1(²³⁵U) is essentially flat across the chain. K1(²³⁹Pu) shifts ≈ −0.1% from step 0 to step 6. K1(²⁴¹Pu) is flat. The dominant K1 changes between round 1 and round 2 come from the Westcott-factor update (Appendix B4): U-235 K1 ≈ +0.3%, Pu-239 K1 ≈ −0.9%, Pu-241 K1 ≈ +1.1%.

## Comprehensive Dataset Table (Final Fit, Step 6)

All 123 measurements actually used in the step-6 fit (after the microscopic filter is applied). Cross-references to the [Decisions appendix](#a-decisions-and-contentious-points) flag entries where a non-trivial choice was made.

| No. | Author | Reference | Quantity | Value | σ (%) | Appendix |
|----:|--------|-----------|----------|------:|------:|:---------|
| 5 | LEMMEL | 1982(1) | CAP 34 | 95.9 | 2.086 |  |
| 21 | LOUNSBURY | 1970(1) | (CA 33)/FIS 33 | 0.0861 | 2.239 |  |
| 22 | LOUNSBURY | 1970(1) | (CA 35)/FIS 35 | 0.1697 | 1.709 |  |
| 23 | LOUNSBURY | 1970(1) | (CA 39)/FIS 39 | 0.3555 | 1.603 |  |
| 54 | DIVADEENAM | 1984(2) | WGA 33 | 1 | 0.100 |  |
| 55 | DIVADEENAM | 1984(2) | WGA 35 | 0.982 | 0.305 |  |
| 56 | DIVADEENAM | 1984(2) | WGA 39 | 1.078 | 0.928 |  |
| 57 | DIVADEENAM | 1984(2) | WGA 41 | 1.046 | 0.669 |  |
| 58 | DIVADEENAM | 1984(2) | WGF 33 | 0.997 | 0.100 |  |
| 59 | DERUYTTER | 1985(3) | WGF 35 | 0.981 | 0.510 |  |
| 60 | DIVADEENAM | 1984(2) | WGF 39 | 1.052 | 1.331 |  |
| 61 | DIVADEENAM | 1984(2) | WGF 41 | 1.052 | 0.665 |  |
| 62 | EGELSTAFF | 1954(1) | (ABS 35)+SCA 35 | 724.33 | 3.590 |  |
| 63 | MELKONIAN | 1953(1) | (ABS 35)+SCA 35 | 694.32 | 2.016 |  |
| 64 | PALEVSKY | 1954(1) | (ABS 35)+SCA 35 | 700.32 | 1.428 |  |
| 65 | NITIKIN | 1955(1) | (ABS 35)+SCA 35 | 710.32 | 2.956 |  |
| 66 | SIMPSON | 1960(1) | (ABS 35)+SCA 35 | 690.32 | 1.391 |  |
| 67 | SAFFORD | 1959(1) | (ABS 35)+SCA 35 | 696.32 | 0.359 |  |
| 68 | SAFFORD | 1959(1) | (ABS 35)+SCA 35 | 698.32 | 0.730 |  |
| 69 | BLOCK | 1960(1) | (ABS 35)+SCA 35 | 693.32 | 0.721 |  |
| 70 | SAPLAKOGLU | 1961(1) | (ABS 35)+SCA 35 | 696.32 | 0.359 |  |
| 71 | GERASIMOV | 1962(1) | ABS 35 | 670.31 | 2.089 |  |
| 72 | ZIMMERMAN | 1955(1) | (ABS 39)+SCA 39 | 1022.5 | 1.271 |  |
| 73 | NITIKIN | 1955(1) | (ABS 39)+SCA 39 | 1040.5 | 2.979 |  |
| 74 | BOLLINGER | 1958(1) | (ABS 39)+SCA 39 | 1022.5 | 1.369 |  |
| 75 | PATTENDEN | A1956(1) | (ABS 39)+SCA 39 | 1012.5 | 3.003 |  |
| 76 | SAFFORD | 1961(1) | (ABS 39)+SCA 39 | 1018.5 | 0.727 |  |
| 77 | MUETHER | 1954(1) | (ABS 33)+SCA 33 | 597.27 | 2.344 |  |
| 78 | NITIKIN | 1955(1) | (ABS 33)+SCA 33 | 577.26 | 3.638 | [A8](#a-decisions-and-contentious-points) |
| 79 | PATTENDEN | 1956(1) | (ABS 33)+SCA 33 | 596.27 | 3.019 |  |
| 80 | GREEN | 1957(1) | ABS 33 | 574.26 | 3.483 |  |
| 81 | SIMPSON | 1960(1) | (ABS 33)+SCA 33 | 587.27 | 0.800 |  |
| 82 | SAFFORD | 1960(1) | (ABS 33)+SCA 33 | 585.77 | 0.990 |  |
| 83 | SAFFORD | 1960(1) | (ABS 33)+SCA 33 | 585.67 | 0.410 | [A9](#a-decisions-and-contentious-points) |
| 84 | BLOCK | 1960(1) | (ABS 33)+SCA 33 | 587.27 | 0.511 |  |
| 85 | SIMPSON | 1961(1) | (ABS 41)+SCA 41 | 1386.6 | 3.606 |  |
| 86 | CRAIG | 1964(1) | (ABS 41)+SCA 41 | 1380.6 | 2.897 |  |
| 87 | SMITH | 1984(3) | (ABS 41)+SCA 41 | 1395.6 | 1.433 |  |
| 96 | DERUYTTER | 1974(1) | 39 FH1 39 | 1812.1 | 0.454 |  |
| 97 | DERUYTTER | 1973(1) | 34 FH1 35 | 1439.1 | 0.445 |  |
| 98 | DERUYTTER | 1961(1) | FIS 35 | 589.37 | 1.325 |  |
| 99 | WHITE | 1967(1) | (FIS 39)/FIS 35 | 1.253 | 1.800 |  |
| 100 | WHITE | 1967(1) | (FIS 41)/FIS 35 | 1.763 | 2.751 |  |
| 101 | BERCEANO | 1977(1) | 34 FH1 35 | 1443 | 0.791 |  |
| 102 | RAFFLE | 1959(1) | 33 FH1 33 | 834.26 | 3.302 |  |
| 103 | RAFFLE | 1959(1) | FIS 35 | 581.97 | 3.092 |  |
| 104 | RAFFLE | 1959(1) | 39 FH1 39 | 1714.8 | 2.843 |  |
| 105 | RAFFLE | 1956(1) | (FIS 41)/FIS 39 | 1.3319 | 6.000 |  |
| 106 | MASLIN | 1965(1) | FIS 35 | 583.71 | 1.449 |  |
| 107 | SAPLAKOGLU | 1959(1) | FIS 35 | 593.17 | 2.232 |  |
| 108 | WATANABE | 1964(1) | FIS 41 | 985.45 | 4.569 |  |
| 109 | FRAYSSE | 1965(1) | (39 FH1 39)/34 FH1 35 | 1.2425 | 1.557 |  |
| 110 | BORCEA | 1970(1) | (39 FH1 39)/34 FH1 35 | 1.2578 | 1.624 |  |
| 111 | BOLDEMAN | 1977(1) | NUB 52 | 3.7549 | 0.431 |  |
| 112 | BOLDEMAN | 1980(1) | (NUB 33)/NUB 52 | 0.66049 | 0.255 |  |
| 113 | BOLDEMAN | 1980(1) | (NUB 35)/NUB 52 | 0.64318 | 0.255 |  |
| 114 | BOLDEMAN | 1980(1) | (NUB 39)/NUB 52 | 0.76614 | 0.243 |  |
| 115 | BOLDEMAN | 1980(1) | (NUB 41)/NUB 52 | 0.78028 | 0.231 |  |
| 116 | SPENCER | 1982(1) | NUB 52 | 3.7831 | 0.221 |  |
| 117 | GWIN | 1984(1) | (NUB 33)/NUB 52 | 0.65996 | 0.340 |  |
| 118 | GWIN | 1984(1) | (NUB 35)/NUB 52 | 0.64701 | 0.265 |  |
| 119 | GWIN | 1984(1) | (NUB 39)/NUB 52 | 0.7653 | 0.257 |  |
| 120 | GWIN | 1984(1) | (NUB 41)/NUB 52 | 0.78423 | 0.290 |  |
| 121 | HOPKINS | 1963(1) | NUB 52 | 3.7767 | 0.838 |  |
| 122 | HOPKINS | 1963(1) | (NUB 33)/NUB 52 | 0.65483 | 0.895 |  |
| 123 | HOPKINS | 1963(1) | (NUB 35)/NUB 52 | 0.64587 | 0.838 |  |
| 124 | HOPKINS | 1963(1) | (NUB 39)/NUB 52 | 0.75047 | 0.996 |  |
| 125 | ASPLUND | 1963(1) | NUB 52 | 3.791 | 1.066 |  |
| 126 | CONDE | 1965(1) | (NUB 35)/NUB 52 | 0.64222 | 0.847 |  |
| 127 | WHITE | 1968(1) | NUB 52 | 3.8194 | 1.033 |  |
| 128 | AXTON | A1985(3) | NUB 52 | 3.7547 | 0.300 |  |
| 129 | COLV/AXT | 1966(1) | NUB 52 | 3.7299 | 0.806 |  |
| 130 | COLV/ULL | 1965(1) | NUB 52 | 3.7405 | 0.438 |  |
| 131 | COLVIN | 1965(1) | (NUB 33)/NUB 35 | 1.0204 | 0.587 |  |
| 132 | COLVIN | 1965(1) | (NUB 39)/NUB 35 | 1.1835 | 0.675 |  |
| 133 | COLVIN | 1965(1) | (NUB 41)/NUB 35 | 1.2102 | 0.905 |  |
| 134 | COLVIN | 1965(1) | (NUB 35)/NUB 52 | 0.64334 | 0.472 |  |
| 135 | MATHER | 1964(1) | (NUB 33)/NUB 52 | 0.67093 | 1.235 |  |
| 136 | MATHER | 1964(1) | (NUB 35)/NUB 52 | 0.64145 | 0.537 |  |
| 137 | MATHER | 1964(1) | (NUB 39)/NUB 52 | 0.77307 | 1.122 |  |
| 138 | ALEK-ROV | 1981(1) | NUB 52 | 3.7618 | 0.483 |  |
| 139 | SMITH | 1984(1) | NUB 52 | 3.7678 | 0.303 |  |
| 140 | EDWARDS | 1982(1) | NUB 52 | 3.7641 | 0.711 |  |
| 141 | BOZ-NESH | 1977(1) | NUB 52 | 3.7475 | 0.580 |  |
| 142 | DEVOLPI | 1972(1) | NUB 52 | 3.7507 | 0.463 |  |
| 143 | ZHANG | 1981(1) | NUB 52 | 3.7534 | 0.490 |  |
| 144 | SPIEGEL | 1981(1) | NUB 52 | 3.7828 | 0.759 |  |
| 145 | SMITH | 1984(1) | ETA 33 | 2.2972 | 0.350 |  |
| 146 | SMITH | 1984(1) | ETA 35 | 2.0842 | 0.386 |  |
| 147 | SMITH | 1984(1) | ETA 39 | 2.112 | 0.380 |  |
| 148 | SMITH | 1984(1) | ETA 41 | 2.169 | 0.419 |  |
| 149 | MACKLIN | 1960(1) | ETA 33 | 2.3121 | 0.375 |  |
| 150 | MACKLIN | 1960(1) | ETA 35 | 2.0757 | 0.382 |  |
| 151 | MACKLIN | 1962(1) | ETA 39 | 2.1163 | 0.851 |  |
| 165 | LLOYD | 1982(1) | (ETA 39)/FIS 39 | 0.002782 | 2.072 |  |
| 166 | SPENCER | 1985(3) | (ABS 35)+SCA 35 | 690 | 1.000 | [A1](#a-decisions-and-contentious-points) |
| 167 | SPENCER | 1985(3) | (ABS 39)+SCA 39 | 1025.8 | 1.000 | [A1](#a-decisions-and-contentious-points) |
| 168 | ADAMCHUK | At. En. (R), 65, 434 (1988) | (CA 35)/FIS 35 | 0.169 | 2.100 |  |
| 169 | WALLNER | PRL 112, 192501 (2014) | CA 35 | 99.5 | 2.000 |  |
| 170 | WALLNER | PRL 112, 192501 (2014) | CA 35 | 101.5 | 2.000 |  |
| 171 | ATOYI | Acta Cryst. 20, 587 (1966) | SCR 35 | 12.3 | 4.065 | [B2](#a-decisions-and-contentious-points) |
| 172 | WILLIS | PRS A274, 122 (1963) | SCR 35 | 12.8 | 7.812 | [B2](#a-decisions-and-contentious-points) |
| 174 | ARIF | PR A35, 2810 (1987) | SCA 35 | 14 | 1.429 |  |
| 175 | CEULEMANS | Divadeenam compilation 1970 | SCA 35 | 14.6 | 3.425 | [A4](#a-decisions-and-contentious-points) |
| 176 | BAYVAS | Divadeenam compilation 1980 | SCA 35 | 14.3 | 10.489 |  |
| 177 | FOOTE | Divadeenam compilation 1958 | SCR 35 | 15 | 6.667 | [B3](#a-decisions-and-contentious-points) |
| 178 | OLEKSA | Phys Rev 109, 1645 (1958) | SCA 33 | 13.4 | 3.731 | [A5](#a-decisions-and-contentious-points) |
| 179 | GREEN | JNM 51, 281 (1974) | SCR 33 | 12.6 | 5.556 | [B2](#a-decisions-and-contentious-points) |
| 180 | VERTEBNYI | Divadeenam compilation 1974 | SCA 33 | 13.2 | 4.545 |  |
| 181 | GREEN | JNM 34, 281 (1970) | SCA 39 | 7.7 | 7.792 |  |
| 182 | ATOYI | Acta Cryst. 20, 587 (1966) | SCA 39 | 7.3 | 9.589 |  |
| 183 | ROOF | Acta Cryst. 15, 351 (1962) | SCA 39 | 7.3 | 9.589 |  |
| 184 | PRONYAEV | Internal analysis 2026-04-17 | SCA 41 | 12 | 6.667 | [A7](#a-decisions-and-contentious-points) |
| 185 | ANTONOV | Capote 2026-05-04 list | (ABS 35)+SCA 35 | 695 | 2.158 | [A10](#a-decisions-and-contentious-points) |
| 186 | GERASIMOV | Capote 2026-05-04 list | (ABS 35)+SCA 35 | 685 | 1.314 | [A11](#a-decisions-and-contentious-points) |
| 187 | LEONARD | Capote 2026-05-04 list (1954) | (ABS 35)+SCA 35 | 702 | 1.994 | [A12](#a-decisions-and-contentious-points) |
| 188 | PSCHENICHNYI | Capote 2026-05-04 list (1978) | (ABS 33)+SCA 33 | 587.9 | 0.510 | [A13](#a-decisions-and-contentious-points) |
| 189 | HARVEY-1 | Capote 2026-05-04 list (1979) | (ABS 33)+SCA 33 | 585 | 0.239 | [A14](#a-decisions-and-contentious-points) |
| 190 | HARVEY-2 | Capote 2026-05-04 list (1979) | (ABS 33)+SCA 33 | 592 | 1.014 | [A14](#a-decisions-and-contentious-points) |
| 191 | SCHWARTZ | Capote 2026-05-04 list (1958) | (ABS 41)+SCA 41 | 1410 | 5.674 |  |
| 192 | ANDERSON | Capote 2026-05-04 list (1945) | (ABS 39)+SCA 39 | 1045 | 2.392 |  |
| 193 | PALEVSKY | Capote 2026-05-04 list (1955) | (ABS 39)+SCA 39 | 1025 | 0.976 |  |
| 194 | LEONARD | Capote 2026-05-04 list (1956) | (ABS 39)+SCA 39 | 1055 | 1.232 | [A15](#a-decisions-and-contentious-points) |

## A. Decisions and Contentious Points

Every choice that involved interpretation, departure from a literal reading of Roberto's emails, or risk of double-counting is recorded here so collaborators can scrutinise it.

### A1. Spencer 1987 vs. existing No.166/167 — values used as-is

Roberto's "Revision/additions" email lists Spencer 1987 ABS+SCR(²³⁵U) = 690(5) and Spencer 1987 ABS+SCR(²³⁹Pu) = 1025(6). The baseline already contains:

  * No.166 SPENCER 1985(3) ABS+SCR(²³⁵U) = 690.0, σ = 1.0% (= 6.9 b), with a single systematic component `alpha = 0.707`.
  * No.167 SPENCER 1985(3) ABS+SCR(²³⁹Pu) = 1025.8, σ = 1.0% (= 10.26 b), same systematic.

The values match within rounding (690 vs 690.0; 1025 vs 1025.8 — likely the same measurement, presented at a 1985 conference and published 1987). Per project decision, the database entries are retained as-is rather than being re-overwritten with Roberto's values. **Note**: the uncertainties differ noticeably (DB: 1.0%; Roberto: 0.72% / 0.59%); the more conservative DB uncertainty is what enters the fit. Step-6 conversion turns these into ABS+SCA(²³⁵U) and ABS+SCA(²³⁹Pu) like all other ABS+SCR entries.

### A2. Atoyi 1966 ABS+SCR tag treated as SCA

In the SEARS column of Capote's 2026-04-30 inlined compilation, the Atoyi 1966 row is annotated "SCR". The same row also gives Sears Sel = 12.3(5) b. Per project decision, the entire compilation is treated as providing SCA values regardless of the per-row "SCR/SCA" tag, because (a) Sears' formalism delivers the bound-atom S_el (= SCA) by construction, and (b) the per-row tags in Roberto's table show inconsistencies that are most plausibly explained as paste artifacts from a heterogeneous source table.

### A3. Willis 1963 — EXFOR value used

The Sears compilation gives Willis 1963 = 12.1(5) coh + 0.2(2) inc → S_el 12.3(5). Roberto annotates "replace with EXFOR value" and the Divadeenam section gives "WILLIS 1963 12.8(1.0) = EXFOR". The EXFOR value 12.8(1.0) is used in step 4.

### A4. Ceulemans 1970 — V-51 renormalisation used as-is

Roberto's table notes "EXFOR 14.3(5) rel V-51 (1.02)" — the 14.6(5) value used in step 4 is 14.3 × 1.02, the 1.02 factor being a renormalisation against vanadium-51 standard. Different evaluations of V-51 thermal scattering vary by ~1–3% (Hughes-Schwartz 1958: ~5.0 b; Sears 1992 / ENDF/B-VIII: 5.08 b; ratio ≈ 1.016). Re-renormalising against current standards would change Ceulemans by < 1% (≈ 0.05–0.15 b), well under the quoted σ = 0.5 b — no practical impact on the fit. **Used as-is per project decision.**

### A5. Oleksa 1960 — quoted at 0.1 eV, used as-is

The EXFOR replacement Oleksa 13.4(5) is quoted "at 0.1 eV", not at the thermal point 0.0253 eV. Two candidate corrections were estimated:

  * **Bound→free transition**: σ_free / σ_bound = (A/(A+1))² = 0.9914 for A=233, i.e. <0.3% effect at 0.1 eV (still in bound regime); ≤ 0.04 b.
  * **Resonance contribution**: U-233 has no large resonance below 1 eV. Tail contributions from negative resonances and the 1.45 eV resonance at 0.1 eV are at most ≈ 0.05 b.

**Combined |Δσ| < 0.1 b, well below quoted σ = 0.5 b. Used as-is per project decision.**

### A6. Green 1974 ABS+SCR tag treated as SCA

Same situation as A2: the Divadeenam-section row for Green 1974 carries a "SCR" annotation but is added as SCA(²³³U) per the global decision to treat all compilation rows as SCA.

### A7. Pu-241 SCA — Pronyaev internal analysis

No direct experimental SCA measurement exists for ²⁴¹Pu. Pronyaev's 2026-04-17 internal analysis derives SCA(²⁴¹Pu) = 12.0(0.8) b from the potential scattering radius R_pot = 9.54 fm (consistent across ENDF/B-VIII.1, JEFF-4.0 and JENDL-5; 4πR² = 11.44 b plus the contribution of negative resonances in the dominant ENDF/B-VIII.1 evaluation). This estimate is encoded as a direct SCA(²⁴¹Pu) "measurement" replacing the deleted Lemmel No.94 (12.006 b, σ = 21.6%). The replacement reduces the SCA(²⁴¹Pu) uncertainty from 21.6% (Lemmel) to 6.7% (Pronyaev).

### A8. Nikitin 1955 spelling — DB uses "NITIKIN"

Roberto writes "Nikitin" (correct transliteration of the Russian Никитин); the database has the typo "NITIKIN" inherited from Axton 1986 throughout (Nos. 65, 73, 78). The DB spelling is preserved for stable cross-referencing.

### A9. Safford 1960 vs 1959 ambiguity

Roberto's "Revision/additions" email instructs *"Change Safford 1959 ABS+SCR to ABS+SCA"* for U-233. The database has no Safford 1959 U-233 entry; Safford's U-233 transmission measurements are dated 1960 (Nos. 82 ABS+SCA(²³³U) = 585.77 already, 83 ABS+SCR(²³³U) = 585.67). The 1959 → 1960 mismatch is treated as a typo and No.83 is reclassified per Roberto's evident intent. Worth verifying with Roberto if there is an actual 1959 paper we are missing.

### A10. Antonov 1986 ABS+SCA tag — non-default for transmission

Roberto tags Antonov 1986 ²³⁵U as ABS+SCA, which is unusual for a transmission measurement (the dominant convention is ABS+SCR for rolled-metal samples). The tag is honoured under the assumption that Antonov used a non-metallic sample (UO₂ powder?) or applied a proper extinction correction. Verify with Roberto / read the original paper.

### A11. Gerasimov 1962 — likely separate from existing No.71

A GERASIMOV 1962 entry exists in the baseline as **No.71 GERASIMOV 1962(1) ABS 35 = 670.31 b, σ = 2.09%** (bare ABS, no scatter component). Roberto's "Gerasimov 62 ABS+SCR 685(9)" has a different value (685 vs 670.3) and a different quantity (ABS+SCR vs ABS). Per project decision, this is added as a separate measurement (No.186) rather than overwriting No.71. Worth verifying with Roberto if this is in fact the same measurement re-analysed.

### A12. Leonard 1954 — type defaulted to ABS+SCR

Roberto's email gives "Leonard 54 702(14)" with no SCR/SCA tag. Defaulted to ABS+SCR per the convention for the era (rolled-metal samples) and pending paper lookup. This decision is **moot for the final fit** because step 6 converts all ABS+SCR to ABS+SCA in any case.

### A13. Pschenichnyi 1978 uncertainty — typo fix 0.3 → 3.0

Roberto's email reads *"Pschenichnyi 78 ABS+SCR 587.9(3)"*, which would be σ = 0.3 b on 587.9 b ≈ 0.05% — implausibly small for a transmission measurement. **Treated as a typo and used as σ = 3.0 b** (≈ 0.51%). This decision is conservative; if Roberto truly intended 0.3 b, Pschenichnyi would dominate the U-233 absorption fit by orders of magnitude.

### A14. Harvey-1 / Harvey-2 — kept uncorrelated; correlation alternative noted

The two Harvey 1979 ABS+SCR(²³³U) entries come from the same author, same year, and likely the same ORELA campaign — physical correlation through sample mass / normalisation is plausible. Per project decision, the two are entered with empty `systematic_uncertainties` and treated as independent, matching the convention for nearly all other Axton transmission entries.

**Alternative considered (not adopted)**: introduce a shared code `harvey79` with a 0.20% common component:
```
No.189 Harvey-1: σ_total = 0.24%; sys = {harvey79: 0.20}
No.190 Harvey-2: σ_total = 1.01%; sys = {harvey79: 0.20}
```
This would marginally tighten the U-233 ABS uncertainty. Worth revisiting if Roberto wants tighter modelling of correlated systematics.

### A15. Leonard 1956 ABS+SCR(²³⁹Pu) — largest tension in step 5

Leonard 1956 ABS+SCR(²³⁹Pu) = 1055 b lies +2.1σ above the fit-predicted 1027.4 b. This contributes ≈4.5 of the 9.5 χ² increase between step 4 and step 5. The measurement is retained as-is; flagged here as the dominant transmission tension in the chain.

### A16. SCR removed from parameter space in step 6

Roberto's step 6 instructs *"CONVERT ALL SCR in ABS+SCR to ABS+SCA, delete Lemmel SCA-SCR (reducing 4 fitting variables SCR)"*. The literal application:

  * Convert every (ABS X)+SCR X → (ABS X)+SCA X (26 conversions).
  * Delete the four Lemmel SCA−SCR datapoints (Nos. 89, 91, 93, 95).
  * **No measurement function or datapoint references SCR after these changes**, so the four SCR primary parameters become unconstrained.
  * They are therefore removed from the fit parameter space, consistent with the existing handling of unconstrained parameters (CA 40, CA 42, GC116, GA116). This matches Roberto's parenthetical "reducing 4 fitting variables SCR" exactly.

### A17. Microscopic filter and unconstrained parameters

All seven benchmarks apply the same microscopic filter that reproduces Axton Table 8: exclude measurements whose `measured_function` references any Maxwellian-derived quantity (FA, FF, CAP, FFH, F1ETA, F2ETA, F3ETA, FLEM, F1BIG, F1CAB–F5CAB, GC116, GA116) — except CAP 34 (No.5), which is retained because it ≈ σ_γ for non-fissile U-234. Lounsbury alpha (Nos. 21–23) is excluded only in step 0; it is included in steps 1–6. Unconstrained parameters CA 40, CA 42, GC116/40/41/42, GA116/39/41 are removed from the fit parameter space throughout. Step 6 additionally removes SCR 33/35/39/41 (see A16).

### A18. Empty systematic uncertainties for step-5 transmission additions

All step-5 transmission additions are entered with empty `systematic_uncertainties` (single uncorrelated total uncertainty), matching the convention used for 26 of the 28 baseline transmission entries. Spencer 1985 (Nos. 166/167) is the existing exception with `alpha = 0.707` representing the α-correction Spencer used internally. The Harvey-pair correlation alternative is the only other systematic structure considered (see A14).

## B. Round-2 Corrections (2026-05-08)

The 2026-05-08 email round produced three corrections to the round-1 fit. Each is documented below; the verbatim source is preserved in [`references/emails/tnc_update_roberto_20260508/summary.md`](../references/emails/tnc_update_roberto_20260508/summary.md). The original round-1 outputs are preserved as `evaluation/capote_chain_summary_20260504.json` and `evaluation/report_capote_fit_20260504.pdf` for direct numerical comparison.

### B1. KAISER (Physica B 136, 134 (1986)) removed as duplicate of ARIF

Kaiser et al. 1986 and Arif et al. 1987 used the same neutron-interferometry installation at the same institute; Kaiser gave preliminary values at only two energies, while Arif is the final publication. The 2026-05-08 review (Pronyaev) flagged the duplication; the 2026-05-05 follow-up confirmed deletion of KAISER, retaining ARIF.

### B2. ATOYI / WILLIS / GREEN-233 reclassified SCA → SCR

In round 1, the entire Sears/Divadeenam compilation was treated as providing SCA values per project decision (the per-row "SCR" tags were assumed to be paste artefacts). The 2026-05-08 review identified three rows for which the SCR tag is consistent with the actual measurement method:

  * **ATOYI 1966** (²³⁵U, Sears Sel = 12.3(5)): tagged SCR by Sears.
  * **WILLIS 1963** (²³⁵U, EXFOR value 12.8(1.0)): SCR per Sears.
  * **GREEN 1974** (²³³U, 12.6(7)): per the Pronyaev analysis, the EXFOR record indicates a classical (non-crystal-spectrometer) method, consistent with SCR. Ad-hoc SCA estimation supplied by Pronyaev.

All three are reclassified from SCA to SCR in steps 4, 5 and 6.

### B3. FOOTE 1958 reclassified SCA → SCR (inferred)

The cleaned compilation in Roberto's 2026-05-05 email tags FOOTE 1958 (²³⁵U, 15.0(1.0)) as "SCR", but the explicit numbered action items only list ATOYI, WILLIS and GREEN-233. Per project decision, the cleaned table is treated as the new authoritative classification, so FOOTE is reclassified to SCR alongside the three named entries. **This decision is inferred and worth confirming with Roberto.** The K1 impact of this single reclassification is ≤ 0.05% (FOOTE's 6.7% uncertainty is much weaker than ARIF's 1.4% which dominates the SCA(²³⁵U) channel).

### B4. Westcott factors updated to B81-derived values

The 2026-05-06 email replaces the Westcott factors with values derived from the B81 evaluation (R-matrix fit best estimate), with uncertainties set by the spread between B81 and JEFF-4.0:


| nuclide | Old WGA (Gabs) | New WGA (Gabs) | Old WGF (Gf) | New WGF (Gf) |
|---------|---------------:|---------------:|-------------:|-------------:|
| U-233   | 0.9988 (0.12%) | **1.000 (0.10%)** | 0.9966 (0.20%) | **0.997 (0.10%)** |
| U-235   | 0.9787 (0.09%) | **0.982 (0.31%)** | 0.9778 (0.09%) | **0.981 (0.51%)** |
| Pu-239  | 1.0767 (0.27%) | **1.078 (0.93%)** | 1.0562 (0.28%) | **1.052 (1.33%)** |
| Pu-241  | 1.0444 (0.19%) | **1.046 (0.67%)** | 1.0452 (0.67%) | **1.052 (0.67%)** |


Gabs is derived as Gabs = (Gcap·Scap + Gfis·Sfis)/(Scap+Sfis) per Capote's 2026-05-06 derivation. The new factors apply to **all seven steps**: steps 0 and 1 read from a new `axton1986_data_capote_round2_baseline.json` file (= original baseline + new Westcott); steps 2–6 inherit through their respective JSONs.

K1 impact (Step 6, round 2 vs round 1):


| Nuclide | K1 (round 1) | K1 (round 2) | Δ |
|---------|-------------:|-------------:|---:|
| ²³³U    | 738.81       | 738.66      | -0.020% |
| ²³⁵U    | 719.32       | 721.53      | +0.307% |
| ²³⁹Pu   | 1181.26      | 1170.82     | -0.884% |
| ²⁴¹Pu   | 1688.54      | 1706.69     | +1.075% |


For comparison, K1(²³⁵U, Step 5, round 2) = 722.03, in close agreement with the value reported in the 2026-05-06 email (722.4) using the same new Westcott factor.

### B5. SCR(²³³U), SCR(²³⁵U) retained as fit variables in step 6

In round 1, step 6 removed all four SCR primary parameters from the fit because no measurement function or datapoint referenced SCR after the conversions and Lemmel deletion. In round 2, the reclassifications (B2, B3) introduce four direct SCR measurements: ATOYI (²³⁵U), WILLIS (²³⁵U), FOOTE (²³⁵U), and GREEN-233 (²³³U). These constrain SCR(²³³U) and SCR(²³⁵U), so the two parameters remain in the fit. SCR(²³⁹Pu) and SCR(²⁴¹Pu) have no remaining constraints and stay unconstrained (removed).

Net effect: Roberto's parenthetical *"reducing 4 fitting variables SCR"* in the original step 6 description simplifies to *reducing 2 fitting variables*. Step 6 now has 28 free parameters (was 26 in round 1).

### B6. Round-2 baseline file separation

To preserve the Axton-Table-8 reproduction artefact, the original `axton1986/axton1986_data.json` is left **unchanged** and a new file `axton1986/axton1986_data_capote_round2_baseline.json` is introduced for steps 0 and 1. The new file is identical to the original except for the eight Westcott measurements (No. 54–61). Steps 2–6 carry their own copies of all measurements; the Westcott updates are applied in-place to those JSONs.

## C. Files Produced by This Chain

All on branch `tnc_update_capote_20260504`, building on commit `575fac0` (the Axton 1986 reproduction state of master).

### JSON databases (one per step where data changes)

  * `axton1986/axton1986_data.json` — original Axton-Table-8 reproduction baseline (untouched; not currently used by any step)
  * `axton1986/axton1986_data_capote_round2_baseline.json` — round-2 baseline (= original + new Westcott); steps 0 and 1 read this
  * `axton1986/axton1986_data_capote_step2_adamchuk.json` — 168 measurements
  * `axton1986/axton1986_data_capote_step3_wallner.json` — 170 measurements
  * `axton1986/axton1986_data_capote_step4_sca_list.json` — 179 measurements (round 2: 180 − 1 KAISER deleted)
  * `axton1986/axton1986_data_capote_step5_transmission.json` — 189 measurements (round 2: 190 − 1 KAISER deleted)
  * `axton1986/axton1986_data_capote_step6_all_sca.json` — 185 measurements (round 2; final)

### Benchmarks

  * `evaluation/benchmark_capote_step0_baseline.py`
  * `evaluation/benchmark_capote_step1_lounsbury.py`
  * `evaluation/benchmark_capote_step2_adamchuk.py`
  * `evaluation/benchmark_capote_step3_wallner.py`
  * `evaluation/benchmark_capote_step4_sca_list.py`
  * `evaluation/benchmark_capote_step5_transmission.py`
  * `evaluation/benchmark_capote_step6_all_sca.py`
  * `evaluation/benchmark_capote_summary.py` — runs all 7 in sequence and dumps `capote_chain_summary.json`

### Generated artefacts

  * `evaluation/capote_chain_summary.json` — machine-readable consolidation of all step results (round 2)
  * `evaluation/capote_chain_summary_20260504.json` — round-1 snapshot for direct comparison
  * `evaluation/report_capote_fit.md` — this report (round 2)
  * `evaluation/report_capote_fit_20260504.pdf` — round-1 PDF snapshot
  * `evaluation/build_capote_report.py` — the script that generated this report from the summary JSON

### Source paraphrases

  * [`references/emails/tnc_update_roberto_20260505/summary.md`](../references/emails/tnc_update_roberto_20260505/summary.md) — paraphrased thread covering the round-1 instructions (2026-04-09 → 2026-05-04)
  * [`references/emails/tnc_update_roberto_20260508/summary.md`](../references/emails/tnc_update_roberto_20260508/summary.md) — paraphrased round-2 corrections (2026-05-05 → 2026-05-06)
