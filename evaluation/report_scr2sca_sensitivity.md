# Sensitivity Check: SCR → SCA Reclassification of the Two High Pu-239 Total-XS Points

*Round-3 TNC fit (2026-06-23 database). Sensitivity test prompted by G. Noguère's query on the scattering classification of the two highest Pu-239 total-cross-section measurements.*

## Question

Two Pu-239 total-cross-section (transmission) measurements sit clearly above the rest of the Pu-239 (n,tot) set (which clusters at ~1022–1030 b):

| No. | Author | Year | Quantity (round 3) | Value | σ |
|----:|--------|------|--------------------|------:|---:|
| 75  | Pattenden | 1956 | (ABS 39)+SCR 39 | 1039 b | ±25 b (2.41%) |
| 192 | Anderson  | 1945 | (ABS 39)+SCR 39 | 1045 b | ±25 b (2.39%) |

Both currently carry their scattering component as **SCR** (the "rolled metallic sample" / reduced-coherent convention). The question raised is whether that assignment should instead be **SCA** (true elastic). This note refits the round-3 database with both entries changed to `(ABS 39)+SCA 39` and compares against the round-3 baseline. Nothing else is altered.

## Result

| Quantity | Round 3 (SCR) | Variant (SCA) | Δ | Δ% |
|----------|--------------:|--------------:|---:|---:|
| χ² | 73.42 | 73.36 | -0.07 | — |
| **SCA 39** | 7.4805 | 7.4805 | -0.0001 | -0.0008% |
| **SCR 39** | 6.7373 | 6.6958 | -0.0415 | -0.6165% |
| **ABS 39** | 1022.0765 | 1022.0701 | -0.0063 | -0.0006% |
| **FIS 39** | 750.0765 | 750.0742 | -0.0022 | -0.0003% |
| **CA 39** | 272.0000 | 271.9959 | -0.0041 | -0.0015% |
| **ALPHA 39** | 0.3626 | 0.3626 | -0.0000 | -0.0012% |
| **σ_tot 39 = ABS+SCA** | 1029.5570 | 1029.5506 | -0.0064 | -0.0006% |
| **K1 = F3ETA 39** | 1172.5370 | 1172.5353 | -0.0017 | -0.0001% |

Other nuclides (unaffected — shown for completeness):

| Quantity | Round 3 | Variant | Δ% |
|----------|--------:|--------:|---:|
| σ_tot 33 | 588.3303 | 588.3302 | -0.0000% |
| σ_tot 35 | 698.5938 | 698.5932 | -0.0001% |
| σ_tot 41 | 1392.6215 | 1392.6209 | -0.0000% |
| K1 = F3ETA 33 | 742.0951 | 742.0951 | -0.0000% |
| K1 = F3ETA 35 | 723.3823 | 723.3818 | -0.0001% |
| K1 = F3ETA 41 | 1704.6467 | 1704.6464 | -0.0000% |

Normalized residual (pull) of the two reclassified points, before and after:

| No. | Author | pull (SCR) | pull (SCA) |
|----:|--------|-----------:|-----------:|
| 75  | Pattenden | +0.41 | +0.38 |
| 192 | Anderson  | +0.65 | +0.62 |

## Interpretation

**The classification has no practical consequence for the evaluation.** Every physically relevant quantity — ABS, FIS, CA, the total cross section σ_tot, and the Westcott-folded K1 = F3ETA — for Pu-239 and for all other nuclides is stable to **< 0.002 %**, and χ² improves by a negligible 0.07. The only visibly moving parameter is **SCR(²³⁹Pu)**, which shifts by ≈ −0.6 %; it is itself weakly determined (≈ 16 % uncertainty) and does not feed the recommended thermal constants.

The reason is quantitative: the fitted Pu-239 scattering terms differ by only SCA − SCR ≈ 7.48 − 6.74 = **0.74 b**. Re-labelling a point from ABS+SCR to ABS+SCA changes its predicted total by exactly this ~0.74 b, which is negligible against the **±25 b (2.4 %)** uncertainties of Pattenden and Anderson. Both points are also weak constraints on ABS(²³⁹Pu) — which is pinned by many tighter entries (Bollinger, Safford, Palevsky, Spencer, Zimmerman at 0.5–1.3 %) — so moving 0.74 b of their value between the SCR and SCA channels barely perturbs the fit. Consistently, their pulls stay below 1σ in both cases (Pattenden +0.41→+0.38, Anderson +0.65→+0.62).

**Conclusion.** The SCR-vs-SCA assignment of Pattenden 1956 (No. 75) and Anderson 1945 (No. 192) is physically debatable but numerically immaterial at the round-3 level: the recommended Pu-239 constants and K1 are identical to within rounding whichever convention is adopted. There is therefore no need to resolve the question to finalise the fit; it can be settled on physical grounds (sample form / extinction corrections in the original experiments) without affecting the results.

## Reproducibility

  * Script: `evaluation/sensitivity_scr2sca_75_192.py`
  * Variant database: `axton1986/axton1986_data_capote_round3_scr2sca_75_192.json` (= round-3 DB with Nos. 75 and 192 changed to `(ABS 39)+SCA 39`)
  * Baseline: `axton1986/axton1986_data_capote_round3.json` and `evaluation/report_capote_round3.md`
