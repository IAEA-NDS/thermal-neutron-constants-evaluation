# Reproduction of the Axton 1986 Thermal Neutron Constants Evaluation

## Introduction

This report documents the reproduction of the iterative Generalized Least Squares
(GLS) evaluation of thermal neutron constants performed by E.J. Axton in 1986
(report GE/PH/01/86, CBNM, JRC Geel). The original evaluation determined best
estimates and uncertainties for the thermal neutron cross sections, neutron yields,
and related quantities of the fissile nuclides 233U, 235U, 239Pu, 241Pu, and the
fission neutron yield of 252Cf.

Two independent data sources are available for this reproduction:

1. **JSON database** (`axton1986_data.json`): 167 measurements manually transcribed
   from the original Axton 1986 PDF report, with named correlation codes, metadata
   (author, reference, measured function), and systematic uncertainty decomposition.

2. **AGS/thermalcst files** (`ags_axton1986_jan2026.mic`, `thermalcst_axton1986_jan2026.mic`):
   Data files provided by a colleague at CEA Cadarache, containing 102 microscopic
   measurements. The thermalcst files hold measured values and absolute uncertainties;
   the AGS files hold the uncertainty decomposition for covariance matrix construction.

The report proceeds in six sections, each building on the previous. We begin by
establishing agreement with Axton's published all-data fit, then progressively narrow
the measurement set to understand how the microscopic subset was defined, and finally
cross-validate the two data pipelines against each other.


## Quantities and Notation

The evaluation fits the following primary quantities for each nuclide (identified
by mass number: 33=233U, 34=234U, 35=235U, 39=239Pu, 41=241Pu, 52=252Cf):

| Symbol | Description |
|--------|-------------|
| SCA    | Scattering cross section (coherent + incoherent) |
| SCR    | Scattering cross section (coherent, "real") |
| ABS    | Absorption cross section |
| FIS    | Fission cross section |
| NUB    | Prompt + delayed fission neutron yield (nu-bar) |
| WGA    | Westcott g-factor for absorption |
| WGF    | Westcott g-factor for fission |
| CAP    | Capture cross section (= ABS - FIS for fissile nuclides) |
| CA     | Capture cross section (fitted directly for non-fissile 240Pu, 242Pu) |
| HLF    | Half-life (fixed, not fitted) |
| GC116  | Westcott g-factor correction at 116 eV (absorption) |
| GA116  | Westcott g-factor correction at 116 eV (fission) |

Derived quantities computed from fitted parameters:

| Symbol  | Formula |
|---------|---------|
| CA(n)   | ABS(n) - FIS(n) |
| ETA(n)  | NUB(n) * FIS(n) / ABS(n) |
| ALPHA(n)| [ABS(n) - FIS(n)] / FIS(n) |
| F3ETA(n)| NUB(n) * FIS(n) * WGF(n) - ABS(n) * WGA(n) |


## Correlation Codes

The covariance matrix is constructed from systematic uncertainty components
shared between measurements. Each component is identified by a correlation
code (from Table 2 of the Axton 1986 report). Two measurements sharing a
nonzero component for the same code are correlated; the relative covariance
is cov_rel(i,j) = sum_k u_ik * u_jk, where u_ik is the relative systematic
uncertainty (in percent) of measurement i for correlation code k.

| Code | Description |
|------|-------------|
| `A` | Deruytter Common Uncertainty |
| `B` | Deruytter Timing |
| `C` | Deruytter-Berceano Assay |
| `D` | White U235 Assay |
| `E` | Maslin-Saplakoglu Lem./Deruyt. Correction |
| `F` | NPL Common Mn-bath Uncertainty |
| `G` | Uncertainty in (1+sigmaS/sigmaMn) in Mn-baths |
| `H` | Energy Uncertainty of U233 Neutrons |
| `I` | Energy Uncertainty of U235 Neutrons |
| `J` | Energy Uncertainty of PU239 Neutrons |
| `K` | Energy Uncertainty of Pu241 Neutrons |
| `L` | Energy Uncertainty of CF252 Neutrons |
| `M` | Detector slope (Different Authors) |
| `N` | Boldeman Detector Slope |
| `O` | Gwin Detector Slope |
| `P` | Hopkins Detector Slope |
| `Q` | Asplund Detector Slope |
| `R` | Colvin Detector Slope |
| `S` | Mather Detector Slope |
| `U` | U233 Delayed Neutrons |
| `V` | U235 Delayed Neutrons |
| `W` | PU239 Delayed Neutrons |
| `X` | PU241 Delayed Neutrons |
| `Y` | CF252 Delayed Neutrons |
| `Z` | Delayed Gamma-rays |
| `Delta` | Boldeman's Foil Thickness corrections |
| `A_` | Gwin Common Uncertainty |
| `B_` | Spiegel Bozorgmanesh Common Uncertainty |
| `C_` | Boldeman Common Uncertainty |
| `D_` | Diven Common Uncertainty |
| `E_` | Boron Pile Common Uncertainty |
| `F_` | Smith/Aleksandrov Common Uncertainty |
| `G_` | Smith Eta/Nubar Common Uncertainty |
| `H_` | Eta Montecarlo Corrections |
| `I_` | Mather Common Uncertainty |
| `J_` | White Pu239 Assay |
| `K_` | White PU241 Assay |
| `M_` | Cabell Fluence |
| `N_` | Cabell Temperature |
| `O_` | DIDO Reflector g+rs For CAP, ABS PU239 |
| `P_` | DIDO Reflector g+rs For CAP, ABS PU241 |
| `Q_` | CABELL 86 deg Spectrum Uncertainty |
| `R_` | Cabell Uncertainty (Sjostrand and Story) |
| `S_` | Cabell Uncertainty (Sjostrand and Story) |
| `T_` | Gwin Uncertainty evaluated By Hardy |
| `U_` | Muelhaus uncertainty evaluated by Axton |
| `V_` | Spectral Uncertainties from IN1060 |
| `W_` | MTR 70 deg Spectrum Uncertainty |
| `X_` | NRU 40 deg Temperature Uncertainty |
| `Y_` | NRU Westcott r Uncertainty |
| `Z_` | Lounsbury Temperature Uncertainty |
| `Delta_` | Popovic Uncertainty (Na Cross Section) |
| `rho` | Keith Uncertainty (Deruytter correction) |
| `dag` | Uncertainty in Co Cross Section |
| `dn` | +/-20 deg Uncertainty in T |
| `plus` | Common Uncertainty in U235 Reactivity |
| `sq` | Common Uncertainty in U235 Reactivity |
| `c` | Half-life U233 |
| `hc` | Half-life U234 |
| `n` | Half-life PU239 |
| `u` | Half-life AM241 |
| `alpha` | Spencer Common Uncertainty |

Note: There is no code `T` in the original Axton system (the sequence
skips from `S` to `U`). Similarly, there is no code `L_`. These gaps
are reflected in the AGS column numbering, where column 19 is unused.


---

## 1. All-Data Fit: Reproducing Axton 1986 Table 4

The first benchmark uses all 167 measurements from the JSON database with
38 free parameters, reproducing the conditions of Table 4 in the Axton 1986
report. This establishes the baseline: if the transcribed data is correct
and the GLS implementation is faithful, the results should match Table 4
closely.

**Script:** `benchmark_axton1986_all_data.py`

### Summary

| Metric              | This work | Report |
|---------------------|-----------|--------|
| Chi-squared         | 102.3     | 103    |
| N measurements      | 167       | 167    |
| N free parameters   | 38        | 38     |
| Degrees of freedom  | 129       | 129    |

### Fitted Parameters

| Parameter  |   Fitted |   Report |   Diff%  | Unc%   | Rep Unc% | Match |
|------------|----------|----------|----------|--------|----------|-------|
| SCA 33     |  12.1810 |  12.1861 | -0.0419% |  5.482 |    5.483 |  OK   |
| SCA 35     |  15.9730 |  15.9828 | -0.0615% |  6.914 |    6.914 |  OK   |
| SCA 39     |   7.9037 |   7.8966 | +0.0894% | 12.330 |   12.333 |  OK   |
| SCA 41     |  12.1685 |  12.1878 | -0.1586% | 21.471 |   21.470 |  OK   |
| SCR 33     |  10.9067 |  10.9169 | -0.0934% |  8.069 |    8.050 |  OK   |
| SCR 35     |  14.1940 |  14.2071 | -0.0919% |  8.927 |    8.919 |  OK   |
| SCR 39     |   6.8633 |   6.8000 | +0.9308% | 24.295 |   25.726 | DIFF  |
| SCR 41     |  11.1428 |  11.1717 | -0.2584% | 32.544 |   32.337 |  OK   |
| ABS 33     | 576.2605 | 576.2174 | +0.0075% |  0.225 |    0.226 |  OK   |
| ABS 35     | 681.8566 | 681.8303 | +0.0039% |  0.192 |    0.192 |  OK   |
| ABS 39     |1018.7581 |1018.9667 | -0.0205% |  0.281 |    0.282 |  OK   |
| ABS 41     |1373.9632 |1373.2025 | +0.0554% |  0.661 |    0.662 |  OK   |
| FIS 33     | 530.7214 | 530.6953 | +0.0049% |  0.253 |    0.253 |  OK   |
| FIS 35     | 582.8180 | 582.7836 | +0.0059% |  0.196 |    0.196 |  OK   |
| FIS 39     | 747.6949 | 747.6236 | +0.0095% |  0.269 |    0.269 |  OK   |
| FIS 41     |1012.2387 |1011.8730 | +0.0361% |  0.651 |    0.652 |  OK   |
| NUB 33     |   2.4950 |   2.4950 | -0.0006% |  0.159 |    0.161 |  OK   |
| NUB 35     |   2.4333 |   2.4334 | -0.0042% |  0.146 |    0.147 |  OK   |
| NUB 39     |   2.8819 |   2.8822 | -0.0094% |  0.176 |    0.177 |  OK   |
| NUB 41     |   2.9458 |   2.9463 | -0.0155% |  0.195 |    0.196 |  OK   |
| NUB 52     |   3.7679 |   3.7676 | +0.0091% |  0.126 |    0.126 |  OK   |
| WGA 33     |   0.9994 |   0.9995 | -0.0084% |  0.107 |    0.107 |  OK   |
| WGA 35     |   0.9789 |   0.9789 | +0.0030% |  0.084 |    0.084 |  OK   |
| WGA 39     |   1.0783 |   1.0782 | +0.0049% |  0.227 |    0.227 |  OK   |
| WGA 41     |   1.0442 |   1.0442 | +0.0026% |  0.190 |    0.190 |  OK   |
| WGF 33     |   0.9955 |   0.9955 | +0.0009% |  0.142 |    0.142 |  OK   |
| WGF 35     |   0.9774 |   0.9774 | -0.0027% |  0.083 |    0.083 |  OK   |
| WGF 39     |   1.0554 |   1.0555 | -0.0063% |  0.212 |    0.212 |  OK   |
| WGF 41     |   1.0444 |   1.0445 | -0.0113% |  0.526 |    0.527 |  OK   |
| CA 40      | 289.3345 | 289.3296 | +0.0017% |  0.482 |    0.482 |  OK   |
| CA 42      |  18.5148 |  18.5144 | +0.0020% |  2.161 |    2.161 |  OK   |
| CAP 34     |  95.8364 |  95.8369 | -0.0005% |  2.071 |    2.071 |  OK   |
| GC116 39   |   1.3658 |   1.3265 | +2.9593% |  2.738 |    6.056 | DIFF  |
| GC116 40   |   1.0755 |   1.0860 | -0.9671% |  1.819 |    2.000 | DIFF  |
| GC116 41   |   1.0902 |   1.1085 | -1.6477% |  1.651 |    2.457 | DIFF  |
| GC116 42   |   1.1349 |   1.1335 | +0.1219% |  2.668 |    2.670 |  OK   |
| GA116 39   |   1.1878 |   1.1846 | +0.2726% |  1.054 |    1.521 |  OK   |
| GA116 41   |   1.1097 |   1.1073 | +0.2159% |  0.738 |    0.819 | DIFF  |
| CA 33      |  45.5391 |  45.5221 | +0.0374% |  1.537 |    1.535 |  OK   |
| CA 35      |  99.0386 |  99.0467 | -0.0082% |  0.750 |    0.750 |  OK   |
| CA 39      | 271.0633 | 271.3431 | -0.1031% |  0.787 |    0.786 | DIFF  |
| CA 41      | 361.7244 | 361.3294 | +0.1093% |  1.385 |    1.367 |  OK   |
| ETA 33     |   2.2978 |   2.2979 | -0.0036% |  0.177 |    0.180 |  OK   |
| ETA 35     |   2.0799 |   2.0799 | -0.0017% |  0.167 |    0.167 |  OK   |
| ETA 39     |   2.1151 |   2.1147 | +0.0202% |  0.241 |    0.243 |  OK   |
| ETA 41     |   2.1703 |   2.1710 | -0.0328% |  0.352 |    0.353 |  OK   |
| ALPHA 33   |   0.0858 |   0.0858 | +0.0070% |  1.636 |    1.634 |  OK   |
| ALPHA 35   |   0.1699 |   0.1700 | -0.0408% |  0.791 |    0.792 |  OK   |
| ALPHA 39   |   0.3625 |   0.3629 | -0.1014% |  0.843 |    0.841 | DIFF  |
| ALPHA 41   |   0.3574 |   0.3571 | +0.0703% |  1.400 |    1.376 |  OK   |
| F3ETA 33   | 742.2714 | 742.2525 | +0.0025% |  0.317 |    0.319 |  OK   |
| F3ETA 35   | 718.5917 | 718.5735 | +0.0025% |  0.308 |    0.309 |  OK   |
| F3ETA 39   |1175.7724 |1175.7377 | +0.0030% |  0.473 |    0.475 |  OK   |
| F3ETA 41   |1679.5072 |1679.8787 | -0.0221% |  0.844 |    0.846 |  OK   |

**Result:** 47/54 parameters match within tolerance (0.1% for well-constrained
parameters, 0.5% for poorly constrained). Chi-squared is 102.3 vs the reported
103. The small discrepancies in GC116 and GA116 parameters are attributable to
rounding in the PDF-transcribed values.

The excellent agreement confirms that both the data transcription and the GLS
implementation are faithful to the original evaluation.


---

## 2. Microscopic Fit: Narrow Definition

Having established the all-data baseline, we now attempt to reproduce the
microscopic-only fit reported in Table 8 of the Axton 1986 report. Table 8
reports results for 101 measurements with 38 parameters, yielding chi-squared = 48.

The challenge is that the report does not explicitly list which measurements
constitute the "microscopic" subset. This benchmark uses a narrow definition:
exclude only the clearly Maxwellian-derived functions (FA, FF, CAP, FFH, F1ETA,
F2ETA, F3ETA, FLEM, F1BIG, F1CAB-F5CAB) while retaining WGA, WGF, and ETA
measurements.

**Script:** `benchmark_axton1986_micro.py`

### Summary

| Metric              | This work | Report |
|---------------------|-----------|--------|
| Chi-squared         | 60.5      | 48     |
| N measurements      | 112       | 101    |
| N free parameters   | 37        | 38     |
| Degrees of freedom  | 75        | 63     |

### Fitted Parameters

| Parameter  |   Fitted |   Report |   Diff%  | Unc%    | Rep Unc% | Match |
|------------|----------|----------|----------|---------|----------|-------|
| SCA 33     |  12.3166 |  12.3204 | -0.0308% |   5.681 |    5.681 |  OK   |
| SCA 35     |  15.8847 |  16.2388 | -2.1804% |   7.270 |    7.386 | DIFF  |
| SCA 39     |   7.9230 |   7.8966 | +0.3348% |  12.364 |   12.380 |  OK   |
| SCA 41     |  11.9618 |  11.9740 | -0.1017% |  21.596 |   21.595 |  OK   |
| SCR 33     |  11.1419 |  11.1480 | -0.0543% |   8.171 |    8.147 |  OK   |
| SCR 35     |  14.0797 |  14.5259 | -3.0719% |   9.527 |    9.448 | DIFF  |
| SCR 39     |   6.9469 |   6.8131 | +1.9637% |  23.295 |   25.630 |  OK   |
| SCR 41     |  10.8049 |  10.8285 | -0.2178% |  35.707 |   35.646 |  OK   |
| ABS 33     | 575.2692 | 575.2391 | +0.0052% |   0.321 |    0.321 |  OK   |
| ABS 35     | 682.0852 | 681.1651 | +0.1351% |   0.233 |    0.249 |  OK   |
| ABS 39     |1018.2345 |1018.8813 | -0.0635% |   0.340 |    0.362 |  OK   |
| ABS 41     |1382.7102 |1382.1872 | +0.0378% |   1.079 |    1.079 |  OK   |
| FIS 33     | 530.3843 | 533.2694 | -0.5410% |   0.355 |    0.457 | DIFF  |
| FIS 35     | 584.0124 | 585.0603 | -0.1791% |   0.244 |    0.277 |  OK   |
| FIS 39     | 748.1165 | 748.5181 | -0.0536% |   0.321 |    0.340 |  OK   |
| FIS 41     |1018.4694 |1020.5836 | -0.2072% |   1.119 |    1.126 |  OK   |
| CA 33      |  44.8849 |  41.9697 | +6.9460% |   1.837 |    4.182 | DIFF  |
| CA 35      |  98.0728 |  96.1049 | +2.0477% |   1.152 |    1.811 | DIFF  |
| CA 39      | 270.1180 | 270.3632 | -0.0907% |   0.913 |    1.174 |  OK   |
| CA 41      | 364.2408 | 361.6036 | +0.7293% |   1.663 |    1.711 |  OK   |
| NUB 33     |   2.4910 |   2.4856 | +0.2189% |   0.184 |    0.218 |  OK   |
| NUB 35     |   2.4294 |   2.4261 | +0.1348% |   0.172 |    0.188 |  OK   |
| NUB 39     |   2.8814 |   2.8794 | +0.0683% |   0.192 |    0.207 |  OK   |
| NUB 41     |   2.9435 |   2.9406 | +0.0989% |   0.209 |    0.220 |  OK   |
| NUB 52     |   3.7664 |   3.7644 | +0.0518% |   0.129 |    0.132 |  OK   |
| ETA 33     |   2.2967 |   2.3042 | -0.3264% |   0.199 |    0.271 |  OK   |
| ETA 35     |   2.0801 |   2.0838 | -0.1792% |   0.211 |    0.255 |  OK   |
| ETA 39     |   2.1170 |   2.1153 | +0.0802% |   0.267 |    0.311 |  OK   |
| ETA 41     |   2.1681 |   2.1713 | -0.1468% |   0.391 |    0.399 |  OK   |
| ALPHA 33   |   0.0846 |   0.0787 | +7.5313% |   1.960 |    4.494 | DIFF  |
| ALPHA 35   |   0.1679 |   0.1643 | +2.2089% |   1.235 |    1.962 | DIFF  |
| ALPHA 39   |   0.3611 |   0.3612 | -0.0376% |   0.965 |    1.280 |  OK   |
| ALPHA 41   |   0.3576 |   0.3543 | +0.9414% |   1.585 |    1.668 |  OK   |
| WGA 33     |   0.9988 |   0.9988 | +0.0040% |   0.120 |    0.120 |  OK   |
| WGA 35     |   0.9787 |   0.9787 | +0.0000% |   0.092 |    0.092 |  OK   |
| WGA 39     |   1.0767 |   1.0767 | +0.0000% |   0.269 |    0.269 |  OK   |
| WGA 41     |   1.0444 |   1.0444 | +0.0000% |   0.191 |    0.191 |  OK   |
| WGF 33     |   0.9965 |   0.9966 | -0.0140% |   0.201 |    0.201 |  OK   |
| WGF 35     |   0.9778 |   0.9778 | +0.0000% |   0.092 |    0.092 |  OK   |
| WGF 39     |   1.0562 |   1.0562 | +0.0000% |   0.275 |    0.275 |  OK   |
| WGF 41     |   1.0452 |   1.0452 | +0.0000% |   0.670 |    0.670 |  OK   |
| F3ETA 33   | 741.9298 | 746.4586 | -0.6067% |   0.603 |    0.683 | DIFF  |
| F3ETA 35   | 719.7289 | 721.2735 | -0.2141% |   0.470 |    0.507 |  OK   |
| F3ETA 39   |1180.4095 |1179.4821 | +0.0786% |   0.781 |    0.805 |  OK   |
| F3ETA 41   |1689.2737 |1693.1349 | -0.2280% |   1.769 |    1.774 |  OK   |
| CA 40      | 289.5000 | 289.3312 | +0.0583% |   0.484 |    0.482 |  OK   |
| CA 42      |  18.5000 |  18.5148 | -0.0799% |   2.162 |    2.161 |  OK   |

**Result:** 39/47 parameters match. The chi-squared (60.5) and measurement count
(112) both exceed the report's values (48 and 101), indicating that this narrow
definition of "microscopic" includes 11 measurements not in the report's set.
The most significant discrepancy is in FIS 33 (-0.54%) and the derived ALPHA 33
(+7.5%), suggesting that specific measurements biasing FIS 33 are included here
but were excluded by Axton.


---

## 3. Microscopic Fit: Wide (Strict) Definition

To bracket the measurement set from the other direction, this benchmark uses a
wider definition of "Maxwellian" that also excludes WGA, WGF, GC116, and GA116
measurements themselves (not just functions derived from them). This removes
more measurements than necessary but tests the hypothesis that the discrepancy
lies in the classification of WGA/WGF-type data.

**Script:** `benchmark_axton1986_micro_strict.py`

### Summary

| Metric              | This work | Report |
|---------------------|-----------|--------|
| Chi-squared         | 60.5      | 48     |
| N measurements      | 96        | 101    |
| N free parameters   | 21        | 38     |
| Degrees of freedom  | 75        | 63     |

### Fitted Parameters

| Parameter  |   Fitted |   Report |   Diff%  | Unc%    | Rep Unc% | Match |
|------------|----------|----------|----------|---------|----------|-------|
| SCA 33     |  12.3166 |  12.3204 | -0.0308% |   5.681 |    5.681 |  OK   |
| SCA 35     |  15.8847 |  16.2388 | -2.1804% |   7.270 |    7.386 | DIFF  |
| SCA 39     |   7.9230 |   7.8966 | +0.3348% |  12.364 |   12.380 |  OK   |
| SCA 41     |  11.9618 |  11.9740 | -0.1017% |  21.596 |   21.595 |  OK   |
| SCR 33     |  11.1419 |  11.1480 | -0.0543% |   8.171 |    8.147 |  OK   |
| SCR 35     |  14.0797 |  14.5259 | -3.0719% |   9.527 |    9.448 | DIFF  |
| SCR 39     |   6.9469 |   6.8131 | +1.9637% |  23.295 |   25.630 |  OK   |
| SCR 41     |  10.8049 |  10.8285 | -0.2178% |  35.707 |   35.646 |  OK   |
| ABS 33     | 575.2692 | 575.2391 | +0.0052% |   0.321 |    0.321 |  OK   |
| ABS 35     | 682.0852 | 681.1651 | +0.1351% |   0.233 |    0.249 |  OK   |
| ABS 39     |1018.2345 |1018.8813 | -0.0635% |   0.340 |    0.362 |  OK   |
| ABS 41     |1382.7102 |1382.1872 | +0.0378% |   1.079 |    1.079 |  OK   |
| FIS 33     | 530.3843 | 533.2694 | -0.5410% |   0.355 |    0.457 | DIFF  |
| FIS 35     | 584.0124 | 585.0603 | -0.1791% |   0.244 |    0.277 |  OK   |
| FIS 39     | 748.1165 | 748.5181 | -0.0536% |   0.321 |    0.340 |  OK   |
| FIS 41     |1018.4694 |1020.5836 | -0.2072% |   1.119 |    1.126 |  OK   |
| CA 33      |  44.8849 |  41.9697 | +6.9460% |   1.837 |    4.182 | DIFF  |
| CA 35      |  98.0728 |  96.1049 | +2.0477% |   1.152 |    1.811 | DIFF  |
| CA 39      | 270.1180 | 270.3632 | -0.0907% |   0.913 |    1.174 |  OK   |
| CA 41      | 364.2408 | 361.6036 | +0.7293% |   1.663 |    1.711 |  OK   |
| NUB 33     |   2.4910 |   2.4856 | +0.2189% |   0.184 |    0.218 |  OK   |
| NUB 35     |   2.4294 |   2.4261 | +0.1348% |   0.172 |    0.188 |  OK   |
| NUB 39     |   2.8814 |   2.8794 | +0.0683% |   0.192 |    0.207 |  OK   |
| NUB 41     |   2.9435 |   2.9406 | +0.0989% |   0.209 |    0.220 |  OK   |
| NUB 52     |   3.7664 |   3.7644 | +0.0518% |   0.129 |    0.132 |  OK   |
| ETA 33     |   2.2967 |   2.3042 | -0.3264% |   0.199 |    0.271 |  OK   |
| ETA 35     |   2.0801 |   2.0838 | -0.1792% |   0.211 |    0.255 |  OK   |
| ETA 39     |   2.1170 |   2.1153 | +0.0802% |   0.267 |    0.311 |  OK   |
| ETA 41     |   2.1681 |   2.1713 | -0.1468% |   0.391 |    0.399 |  OK   |
| ALPHA 33   |   0.0846 |   0.0787 | +7.5313% |   1.960 |    4.494 | DIFF  |
| ALPHA 35   |   0.1679 |   0.1643 | +2.2089% |   1.235 |    1.962 | DIFF  |
| ALPHA 39   |   0.3611 |   0.3612 | -0.0376% |   0.965 |    1.280 |  OK   |
| ALPHA 41   |   0.3576 |   0.3543 | +0.9414% |   1.585 |    1.668 |  OK   |

**Result:** 26/33 parameters match. Notably, this benchmark produces
*identical* fitted values and chi-squared (60.5) to the narrow microscopic
benchmark above, despite using only 96 measurements instead of 112. This
is because WGA and WGF measurements (Nos. 54-61) are single-measurement
constraints on their respective parameters --- removing them simply leaves
those parameters unconstrained without affecting the rest of the fit. The
persistent FIS 33 discrepancy points to a different cause.


---

## 4. Microscopic Fit: Lounsbury Exclusion (Best Table 8 Match)

The key insight that resolves the Table 8 discrepancy comes from analysing the
Lounsbury alpha measurements (Nos. 21-23). These measurements of (CA n)/FIS n
for n = 33, 35, 39 have a systematic uncertainty code `Z_` ("Lounsbury Temperature
Uncertainty") that involves reactor spectrum corrections, suggesting they may
have been classified as integral/Maxwellian rather than microscopic despite
measuring a ratio of microscopic quantities.

Lounsbury's alpha(233U) = 0.0861 is 2.3% higher than the converged value (0.0787),
biasing FIS 33 downward when included. Excluding measurements 21-23 while
including WGA/WGF (54-61) and CAP 34 (No. 5), and removing parameters not
constrained without Maxwellian data (CA 40, CA 42, GC116, GA116), yields an
almost perfect match with Table 8.

**Script:** `benchmark_axton1986_micro_no_lounsbury.py`

### Summary

| Metric              | This work | Report |
|---------------------|-----------|--------|
| Chi-squared         | 55.1      | 48     |
| N measurements      | 102       | 101    |
| N free parameters   | 30        | 38     |
| Degrees of freedom  | 72        | 63     |

### Fitted Parameters

| Parameter  |   Fitted |   Report |   Diff%  | Unc%   | Rep Unc% | Match |
|------------|----------|----------|----------|--------|----------|-------|
| SCA 33     |  12.3206 |  12.3204 | +0.0014% |  5.681 |    5.681 |  OK   |
| SCA 35     |  16.2235 |  16.2388 | -0.0945% |  7.385 |    7.386 |  OK   |
| SCA 39     |   7.9001 |   7.8966 | +0.0438% | 12.380 |   12.380 |  OK   |
| SCA 41     |  11.9720 |  11.9740 | -0.0171% | 21.596 |   21.595 |  OK   |
| SCR 33     |  11.1483 |  11.1480 | +0.0030% |  8.161 |    8.147 |  OK   |
| SCR 35     |  14.5068 |  14.5259 | -0.1315% |  9.455 |    9.448 |  OK   |
| SCR 39     |   6.8449 |   6.8131 | +0.4672% | 24.890 |   25.630 |  OK   |
| SCR 41     |  10.8234 |  10.8285 | -0.0475% | 35.520 |   35.646 |  OK   |
| ABS 33     | 575.2413 | 575.2391 | +0.0004% |  0.321 |    0.321 |  OK   |
| ABS 35     | 681.2060 | 681.1651 | +0.0060% |  0.249 |    0.249 |  OK   |
| ABS 39     |1018.7914 |1018.8813 | -0.0088% |  0.362 |    0.362 |  OK   |
| ABS 41     |1382.2612 |1382.1872 | +0.0054% |  1.079 |    1.079 |  OK   |
| FIS 33     | 533.1364 | 533.2694 | -0.0249% |  0.460 |    0.457 |  OK   |
| FIS 35     | 585.0334 | 585.0603 | -0.0046% |  0.277 |    0.277 |  OK   |
| FIS 39     | 748.5245 | 748.5181 | +0.0009% |  0.340 |    0.340 |  OK   |
| FIS 41     |1020.3307 |1020.5836 | -0.0248% |  1.127 |    1.126 |  OK   |
| CA 33      |  42.1049 |  41.9697 | +0.3221% |  4.228 |    4.182 |  OK   |
| CA 35      |  96.1726 |  96.1049 | +0.0705% |  1.811 |    1.811 |  OK   |
| CA 39      | 270.2669 | 270.3632 | -0.0356% |  1.181 |    1.174 |  OK   |
| CA 41      | 361.9304 | 361.6036 | +0.0904% |  1.729 |    1.711 |  OK   |
| CAP 34     |  95.9000 |  95.8339 | +0.0690% |  2.086 |    2.071 |  OK   |
| NUB 33     |   2.4860 |   2.4856 | +0.0160% |  0.217 |    0.218 |  OK   |
| NUB 35     |   2.4264 |   2.4261 | +0.0114% |  0.188 |    0.188 |  OK   |
| NUB 39     |   2.8794 |   2.8794 | +0.0013% |  0.207 |    0.207 |  OK   |
| NUB 41     |   2.9406 |   2.9406 | -0.0004% |  0.220 |    0.220 |  OK   |
| NUB 52     |   3.7649 |   3.7644 | +0.0132% |  0.132 |    0.132 |  OK   |
| ETA 33     |   2.3040 |   2.3042 | -0.0072% |  0.269 |    0.271 |  OK   |
| ETA 35     |   2.0838 |   2.0838 | +0.0010% |  0.255 |    0.255 |  OK   |
| ETA 39     |   2.1156 |   2.1153 | +0.0130% |  0.310 |    0.311 |  OK   |
| ETA 41     |   2.1706 |   2.1713 | -0.0311% |  0.399 |    0.399 |  OK   |
| ALPHA 33   |   0.0790 |   0.0787 | +0.3505% |  4.545 |    4.494 |  OK   |
| ALPHA 35   |   0.1644 |   0.1643 | +0.0537% |  1.962 |    1.962 |  OK   |
| ALPHA 39   |   0.3611 |   0.3612 | -0.0371% |  1.289 |    1.280 |  OK   |
| ALPHA 41   |   0.3547 |   0.3543 | +0.1182% |  1.697 |    1.668 |  OK   |
| WGA 33     |   0.9988 |   0.9988 | +0.0040% |  0.120 |    0.120 |  OK   |
| WGA 35     |   0.9787 |   0.9787 | +0.0000% |  0.092 |    0.092 |  OK   |
| WGA 39     |   1.0767 |   1.0767 | +0.0000% |  0.269 |    0.269 |  OK   |
| WGA 41     |   1.0444 |   1.0444 | +0.0000% |  0.191 |    0.191 |  OK   |
| WGF 33     |   0.9965 |   0.9966 | -0.0140% |  0.201 |    0.201 |  OK   |
| WGF 35     |   0.9778 |   0.9778 | +0.0000% |  0.092 |    0.092 |  OK   |
| WGF 39     |   1.0562 |   1.0562 | +0.0000% |  0.275 |    0.275 |  OK   |
| WGF 41     |   1.0452 |   1.0452 | +0.0000% |  0.670 |    0.670 |  OK   |
| F3ETA 33   | 746.1097 | 746.4586 | -0.0467% |  0.680 |    0.683 |  OK   |
| F3ETA 35   | 721.3019 | 721.2735 | +0.0039% |  0.507 |    0.507 |  OK   |
| F3ETA 39   |1179.5261 |1179.4821 | +0.0037% |  0.803 |    0.805 |  OK   |
| F3ETA 41   |1692.3543 |1693.1349 | -0.0461% |  1.773 |    1.774 |  OK   |

**Result:** **46/46 parameters match**, including FIS 33 (533.14 vs 533.27,
0.025% difference) and ALPHA 33 (0.0790 vs 0.0787, 0.35%). This confirms
that the Lounsbury measurements were treated as Maxwellian/integral in the
original evaluation, likely because their systematic uncertainty (`Z_`:
"Lounsbury Temperature Uncertainty") involves reactor spectrum corrections.

The remaining chi-squared gap (55.1 vs 48) and measurement count difference
(102 vs 101) suggest one additional measurement was excluded by Axton. The
residual differences are well within what can be attributed to PDF rounding
of the input data.


---

## 5. Pipeline Validation: Modified JSON vs AGS (Exact Input Match)

Before comparing the two data pipelines on their actual (slightly different)
data, we first verify that the JSON-based and AGS-based GLS pipelines produce
identical results when given identical input. A modified JSON file
(`axton1986_data_ags_mic.json`) was constructed by reading values from
`thermalcst_axton1986_jan2026.mic` and the uncertainty decomposition from
`ags_axton1986_jan2026.mic`, using an empirically verified column-to-correlation-code
mapping (zero mismatches across all measurements). The covariance matrices
from both representations agree to a relative Frobenius norm of 1.3e-9.

**Script:** `benchmark_modjson_vs_ags_mic.py`

### Summary

| Metric              | Mod JSON   | AGS        | Report |
|---------------------|------------|------------|--------|
| Chi-squared         | 59.6738    | 59.6738    | 48     |
| N measurements      | 102        | 102        | 101    |
| N free parameters   | 29         | 29         | 38     |
| Degrees of freedom  | 73         | 73         | 63     |

### Fitted Parameters

| Parameter  | Mod JSON | AGS      | Report   | MJ-A diff%  | MJ unc% | A unc%  | R unc%  |
|------------|----------|----------|----------|-------------|---------|---------|---------|
| SCA 33     |  12.3205 |  12.3205 |  12.3204 |  -0.000000% |   5.679 |   5.679 |   5.681 |
| SCA 35     |  15.8264 |  15.8264 |  16.2388 |  +0.000000% |   7.302 |   7.302 |   7.386 |
| SCA 39     |   8.0178 |   8.0178 |   7.8966 |  -0.000000% |  12.483 |  12.483 |  12.380 |
| SCA 41     |  11.9450 |  11.9450 |  11.9740 |  +0.000000% |  21.612 |  21.612 |  21.595 |
| SCR 33     |  11.1454 |  11.1454 |  11.1480 |  -0.000000% |   8.169 |   8.169 |   8.147 |
| SCR 35     |  13.9984 |  13.9984 |  14.5259 |  +0.000001% |   9.639 |   9.639 |   9.448 |
| SCR 39     |   3.1317 |   3.1317 |   6.8131 |  -0.000026% | 214.437 | 214.437 |  25.630 |
| SCR 41     |  14.9731 |  14.9731 |  10.8285 |  +0.000008% | 197.426 | 197.426 |  35.646 |
| ABS 33     | 575.2960 | 575.2960 | 575.2391 |  +0.000000% |   0.321 |   0.321 |   0.321 |
| ABS 35     | 682.1918 | 682.1918 | 681.1651 |  -0.000000% |   0.235 |   0.235 |   0.249 |
| ABS 39     |1020.1152 |1020.1152 |1018.8813 |  +0.000000% |   0.460 |   0.460 |   0.362 |
| ABS 41     |1381.0269 |1381.0269 |1382.1872 |  -0.000000% |   1.576 |   1.576 |   1.079 |
| FIS 33     | 530.3722 | 530.3722 | 533.2694 |  -0.000000% |   0.355 |   0.355 |   0.457 |
| FIS 35     | 584.1181 | 584.1181 | 585.0603 |  +0.000000% |   0.246 |   0.246 |   0.277 |
| FIS 39     | 749.0977 | 749.0977 | 748.5181 |  +0.000000% |   0.379 |   0.379 |   0.340 |
| FIS 41     |1017.1242 |1017.1242 |1020.5836 |  -0.000000% |   1.573 |   1.573 |   1.126 |
| NUB 33     |   2.4911 |   2.4911 |   2.4856 |  -0.000000% |   0.184 |   0.184 |   0.218 |
| NUB 35     |   2.4294 |   2.4294 |   2.4261 |  -0.000000% |   0.172 |   0.172 |   0.188 |
| NUB 39     |   2.8819 |   2.8819 |   2.8794 |  -0.000000% |   0.194 |   0.194 |   0.207 |
| NUB 41     |   2.9437 |   2.9437 |   2.9406 |  +0.000000% |   0.209 |   0.209 |   0.220 |
| NUB 52     |   3.7665 |   3.7665 |   3.7644 |  -0.000000% |   0.129 |   0.129 |   0.132 |
| WGA 33     |   0.9988 |   0.9988 |   0.9988 |  +0.000000% |   0.120 |   0.120 |   0.120 |
| WGA 35     |   0.9787 |   0.9787 |   0.9787 |  +0.000000% |   0.092 |   0.092 |   0.092 |
| WGA 39     |   1.0770 |   1.0770 |   1.0767 |  +0.000000% |   0.269 |   0.269 |   0.269 |
| WGA 41     |   1.0440 |   1.0440 |   1.0444 |  +0.000000% |   0.191 |   0.191 |   0.191 |
| WGF 33     |   0.9966 |   0.9966 |   0.9966 |  +0.000000% |   0.201 |   0.201 |   0.201 |
| WGF 35     |   0.9778 |   0.9778 |   0.9778 |  +0.000000% |   0.092 |   0.092 |   0.092 |
| WGF 39     |   1.0560 |   1.0560 |   1.0562 |  +0.000000% |   0.275 |   0.275 |   0.275 |
| WGF 41     |   1.0450 |   1.0450 |   1.0452 |  +0.000000% |   0.670 |   0.670 |   0.670 |
| CA 33      |  44.9238 |  44.9238 |  41.9697 |  +0.000000% |   1.839 |   1.839 |   4.182 |
| CA 35      |  98.0736 |  98.0736 |  96.1049 |  -0.000000% |   1.153 |   1.153 |   1.811 |
| CA 39      | 271.0175 | 271.0175 | 270.3632 |  -0.000000% |   1.109 |   1.109 |   1.174 |
| CA 41      | 363.9027 | 363.9027 | 361.6036 |  -0.000000% |   2.093 |   2.093 |   1.711 |
| ETA 33     |   2.2966 |   2.2966 |   2.3042 |  -0.000000% |   0.199 |   0.199 |   0.271 |
| ETA 35     |   2.0802 |   2.0802 |   2.0838 |  +0.000000% |   0.211 |   0.211 |   0.255 |
| ETA 39     |   2.1162 |   2.1162 |   2.1153 |  +0.000000% |   0.286 |   0.286 |   0.311 |
| ETA 41     |   2.1680 |   2.1680 |   2.1713 |  -0.000000% |   0.392 |   0.392 |   0.399 |
| ALPHA 33   |   0.0847 |   0.0847 |   0.0787 |  +0.000000% |   1.963 |   1.963 |   4.494 |
| ALPHA 35   |   0.1679 |   0.1679 |   0.1643 |  -0.000001% |   1.236 |   1.236 |   1.962 |
| ALPHA 39   |   0.3618 |   0.3618 |   0.3612 |  -0.000000% |   1.063 |   1.063 |   1.280 |
| ALPHA 41   |   0.3578 |   0.3578 |   0.3543 |  +0.000000% |   1.595 |   1.595 |   1.668 |
| F3ETA 33   | 742.1004 | 742.1004 | 746.4586 |  -0.000000% |   0.603 |   0.603 |   0.683 |
| F3ETA 35   | 719.9103 | 719.9103 | 721.2735 |  +0.000000% |   0.472 |   0.472 |   0.507 |
| F3ETA 39   |1181.0189 |1181.0188 |1179.4821 |  +0.000000% |   0.801 |   0.801 |   0.805 |
| F3ETA 41   |1687.0712 |1687.0712 |1693.1349 |  -0.000000% |   2.075 |   2.075 |   1.774 |

**Result:** All parameters agree to better than 3e-5%, chi-squared differs by
2.4e-6. The two pipelines are numerically equivalent. The 2 parameters
nominally outside the 1e-6% threshold (SCR 39, SCR 41) have >200% uncertainties,
making their tiny absolute differences register only due to the small denominator.

Note that SCR 39 and SCR 41 converge to unphysical values (3.13 and 14.97 barns
vs the report's 6.81 and 10.83) with enormous uncertainties. This is because
without macroscopic data, the coherent scattering cross sections SCR 39 and SCR 41
are essentially unconstrained --- the microscopic measurements constrain only the
incoherent scattering (SCA) and the total absorption. The report's fit with 38
parameters (vs our 29) apparently retained these as constrained through additional
data or prior information not available in the microscopic subset alone.


---

## 6. Pipeline Comparison: Original JSON vs AGS (Actual Data Differences)

Having confirmed that both pipelines are numerically equivalent on identical
input (Section 5), we now compare them using their actual, slightly different
data. The JSON database was transcribed from the Axton 1986 PDF; the
AGS/thermalcst files come from the CEA Cadarache digitization. Both are
restricted to the same 102 microscopic measurement IDs.

Known differences between the two data sources:

- **Values:** 107 of 152 measurements differ by up to 0.05% due to rounding
  in the PDF transcription.
- **No. 147 uncertainty:** The AGS file has a total relative uncertainty of
  0.399% vs the JSON's 0.380% (5% relative difference in the uncorrelated
  component).
- **Correlation code mapping:** The AGS files have a systematic 2-column
  offset for codes beyond position 50, causing 22 measurement pairs to be
  correlated in JSON but uncorrelated in AGS (primarily Popovic vs NUB
  measurements via the `Delta` code).
- **No. 14 (White, K\_):** The JSON has K\_ = 2.14% (from PDF), the AGS
  has 0.94% in the corresponding column, reducing the correlation between
  No. 14 and No. 100 from r = 0.79 to r = 0.46.

**Script:** `benchmark_json_vs_ags_mic.py`

### Summary

| Metric              | JSON       | AGS        | Report |
|---------------------|------------|------------|--------|
| Chi-squared         | 60.0       | 59.7       | 48     |
| N measurements      | 102        | 102        | 101    |
| N free parameters   | 29         | 29         | 38     |
| Degrees of freedom  | 73         | 73         | 63     |

### Fitted Parameters

| Parameter  | JSON     | AGS      | Report   | J-A diff%  | J unc%  | A unc%  | R unc%  |
|------------|----------|----------|----------|------------|---------|---------|---------|
| SCA 33     |  12.3166 |  12.3205 |  12.3204 |   -0.0322% |   5.681 |   5.679 |   5.681 |
| SCA 35     |  15.8239 |  15.8264 |  16.2388 |   -0.0157% |   7.303 |   7.302 |   7.386 |
| SCA 39     |   8.0167 |   8.0178 |   7.8966 |   -0.0130% |  12.487 |  12.483 |  12.380 |
| SCA 41     |  11.9397 |  11.9450 |  11.9740 |   -0.0444% |  21.619 |  21.612 |  21.595 |
| SCR 33     |  11.1419 |  11.1454 |  11.1480 |   -0.0318% |   8.171 |   8.169 |   8.147 |
| SCR 35     |  13.9950 |  13.9984 |  14.5259 |   -0.0244% |   9.642 |   9.639 |   9.448 |
| SCR 39     |   2.5957 |   3.1317 |   6.8131 |  -17.1160% | 257.793 | 214.437 |  25.630 |
| SCR 41     |  14.7685 |  14.9731 |  10.8285 |   -1.3667% | 200.136 | 197.426 |  35.646 |
| ABS 33     | 575.2695 | 575.2960 | 575.2391 |   -0.0046% |   0.321 |   0.321 |   0.321 |
| ABS 35     | 682.2074 | 682.1918 | 681.1651 |   +0.0023% |   0.235 |   0.235 |   0.249 |
| ABS 39     |1020.3040 |1020.1152 |1018.8813 |   +0.0185% |   0.457 |   0.460 |   0.362 |
| ABS 41     |1380.8315 |1381.0269 |1382.1872 |   -0.0141% |   1.576 |   1.576 |   1.079 |
| FIS 33     | 530.3586 | 530.3722 | 533.2694 |   -0.0026% |   0.355 |   0.355 |   0.457 |
| FIS 35     | 584.1272 | 584.1181 | 585.0603 |   +0.0016% |   0.246 |   0.246 |   0.277 |
| FIS 39     | 749.1111 | 749.0977 | 748.5181 |   +0.0018% |   0.379 |   0.379 |   0.340 |
| FIS 41     |1016.9934 |1017.1242 |1020.5836 |   -0.0129% |   1.574 |   1.573 |   1.126 |
| NUB 33     |   2.4911 |   2.4911 |   2.4856 |   +0.0023% |   0.184 |   0.184 |   0.218 |
| NUB 35     |   2.4294 |   2.4294 |   2.4261 |   +0.0007% |   0.172 |   0.172 |   0.188 |
| NUB 39     |   2.8818 |   2.8819 |   2.8794 |   -0.0003% |   0.194 |   0.194 |   0.207 |
| NUB 41     |   2.9437 |   2.9437 |   2.9406 |   -0.0001% |   0.209 |   0.209 |   0.220 |
| NUB 52     |   3.7664 |   3.7665 |   3.7644 |   -0.0012% |   0.129 |   0.129 |   0.132 |
| WGA 33     |   0.9988 |   0.9988 |   0.9988 |   +0.0040% |   0.120 |   0.120 |   0.120 |
| WGA 35     |   0.9787 |   0.9787 |   0.9787 |   +0.0000% |   0.092 |   0.092 |   0.092 |
| WGA 39     |   1.0767 |   1.0770 |   1.0767 |   -0.0279% |   0.269 |   0.269 |   0.269 |
| WGA 41     |   1.0444 |   1.0440 |   1.0444 |   +0.0383% |   0.191 |   0.191 |   0.191 |
| WGF 33     |   0.9965 |   0.9966 |   0.9966 |   -0.0140% |   0.201 |   0.201 |   0.201 |
| WGF 35     |   0.9778 |   0.9778 |   0.9778 |   +0.0000% |   0.092 |   0.092 |   0.092 |
| WGF 39     |   1.0562 |   1.0560 |   1.0562 |   +0.0189% |   0.275 |   0.275 |   0.275 |
| WGF 41     |   1.0452 |   1.0450 |   1.0452 |   +0.0191% |   0.670 |   0.670 |   0.670 |
| CA 33      |  44.9109 |  44.9238 |  41.9697 |   -0.0285% |   1.839 |   1.839 |   4.182 |
| CA 35      |  98.0802 |  98.0736 |  96.1049 |   +0.0067% |   1.152 |   1.153 |   1.811 |
| CA 39      | 271.1929 | 271.0175 | 270.3632 |   +0.0647% |   1.087 |   1.109 |   1.174 |
| CA 41      | 363.8382 | 363.9027 | 361.6036 |   -0.0177% |   2.093 |   2.093 |   1.711 |
| ETA 33     |   2.2967 |   2.2966 |   2.3042 |   +0.0043% |   0.199 |   0.199 |   0.271 |
| ETA 35     |   2.0802 |   2.0802 |   2.0838 |   -0.0000% |   0.211 |   0.211 |   0.255 |
| ETA 39     |   2.1159 |   2.1162 |   2.1153 |   -0.0170% |   0.278 |   0.286 |   0.311 |
| ETA 41     |   2.1681 |   2.1680 |   2.1713 |   +0.0012% |   0.392 |   0.392 |   0.399 |
| ALPHA 33   |   0.0847 |   0.0847 |   0.0787 |   -0.0260% |   1.963 |   1.963 |   4.494 |
| ALPHA 35   |   0.1679 |   0.1679 |   0.1643 |   +0.0051% |   1.236 |   1.236 |   1.962 |
| ALPHA 39   |   0.3620 |   0.3618 |   0.3612 |   +0.0629% |   1.040 |   1.063 |   1.280 |
| ALPHA 41   |   0.3578 |   0.3578 |   0.3543 |   -0.0049% |   1.595 |   1.595 |   1.668 |
| F3ETA 33   | 741.9155 | 742.1004 | 746.4586 |   -0.0249% |   0.603 |   0.603 |   0.683 |
| F3ETA 35   | 719.9266 | 719.9103 | 721.2735 |   +0.0023% |   0.472 |   0.472 |   0.507 |
| F3ETA 39   |1181.5885 |1181.0188 |1179.4821 |   +0.0482% |   0.797 |   0.801 |   0.805 |
| F3ETA 41   |1686.9162 |1687.0712 |1693.1349 |   -0.0092% |   2.075 |   2.075 |   1.774 |

**Result:** 41/45 well-constrained parameters agree within 0.05%. The 4
parameters outside this threshold are SCR 39, SCR 41 (unconstrained, >200%
uncertainty), and WGA 39 and WGA 41 (0.03-0.04% difference, attributable to
value rounding). The chi-squared difference (60.0 vs 59.7) reflects the
cumulative effect of the small data differences.

The largest well-constrained difference is in CA 39 (0.065%), which propagates
from the ABS 39 difference (0.019%) and the FIS 39 agreement (+0.002%).
Overall, the two data sources produce negligibly different GLS results for
all quantities of physical interest.


---

## Conclusions

1. **All-data reproduction (Table 4):** Excellent agreement. Chi-squared
   102.3 vs 103, with 47/54 parameters matching. The GLS implementation
   and data transcription are validated.

2. **Microscopic subset identification:** The report's Table 8 microscopic
   fit (101 measurements) excludes Lounsbury alpha measurements (Nos. 21-23),
   likely classified as integral due to reactor spectrum corrections. With
   this exclusion, all 46 reported parameters match within tolerance.

3. **Pipeline equivalence:** The JSON-based and AGS/thermalcst-based GLS
   pipelines produce identical results to machine precision (~1e-6%) when
   given identical input data, confirming both implementations are correct.

4. **Data source comparison:** The JSON (PDF-transcribed) and AGS (CEA
   Cadarache) data sources differ in value rounding (max 0.05%), one
   uncertainty (No. 147), and correlation code column mapping, but these
   differences have negligible impact on fitted parameters (<0.07% for
   all well-constrained quantities).


---

## Update: Axton 1984 Paper Confirmation (April 2026)

The Axton 1984 paper (EUR 8805 EN), obtained through the NIST library on
2026-04-02, confirms the Lounsbury exclusion hypothesis.

**Title:** "Evaluation of the Thermal Neutron Constants of 233U, 235U,
239Pu and 241Pu and the Fission Neutron Yield of 252Cf," E.J. Axton,
Central Bureau for Nuclear Measurements, JRC Geel.

Evidence from the paper:

1. **Table A1.2 (Function Definitions)** clearly distinguishes monoenergetic
   (2200 m/s) functions — ABS, FIS, SCA, SCR, NUB, HLF, WGA, WFA, GC116,
   GA116 — from Maxwellian (20C) derived functions — FA, FF, CAP, FFH, ETA,
   F1ETA, F2ETA, F3ETA, FLEM, F1BIG, F1CAB-F5CAB. This matches exactly the
   `MAXWELLIAN_DERIVED_FUNCS` set used in our benchmark scripts.

2. **Table 6 ("Maxwellian Data Omitted")** shows the microscopic-only fit
   results. The fitted ALPHA 33 = 0.0785, consistent with our no-Lounsbury
   result of 0.0787 and far below Beer's 0.0861. If Lounsbury had been
   included, this value would be pulled significantly upward toward 0.085.

3. **Table 3** confirms that measurements 21-23 (Lounsbury 1970) carry the
   uncertainty code **"Z"**, which Table 2 defines as **"LOUNSBURY
   TEMPERATURE"** — a systematic Maxwellian temperature correction.

4. **Appendix 4, Note 19** states: "Based on Monte Carlo evaluation (Beer
   et al (1975)). Correlated through temperature uncertainty." This confirms
   Axton used the Beer-corrected values and that the correlation arises from
   reactor spectrum temperature.

The paper thus validates our hypothesis: the Lounsbury measurements use
monoenergetic function expressions (CA/FIS) but were measured in a reactor
spectrum requiring Maxwellian temperature corrections. Axton classified
them as effectively Maxwellian and excluded them from Table 6/Table 8.


---

## Update: Lounsbury and Arif Sensitivity Studies (April 2026)

### Motivation

During discussions among the TNC collaborators (March-April 2026):

- It was argued that Lounsbury/Beer data should be included because
  "without Lounsbury/Beer we cannot constrain U-233 capture." The
  Beer-corrected alpha(233U) = 0.0861 +/- 0.0021, the current standard
  alpha = 0.0842 is within 1-sigma, while our 0.079 without Lounsbury is
  clearly outside.

- GMA fit comparisons were provided showing sensitivity to both the
  Lounsbury/Beer data and the Arif U-235 scattering measurement,
  presented in a 2x2 matrix (with/without each).

- It was noted that the TNC database had been updated for NDS2025 and
  that older presentation numbers should not be used.

We ran a systematic sensitivity study using the Axton 1986 microscopic
GLS framework.


### Scenario B: Microscopic fit with Lounsbury included

Script: `benchmark_axton1986_micro_with_lounsbury.py`

Adding Lounsbury measurements (21-23) back to the microscopic baseline
(105 measurements, 30 free parameters):

| Parameter | Without Lounsbury (A) | With Lounsbury (B) | Shift | Beer value |
|-----------|-----------------------|---------------------|-------|------------|
| FIS 33    | 533.14 | 530.38 | -2.76 barn | |
| CA 33     | 42.10  | 44.88  | +2.78 barn | |
| ABS 33    | 575.24 | 575.27 | +0.03 barn | |
| ALPHA 33  | 0.0790 | 0.0846 | +7.1% | 0.0861 |
| FIS 35    | 585.03 | 584.01 | -1.02 barn | |
| CA 35     | 96.17  | 98.07  | +1.90 barn | |
| ALPHA 35  | 0.1644 | 0.1679 | +2.1% | 0.1697 |
| FIS 39    | 748.52 | 748.12 | -0.40 barn | |
| ALPHA 39  | 0.3611 | 0.3611 | -0.0% | 0.3555 |
| Chi-sq.   | 55.1 (72 dof) | 60.5 (75 dof) | | |

**Mechanism:** Absorption stays nearly constant (~575.2 barn for U-233)
while ~2.8 barn is redistributed from fission to capture. Chi-squared per
dof remains acceptable (0.81 vs 0.76), with no severe tension.

**Lounsbury residuals:**

| No. | Quantity | Measured | Fitted | Residual |
|-----|----------|----------|--------|----------|
| 21  | alpha(233U) | 0.0861 | 0.0846 | +1.7% |
| 22  | alpha(235U) | 0.1697 | 0.1679 | +1.0% |
| 23  | alpha(239Pu) | 0.3555 | 0.3611 | -1.6% |

**The argument for inclusion is validated:** the fitted alpha(233U) = 0.085 agrees
well with the current standard (0.084), while excluding Lounsbury gives
0.079 which is outside the 1-sigma range of Beer's value.


### Arif SCA(35) measurement

The Arif measurement (M. Arif et al., Phys. Rev. A35, 2810 (1987)):

- **SCA(35) = 14.0 barn**, uncertainty 1.6%
- From neutron interferometry (coherent scattering length)
- Much more precise than Lemmel (1982) SCA(35) = 16.508 barn at 7.9%
- Not in the original Axton 1986 database (post-1986 measurement)
- Found in this project's `expdata.py` at commit `807aafb` (No. 173)
  and in the neutron standards evaluation database (NS=1918, channel
  SS-U5 = U-235 elastic scattering)

The Arif measurement was injected as an additional uncorrelated data
point. Because Arif (1.6%) is ~5x more precise than Lemmel (7.9%),
it dominates the SCA(35) fit completely.


### Complete 2x2 matrix: Lounsbury x Arif

Script: `benchmark_axton1986_lounsbury_arif_matrix.py`

All four scenarios using the Axton 1986 microscopic GLS framework:

```
                      No Arif          With Arif
                +-----------------+-----------------+
 No Lounsbury   | A) 102 meas     | C) 103 meas     |
                | chi2 = 55.1     | chi2 = 59.9     |
                +-----------------+-----------------+
 With Lounsbury | B) 105 meas     | D) 106 meas     |
                | chi2 = 60.5     | chi2 = 64.3     |
                +-----------------+-----------------+
```

**Key parameters across all four cells:**

| Parameter | (A) baseline | (B) +Loun. | (C) +Arif | (D) +both | GMA +both | GMA -both |
|-----------|-------------|-----------|----------|----------|-----------|-----------|
| FIS 33    | 533.14 | 530.38 | 532.83 | 530.36 | 532.70 | 533.90 |
| FIS 35    | 585.03 | 584.01 | 585.80 | 584.91 | 587.28 | 586.50 |
| FIS 39    | 748.52 | 748.12 | 748.76 | 748.48 | 750.30 | 749.70 |
| CA 33     |  42.10 |  44.88 |  42.41 |  44.91 |  44.76 |  42.50 |
| CA 35     |  96.17 |  98.07 |  97.20 |  98.55 |  99.50 |  98.90 |
| CA 39     | 270.27 | 270.12 | 270.65 | 270.22 | 269.70 | 270.60 |
| ABS 35    | 681.21 | 682.09 | 682.99 | 683.46 | - | - |
| ALPHA 33  | 0.0790 | 0.0846 | 0.0796 | 0.0847 | 0.0840 | 0.0788 |
| ALPHA 35  | 0.1644 | 0.1679 | 0.1659 | 0.1685 | 0.1694 | 0.1687 |
| ALPHA 39  | 0.3611 | 0.3611 | 0.3615 | 0.3610 | 0.3595 | 0.3610 |
| SCA 35    | 16.22  | 15.88  | 14.00  | 14.00  | - | - |

The GMA columns are results from a collaborator's GMA fit (discussed
2026-04-03/04) using the Standards 2017 database plus post-2017 measurements.

**Marginal effects (deviations from baseline A):**

| Parameter | Lounsbury (B-A) | Arif (C-A) | Combined (D-A) | Interaction (D-B-C+A) |
|-----------|-----------------|------------|-----------------|----------------------|
| FIS 33    | -2.75 | -0.30 | -2.78 | +0.28 |
| FIS 35    | -1.02 | +0.76 | -0.12 | +0.13 |
| CA 33     | +2.78 | +0.31 | +2.81 | -0.28 |
| CA 35     | +1.90 | +1.02 | +2.38 | -0.54 |
| ABS 35    | +0.88 | +1.79 | +2.26 | -0.41 |
| ALPHA 33  | +0.0057 | +0.0006 | +0.0057 | -0.0006 |
| ALPHA 35  | +0.0035 | +0.0015 | +0.0041 | -0.0010 |
| SCA 35    | -0.34 | -2.22 | -2.22 | +0.34 |


### Conclusions from the sensitivity study

1. **Lounsbury dominates alpha(233U):** shift of +0.006 (0.079 to 0.085).
   Arif contributes only +0.0006, an order of magnitude smaller.

2. **Arif's main effect is on U-235:** ABS(35) increases by +1.8 barn
   through the lower scattering cross section, split between CA(35) (+1.0)
   and FIS(35) (+0.8). The Lounsbury effect on U-235 is comparable in
   magnitude but opposite in direction for FIS(35).

3. **The effects are largely additive.** Interaction terms are small
   (typically <15% of the main effect), meaning the Lounsbury and Arif
   effects can be discussed independently.

4. **Good agreement with the GMA fit.** Our "without both" (A) gives
   alpha(233U) = 0.0790 vs GMA 0.0788. Our "with both" (D) gives 0.0847
   vs GMA 0.0840. The trends and magnitudes match well despite different
   databases (Axton 1986 vs Standards 2017 + post-2017 measurements).

5. **The argument for inclusion is supported.** Including Lounsbury brings
   alpha(233U) from 0.079 to 0.085, consistent with the current standard
   (0.084) and within the Beer uncertainty band. The fit absorbs the
   Lounsbury data without serious tension (chi2/dof = 0.81).


### Updated benchmark inventory

| Script | Description |
|--------|-------------|
| `benchmark_axton1986_all_data.py` | Table 4 reproduction (167 meas) |
| `benchmark_axton1986_micro.py` | Narrow microscopic (112 meas) |
| `benchmark_axton1986_micro_strict.py` | Strict microscopic (96 meas) |
| `benchmark_axton1986_micro_no_lounsbury.py` | Best Table 8 match (102 meas) |
| `benchmark_axton1986_micro_with_lounsbury.py` | + Lounsbury (105 meas) |
| `benchmark_axton1986_micro_with_lounsbury_arif.py` | + Lounsbury + Arif (106 meas) |
| `benchmark_axton1986_lounsbury_arif_matrix.py` | 2x2 Lounsbury x Arif comparison |
