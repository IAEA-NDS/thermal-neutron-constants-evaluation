# Email round: tnc_update_roberto_20260505

Working group thread that converged on the 7-step Capote fit chain and the
revised Axton transmission/SCA database used in branch
`tnc_update_capote_20260504`. The folder name is the date the .msg files were
archived; the underlying messages span 2026-04-09 → 2026-05-04.

Participants: Roberto Capote (rcapotenoy@yahoo.com), Vladimir Pronyaev
(vgpronyaev@yandex.ru), Allan D. Carlson (carlson.std@gmail.com), Gilles
Noguère (gilles.noguere@cea.fr), Ignacio Durán Escribano
(ignacio.duran@usc.es), Georg Schnabel (g.schnabel@iaea.org).

Source files (not committed):

- `NEW TNC FIT steps.msg` — Roberto, 2026-05-04 04:13
- `Revision_additions to Axton Total cross sections experimental data.msg`
  — Roberto, 2026-05-04 03:35, with a long inline thread going back to
  2026-04-09 covering the SCA strategy.

---

## 2026-04-09 — Pronyaev: bound-coherent length data not used by Axton

Vladimir notes that bound-coherent neutron scattering lengths from neutron
interferometry, available since the 1980s, were not used by Axton. These
measurements are independent of sample form (metal, powder, liquid), have
much smaller uncertainties than classical scattering measurements, and could
fix the SCA values directly via Sig(SCA) = π·(B_coh² + B_incoh²).

## 2026-04-10 — Pronyaev: proposal to replace Lemmel SCA values

Pronyaev proposes:

1. Define SCA in the TNC database as the *true* nuclear elastic scattering
   cross section, free from material structure effects.
2. For U-235 and Pu-239, neutron interferometry results yield the
   bound-coherent length B_coh with very small uncertainties; combined with
   B_incoh from NIST, this gives consistent SCA values.
3. Replace all Lemmel SCA values with SCA derived from the WAI (Wien Atomik
   Institut) B_coh compilation + NIST B_incoh + EXFOR X4=21688 (Türkay 1980).
4. Use Lemmel's SCA−SCR difference, or modern model calculations (per
   Noguère's plot for natU): about 1 ± 0.2 b, vs. Lemmel's 1.8 ± 0.7 b. The
   value depends on sample preparation (rolling, annealing).
5. AI assignment of ABS+SCA vs. ABS+SCR for transmission measurements is
   acceptable.
6. No B_coh data exist for U-233 and Pu-241; NIST (1992) recommendation
   exists for U-233. Pu-241 needs separate treatment.

## 2026-04-12 — Pronyaev: U-233 analysis

Pronyaev attaches an analysis for U-233 (saved separately in the project's
external-data references). Conclusion: SCA for Green's value will be ~1 b
higher than Green's published result corrected to the new V-monitor.

Roberto suggests classifying Green (corrected) as SCR and using SCA−SCR =
1 ± 0.2 b.

## 2026-04-12 — Roberto: SCA/SCR classification framework

Roberto consolidates the agreed approach:

1. All transmission experiments must be classified as SCA (true elastic for
   non-crystalline targets) or SCR (rolled metallic samples). Some
   experiments using rolled samples did proper corrections and remain SCA
   (e.g. Arif).
2. Use Noguère's prior in place of Lemmel's: SCR = N·SCA, with N = 1.015,
   ΔN = 1 %.
3. Add all TUW + EXFOR data with SCA/SCR classification, corrected to current
   standards.

## 2026-04-17 — Pronyaev: R_pot analysis for Pu-241

> ENDF/B-VIII.1, JEFF-4.0 and JENDL-5 have R(pot) = 9.54 fm.
> 4·π·R(pot)² = 11.44 b and SCA(0.0253 eV) is 12.0, 11.76 and 11.26 b
> correspondingly.
>
> The TNC value is 11.9 ± 2.6 b.

Vladimir notes the spread between evaluations (0.8 b) is far smaller than
the TNC uncertainty (2.6 b), so the Pu-241 SCA can be fixed with much
reduced uncertainty from this analysis.

## 2026-04-21 — Roberto: SCA "cut posterior" strategy

Roberto consolidates the strategy as four points:

1. **Fix SCA mean values to Sears (Neutron News 3 (1992) 26-37):**

   > | nuclide | Sears      | Mughabghab 2018 |
   > |---------|------------|-----------------|
   > | U-233   | 12.9 (3)   | 12.4 (3)        |
   > | U-235   | 14.0 (2)   | 14.0 (2)        |
   > | Pu-239  |  7.7 (6)   |  7.94 (36)      |

2. **Pu-241 from Pronyaev's potential-scattering analysis:**

   > Pu-241: 12.0 (0.8)

   In disagreement with Mughabghab (8.9), but no direct data exist.

3. With SCA fixed from (1)+(2), the TNC fit is run as a "cut posterior"
   evaluation. Per-isotope SCA−SCR > 0 is the only scattering parameter,
   replacing Noguère's N-form. Prior on SCA−SCR = 1 ± 0.2 b.

4. Roberto is in parallel revising the total-cross-section database,
   separating SCR (metallic rolled) and SCA (otherwise). Once finished,
   ARIF and other direct SCA inputs can be removed since SCA is fixed.

Pronyaev (same day) replies that there is also benefit to fitting *all*
available scattering data with SCA−SCR estimation rather than fixing SCA
to Sears, since closeness to Sears would then validate the TNC SCA
evaluation. Roberto disagrees, citing that Sears is well-documented and
accepted in the atomic and solid-state community.

## 2026-04-22 — Sears compilation values

Roberto sends recommended Sears scattering values for the working group's
attention:

> ```
> 94-Pu-239 24400 Y 1/2
>   7.7 ± 0.1  BD  70Gre1
>   7.5 ± 0.3  BD  66Ato1
>   7.5 ± 0.3  BD  62Roo1
>
> 92-U-235 0.72 7/2
>  10.5  ± 0.03 IN  86Kai1
>  10.74 ± 0.04 IN  87Ari1
>   9.8  ± 0.6  BD  66Ato1
>   9.8  ± 0.6  BD  63Wil1
> ```

## 2026-04-30 — Roberto: SCA experimental data compilation

Initial compilation of SCA data from Sears, Divadeenam and EXFOR for
U-235, U-233 and Pu-239. SCR/SCA tags appear inconsistent in this email
(some pasted from source tables); resolved later in 20260508 round.

> ```
> SEARS:
> U-235                                   Scoh        Si        Sel
>   ACR20 (1966) 587  ATOYI            12.1(5)    0.2(2)   12.3(5)   SCR
>   PRS/A274 (1963) 122 WILLIS         12.1(5)    0.2(2)   12.3(5)   replace with EXFOR
>   PHY/136B (1986) 134 KAISER+        13.80(11)  0.2(2)   14.0(2)
>   PR/A35 (1987) 2810 ARIF+           13.80(11)  0.2(2)   14.0(2)
>
> DIVADEENAM:
>   CEULEMANS 1970                                          14.4(6)  replace with EXFOR
>   WILLIS 1963                                             12.8(1.0) = EXFOR  SCR
>   BAYVAS 1980 (TUK)                                       14.3(1.5)
>   FOOTE   1958                                            15(1)
>
> EXFOR:
>   CEULEMANS 1970                                          14.6(5)  EXFOR 14.3(5) rel V-51 (1.02)
>   FOOTE 1958                                              15.0(1.0)
>
> SEARS:
> U-233                                                     Sel
> DIVADEENAM:
>   OLEKSA 1960                                             12.5(5)  SCR replace by EXFOR below
>   GREEN 1974                                              12.6(7)  SCR
>   VERTEBNYI   1974                                        13.2(6)  SCA
> EXFOR:
>   OLEKSA 1960                                             13.4(5)  SCA (at 0.1 eV)
>   MOORE 1962                                              14.5(7)  SCA (at 1.5 eV)
>
> SEARS:
> Pu-239                                  Scoh        Si        Sel
>   JNM34 (1970) 281 GREEN              7.5(2)    0.2(6)    7.7(6)
>   ACR20 (1966) 587  ATOYI             7.1(3)    0.2(6)    7.3(7)
>   ACR15 (1962) 351 ROOF+              7.1(3)    0.2(6)    7.3(7)
> ```

## 2026-05-04 03:35 — Roberto: revision/additions to Axton total XS database

> **U-235**
> 1. Change Safford 1959 ABS+SCR → ABS+SCA
>    Add:
> 2. Spencer 87 ABS+SCR        690 (5)
> 3. Antonov 86 ABS+SCA        695 (15)
> 4. Gerasimov 62 ABS+SCR      685 (9)
> 5. Leonard 54                702 (14)
>
> **U-233**
> 1. Change Safford 1959 ABS+SCR → ABS+SCA
> 2. Change Nikitin 55 ABS+SCR → ABS+SCA
>    Add:
> 3. Pschenichnyi 78 ABS+SCR   587.9 (3)
> 4. Harvey 1 79  ABS+SCR      585.0 (1.4)
> 5. Harvey 2 79  ABS+SCR      592 (6)
>
> **Pu-241** (add):
> 1. Schwartz 58 ABS+SCR       1410 (80)
>
> **Pu-239** (add):
> 1. Spencer 87 ABS+SCR        1025 (6)
> 2. Anderson 45 ABS+SCR       1045 (25)
> 3. Palevsky 55 ABS+SCR       1025 (10)
> 4. Leonard 56 ABS+SCR        1055 (13)

(Note: Spencer 87 entries for U-235 (690 ± 5) and Pu-239 (1025 ± 6)
matched values already present in our database with different
uncertainties; we kept the database values and flagged in appendix A1.
Pschenichnyi 78 uncertainty was treated as 3.0 b, not 0.3 b — see A4.)

## 2026-05-04 04:13 — Roberto: the 7-step fit chain

> | # | Step                                                                  |
> |---|------------------------------------------------------------------------|
> | 0 | Axton Table 8 (no Lounsbury)                                           |
> | 1 | With Lounsbury alpha (21-23)                                           |
> | 2 | + Adamchuk alpha                                                       |
> | 3 | + Wallner U5 capture                                                   |
> | 4 | + ADD LISTED SCA measurements + Pu241 estimate, delete Lemmel SCA      |
> |   |   values (leave SCA-SCR)                                               |
> | 5 | + ADD LISTED ABS+SCR, ABS+SCA MEASUREMENTS                             |
> | 6 | CONVERT ALL SCR in ABS+SCR to ABS+SCA, delete Lemmel SCA-SCR           |
> |   |   (reducing 4 fitting variables SCR)                                   |

> We aim to compare steps 5 and 6. Hopefully one of them will be the final
> fit. I leave the Westcott factors, should replace them by the most updated
> ones from current libraries (will do after the fit). The reason is not to
> fit Maxwellian data, but to calculate the derived Maxwellian function
> K1 = F3ETA. For that you may need to print FA, FF, ETA, F1ETA, F2ETA, F3ETA.

---

## What was implemented (round 1)

- Branch `tnc_update_capote_20260504` was created from the
  microscopic-Axton baseline, and seven step JSONs / benchmarks were added
  (commits `a3fa36f` … `8dd3b31`).
- Step 4 SCA additions used the literal compilation values, treating all
  rows as SCA per user instruction (paste-artifact "SCR" tags ignored).
- Step 5 transmission additions per the 2026-05-04 03:35 list.
- Step 6 converted all ABS+SCR → ABS+SCA and removed SCR variables for the
  four nuclides.
- Pu-241 SCA = 12.0 (0.8) added per Pronyaev's R_pot analysis.
- Comprehensive report committed in `3af22ed` covering the chain.
