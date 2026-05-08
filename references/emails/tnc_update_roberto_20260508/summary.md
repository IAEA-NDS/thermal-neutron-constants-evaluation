# Email round: tnc_update_roberto_20260508

Three follow-up emails after the round-1 report (commit `3af22ed`) was
distributed to the working group. Two corrections to the round-1 inputs and
one new sensitivity check.

Participants: Roberto Capote (rcapotenoy@yahoo.com), Vladimir Pronyaev
(vgpronyaev@yandex.ru), Allan D. Carlson, Gilles Noguère, Ignacio Durán
Escribano, Georg Schnabel.

Source files (not committed):

- `RE_ NEW TNC FIT steps_01.msg` — Roberto, 2026-05-05 23:21
- `RE_ NEW TNC FIT steps_02.msg` — Roberto, 2026-05-06 01:43
- `Re_ NEW TNC FIT steps_03.msg` — Roberto, 2026-05-06 20:17

---

## 2026-05-05 21:47 — Pronyaev: review of the round-1 report

Vladimir spots two issues with the step-4 SCA list:

1. **KAISER (Physica B 136, 134 (1986)) and ARIF (PR A35, 2810 (1987))**
   may be the same experiment published in two different journals — both
   first authors are from the same institute and installation. Needs
   journal verification.

2. **GREEN (JNM 51, 281 (1974), U-233)** is most probably an SCR
   (not SCA) measurement: per the EXFOR record, the experiment used the
   classical (non-crystal-spectrometer) method.

Vladimir attaches an SCA estimation for Green at the thermal point and finds
no other problems.

## 2026-05-05 23:21 — Roberto: SCA-list corrections

Roberto agrees with Vladimir's review and adds further reclassifications:

> For my compilation of elastic XS experiments I assumed that ALL measured
> values were SCA. But in the list it was indicated otherwise (Atoyi U235
> is SCR, Willis U235 is SCR, Green U233 1974 is SCR). All Pu experiments
> were SCA.
>
> 1. I agree to reclassify those listed above as SCR.
> 2. Kaiser should be removed (it is the same experiment as Arif).

> @Georg please correct point 4) in your document with changes 1) and 2).
> I expect the change to be very small, but it is the right thing to do.

The cleaned-up SCA list, with SCR/SCA tagging now intended literally:

> ```
> SEARS:
> U-235                                       Scoh        Si        Sel
>   ACR20 (1966) 587  ATOYI                 12.1(5)    0.2(2)   12.3(5)   SCR
>   PRS/A274 (1963) 122 WILLIS              12.1(5)    0.2(2)   12.8(1.0) SCR EXFOR
>   PR/A35 (1987) 2810 ARIF+                13.80(11)  0.2(2)   14.0(2)
> DIVADEENAM:
>   CEULEMANS 1970                                                14.6(5)  EXFOR 14.3(5) rel V-51 (1.02)
>   BAYVAS 1980 (TUK)                                             14.3(1.5)
>   FOOTE   1958                                                  15(1)    SCR
>
> SEARS:
> U-233                                                           Sel
> DIVADEENAM:
>   OLEKSA 1960                                                   13.4(5)  SCA (at 0.1 eV) from EXFOR
>   GREEN 1974                                                    12.6(7)  SCR
>   VERTEBNYI 1974                                                13.2(6)  SCA
> EXFOR:
>   MOORE 1962                                                    14.5(7)  SCA (at 1.5 eV)   [skipped per round 1]
>
> SEARS:
> Pu-239                                      Scoh        Si        Sel
>   JNM34 (1970) 281 GREEN                   7.5(2)    0.2(6)    7.7(6)    SCA
>   ACR20 (1966) 587  ATOYI                  7.1(3)    0.2(6)    7.3(7)    SCA
>   ACR15 (1962) 351 ROOF+                   7.1(3)    0.2(6)    7.3(7)    SCA
> ```

> **********************************************************************
>
> Analysis of Kaiser vs Arif (attached)
>
> It is the same experiment, being Arif the latest publication. Kaiser gave
> preliminary values at only two energies. We can delete Kaiser 1986 from
> the database.

### Net changes vs. round 1

| Action     | Entry                              | Effect                                              |
|------------|------------------------------------|-----------------------------------------------------|
| Remove     | KAISER (Physica B 136 (1986), U-235) | duplicate of ARIF; was step-4 No.173                |
| Reclassify | ATOYI (ACR20 (1966), U-235)         | SCA → SCR; was step-4 No.171, value 12.3 (5)        |
| Reclassify | WILLIS (PRS A274 (1963), U-235)     | SCA → SCR; was step-4 No.172, value 12.8 (1.0)      |
| Reclassify | GREEN (JNM 51 (1974), U-233)        | SCA → SCR; was step-4 No.179, value 12.6 (7)        |
| Reclassify | FOOTE (1958, U-235)                 | SCA → SCR; tagged "SCR" in cleaned table though not in Roberto's numbered action items — decision inferred (logged in appendix) |

Roberto's expectation: "I expect the change to be very small, but it is the
right thing to do."

## 2026-05-06 01:43 — Roberto: new Westcott factors

Replace the current Westcott factors with values derived from the B81
evaluation (R-matrix fit best estimate). Uncertainties = difference between
B81 and JEFF-4.0.

> | nuclide | Gf       | Gabs      |
> |---------|----------|-----------|
> | U-233   | 0.997(1) | 1.000(1)  |
> | U-235   | 0.981(5) | 0.982(3)  |
> | Pu-239  | 1.052(14)| 1.078(10) |
> | Pu-241  | 1.052(7) | 1.046(7)  |

> I derived Gabs = (Gcap·Scap + Gfis·Sfis)/(Scap+Sfis)
>
> Note that with the new Westcott factor for U-235 I got the K1 equal to 722.4.

> @Gilles, what is the recommended K1 value for Pu-239? I remember you
> presented it in one of our previous meetings?

## 2026-05-06 20:17 — Roberto: Step 5 favored, request 2× SCA−SCR sensitivity

> One additional request. I marginally favor solution at Step 5 over the one
> at Step 6. It seems that we have some systematic effect of reduced SCR vs
> SCA as we physically expect. That effect leads to slightly larger fission
> XS in better agreement with what we have in the libraries (the R-matrix
> fit). Of course, differences are minor and within uncertainties.
>
> I would like you to do another check. Pls start from Step 5 results and
> increase the SCA-SCR by a factor of 2 (both mean values and
> uncertainties). I like to see how stable our solution is relative to the
> SCA-SCR assumed values.

---

## Action items derived

1. **Patch the step-4 SCA list** to drop KAISER and reclassify
   ATOYI-235 / WILLIS-235 / FOOTE-235 / GREEN-233 from SCA to SCR. Cascade
   the renumbering through steps 5 and 6.
2. **Update the Westcott factors** (WGA, WGF) and their uncertainties in the
   reference data for all four nuclides. Changes the derived
   FA / FF / F1ETA / F2ETA / F3ETA outputs.
3. **Add a Step 5b sensitivity scenario**: identical to step 5 except
   SCA−SCR mean values and uncertainties are scaled by 2.
4. **Re-run the chain summary**, regenerate the report, and add an appendix
   section B documenting these round-2 corrections.
