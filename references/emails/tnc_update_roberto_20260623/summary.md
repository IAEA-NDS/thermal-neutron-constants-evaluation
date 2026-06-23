# Email round: tnc_update_roberto_20260623

Round-3 revision of the TNC input database — a comprehensive fission and
total-cross-section (n,tot) data review driven by Roberto Capote and Ignacio
("Nacho") Durán. Roberto's English summaries sit on top of Nacho's underlying
EXFOR / integral re-fits (Spanish working notes preserved in the source
`.msg` files). The folder name is the date the `.msg` files were archived;
the underlying messages span 2026-05-15 → 2026-06-02.

Implemented on branch `tnc_update_capote_20260623` (forked from
`tnc_update_capote_20260504`), building on the round-2 Step-5 favoured fit.

Participants: Roberto Capote (rcapotenoy@yahoo.com), Ignacio Durán Escribano
(ignacio.duran@usc.es), Vladimir Pronyaev (vgpronyaev@yandex.ru), Allan D.
Carlson (carlson.std@gmail.com), Gilles Noguère (gilles.noguere@cea.fr),
Georg Schnabel (g.schnabel@iaea.org).

Source files (not committed):

- `Data corrections_additions for the TNC evaluation (FISSION).msg`
  — Roberto, 2026-06-01 01:28
- `Revision Pu41(n_tot).msg` — Roberto, 2026-06-02 14:51
- `Revised Pu9(n_tot).msg` — Roberto, 2026-06-02 15:45
- `Revised U5(n_tot).msg` — Roberto, 2026-06-02 16:25
- `Revised (n_tot) U3.msg` — Roberto, 2026-06-02 17:17
- `new TNC fit.msg` — Roberto, 2026-06-02 17:24

A datapoint-level before→after manifest of every change is kept alongside in
[`data_change_manifest.md`](data_change_manifest.md).

Governing instructions (apply to all five data emails):

  * **Keep the current SCA/SCR classification** of every existing transmission
    entry (do not re-tag SCA vs SCR; only values/uncertainties change).
  * Use Nacho's **locally re-fitted ("adjusted by Nacho") values**, which are
    traceable to EXFOR and were derived without renormalisation.
  * After all inputs are in, produce a **full input-data list with both
    absolute and relative uncertainties** for cross-checking vs. Axton and
    Nacho's changes (the "new TNC fit" email).

---

## 2026-06-01 — Fission data corrections / additions

Roberto and Nacho went over all available fission data, found many values
missing from the TNC input database, plus some outliers and repeated data.

> **Pu-239**
> Tovesson 10   (FIS 39)/FIS 35  1.286 (2%)   X4: 14271  (calc. using B70 value 585 and measured 752.4)
> Gwin 84        FIS 39           750.2 (1.7%)  X4: 12905  (I1=Nacho)
> Zhuravlev 76  (FIS 39)/FIS 35  1.274 (2%)    X4: 40476
> Deruytter 70   FIS 39           749.3 (0.5%)  X4: 20143  (adjusted by Nacho 741.9->749.3)
> Deruytter 72   FIS 39           748.8 (1.0%)  X4: 20132  (renormalized to Deruytter 70 adjusted by Nacho)
> Stavisskii 65  FIS 39           748 (1%)      X4: 41635
> Raffle 59      OUTLIER (data # 104) DISCARD
> BERCEANU 77   (data # 101) CONTAINS BORCEA 70   X4: 30089
>
> **Pu-241**
> Egelstaff 55  (FIS 41)/FIS 39  1.33 (3%)     X4: 23526
> Jaffey 55     (FIS 41)/FIS 39  1.362 (1.7%)  X4: 12521
> James 65       FIS 41           1014.4 (2%)   X4: 20940  adjusted by Nacho
> Watanabe 64    FIS 41           1019.3 (2.4%) X4: 12529  adjusted by Nacho
> Weston 78      FIS 41           1018.8 (2.7%) X4: 10768  adjusted by Nacho
> Tovesson 10   (FIS 41)/FIS 35  1.73 (2%)     X4: 14271  Used 585 from B70 for U5 XS
>
> **U-233**
> Berceanu 82   (FIS 33)/FIS 35  0.892 (2%)    X4: 30750
> Zhuravlev 76  (FIS 33)/FIS 35  0.908 (1.5%)  X4: 40476
> Wenhai 75     (FIS 33)/FIS 35  0.916 (1%)    X4: 32624
> Deruytter 74   FIS 33           533.2         X4: 20411
>
> **U-235**
> Deruytter 71   586.6 (0.8%)   20131  (updated with Nacho's fit)
> Gwin 84        586.1 (0.3%)   12905  (updated with Nacho's fit and renormalized to Nacho's I3)
> Deruytter 61   587 (1%)       20189
> Borcea 73      586.3 (1.3%)   30140  Mihailescu relative to Borcea
> Amaducci 22    586.2 (0.5%)   23453
> Berceanu       589.3 (0.8%)   30399  (replace by Borcea 73)
> Deruytter 70   587.6 (0.4%)   20143
> Maslin 65      572 (6)        21387  OUTLIER (delete)
> Ye 64          582 (1.5%)     32827  (relative to B-nat 764 b)
> Fraysse 65     584.3 (1.5%)   20071  (relative to B-nat 761 b, updated using 764, factor 764/761)
> Safford 59     580 (1.2%)     12439
> Raffle 59      582 (1.5%)     21471  (delete, outlier)
>
> Bowman, Melkonian, Schrack, Wagemans 79,88, Mihailescu are SHAPE
> measurements (relative to the thermal point)
>
> Additionally:
> Update Pattenden 56   Pu9(n,tot)  1039(5%)        Nacho fit
> Update Gerasimov 62  (data #186)  U5(n,tot)  692 (1.4%)   Nacho fit
> Delete Gerasimov 62 ABS (data # 71 Axton) same exp as total, derived quantity
> Delete Maslin 65 FIS 35 (data # 106)   OUTLIER
> Delete Saplakoglu 59 FIS 35 (data # 107) OUTLIER
> Delete Leonard 56 ABS 39 + SCA 39 (OUTLIER)
> Update U-234(n,g) to the B80 value: 100.9 barn

## 2026-06-02 — Revised U-233 (n,tot)

> Keep the current SCA/SCR classification:
>
>                    Integrals fit   Georg        EXFOR
> Block 60         590.2  -  587.3(5)  --> 587@12024      keep as it is.  (data # 84)
> Harvey 79        590.4  -  592.0     --> 588(9)@10955-005 (data # 189)  adjusted by Nacho
> Delete Harvey 79 data # 190
> Moore 60         591.1             --> 587(18)@12341    adjusted by Nacho, NEW DATA
> Pattenden 56            -  596.3   --> 592(?)@21065     Discard, too much scatter (data # 79)
> Nikitin 55       589.7  -  577.3   --> 580(20)@41219-003  (data # 78) adjusted by Nacho
> Pschenichnyj 78  591.8  -  587.9   --> 588(9)@40426       (data # 188) adjusted by Nacho
> Safford 60       keep as it is.  (data # 82)
> Safford 60       keep as it is.  (data # 83)
> Muether  54      keep as it is.  (data # 77)
> Simpson 60       keep as it is.  (data # 81)

Nacho's underlying note ("assume they are ABS33+SCA33") additionally listed a
Bollinger 52 = 592.6(20) value as a candidate addition; Roberto's
consolidated list carries only Moore as NEW DATA and omits Bollinger.

## 2026-06-02 — Revised Pu-239 (n,tot)

> keep the current SCA/SCR classification (unless explicitly indicated)
>
>                  Integrals fit  Georg        EXFOR
> Bollinger 58  1028.2  - 1022(14) - 1029(10)  adjusted by Nacho (data #74)
> Nikitin 55    1029.4  - 1040(30) - 1030(20)  adjusted by Nacho (data #73)
> Palevsky 55   1028.0  - 1025(10) - 1030(10)  adjusted by Nacho (data #193)
> Safford 61    1026.5  - 1018(7)  - 1030(15)  adjusted by Nacho (data #76)
> Pattenden 56  1029.5  - 1012(30) - 1039(25)  adjusted by Nacho (data #75)
> Havens        1029.8  -          --           do not consider, outlier (1080)
> Zimmerman 55          - 1022(13)              (data #72, not in EXFOR)  ok
> Anderson 45           - 1045(24)              (data #192)               ok
> Leonard 56            - 1055(12) - 1042(25)  adjusted by Nacho (data #194)
> Spencer 85            - 1025.8(10)            (data #167)

Nacho's note: the five good values cluster around 1030 b once correctly
fitted (Safford was mis-estimated in the database); Leonard is an outlier;
Havens is a mis-calibrated experiment; Anderson and Spencer are
monoenergetic. The Pattenden uncertainty here (25 b) differs from the
fission email's "(5%)" — see manifest / decision register.

## 2026-06-02 — Revised U-235 (n,tot)

> keep the current SCA/SCR classification. Those to be used are highlighted (green):
>
>                  Integrals fit  Georg          EXFOR
> Block 60      698.3  - 693.3(7)            ok as it is, data # 69
> Gerasimov 62  699.6  - 685(1.3%) --> 692(8)@40651   adjusted by Nacho. Data # 186 (check #71 deleted)
> Leonard 54    700.9  - 702(2%)   --> 701(14)@12430  adjusted by Nacho. Data # 187
> Lynn 55       701.6  -           --> 704(14)@21027  adjusted by Nacho. NEW set
> Melkonian 53  701.3  - 694.3(1.5%) --> 699(10)@12426  adjusted by Nacho, data # 63
> Nikitin 55    700.3  - 710(21)   --> 707(10)@41219  adjusted by Nacho, data # 65
> Palevsky 54   700.5  - 700.3(10) --> 700(5)@12410   adjusted by Nacho, data # 64
> Simpson 66    ok as it is, data # 66
> Safford 59    ok as it is, data # 67
> Safford 59    ok as it is, data # 68
> Saplakoglu 61 ok as it is, data # 70
> Antonov       ok as it is, data # 185
> Spencer       ok as it is, data # 166

Nacho's note recommended **against** introducing Lynn ("old, large
discrepancy between local and integral fit, not worth it"); Roberto's
summary nonetheless lists it as a "NEW set" to use — flagged in the decision
register.

## 2026-06-02 — Revised Pu-241 (n,tot)

> Use values highlighted (green), keep the SCA/SCR classification:
>
>                Integrals fit  Georg        EXFOR
> Simpson 61  1393.3  - 1386(50) --> 1385(15)@12537       (data #85)
> Smith 84    1392.5  - 1396(20) --> 1395(10)@13019-003   (data #87)
> Add:
> Young 68    1393.1  - 1390(10) --> 1390(10)@10406-004   (NEW, ABS+SCR, metallic foil)
> Discard
> Craig 64            - 1381(40) --> discard, too large scattering (delete data #86)
> Schwartz 58         - 1410(80) --> discard, too large scattering (delete data #191)

## 2026-06-02 — "new TNC fit" (wrap-up)

> Once you introduce all inputs, please provide a new list of input data with
> all values (please add both absolute and relative uncertainty) for double
> checking vs Axton and Nacho's changes.

Roberto notes this is a comprehensive review/revision of all data; Nacho
cross-checked all fits, giving a traceable way of verifying his numbers
(without renormalization).

---

## Action items derived

1. **Apply value/uncertainty updates** to the existing transmission and
   fission entries listed above (keep SCA/SCR classification). See manifest §A.
2. **Delete** the outlier / duplicate entries: #71, #79, #86, #103, #104,
   #106, #107, #190, #191, #194. See manifest §C.
3. **Add** the new fission datapoints (all four fissile) and the new
   transmission points (Moore-U3, Lynn-U5, Young-Pu41). See manifest §D.
4. **Update U-234(n,γ)** central value to the B80 value 100.9 b (#5).
5. **Resolve and document** the open interpretation points (manifest §F and
   the decision register) for review by Roberto and the working group.
6. **Re-run** the favoured Step-5 fit (round-3 base), regenerate the chain
   summary and report (new round-3 appendix), and produce the requested
   full input-data list with absolute + relative uncertainties.
