# What makes a case fall in each severity tier

185 cases with a human severity label: non-serious 155, serious 6, substantial+ 24.
Features come from the LLM-extracted case JSON; the tier comes from the human annotation.

## Key takeaways

- 1 comparison(s) are free of court confounding (smallest group 6 cases).
- serious vs non-serious [HKDC]: no feature is significant after correcting for multiple testing.
- substantial+ vs serious [ALL]: no feature is significant after correcting for multiple testing. ⚠ court mix distance 1.00: possibly a court effect.
- non-serious vs all other tiers [ALL]: 43 significant feature(s); strongest is “Hospital stay, days” (less in non-serious, effect -0.94). ⚠ court mix distance 0.80: possibly a court effect.
- serious vs all other tiers [ALL]: no feature is significant after correcting for multiple testing.
- substantial+ vs all other tiers [ALL]: 43 significant feature(s); strongest is “Hospital stay, days” (more in substantial+, effect +0.91). ⚠ court mix distance 1.00: possibly a court effect.
- Decision tree [HKDC]: macro-F1 0.48 on unseen cases, which does not clearly beat always predicting the biggest tier (0.49).
- Decision tree [ALL]: macro-F1 0.47 on unseen cases, which beats always predicting the biggest tier (0.30).

## 1. Read this first: data and caveats

Cases per tier and court:

| tier | HKCFI | HKDC |
|---|---|---|
| non-serious | 0 | 155 |
| serious | 0 | 6 |
| substantial+ | 24 | 0 |

- ⚠ Tiers 'substantial+' and 'non-serious' come from different courts (court mix distance 1.00), so pooled differences between them may be court or era effects, not injury severity.
- ⚠ Tiers 'substantial+' and 'serious' come from different courts (court mix distance 1.00), so pooled differences between them may be court or era effects, not injury severity.
- ⚠ Tier 'serious' has only 6 cases: patterns for it are leads to check, not findings.

Features that almost perfectly tell the two biggest courts apart (AUC far from 0.5). Because tier and court overlap in this data, these cannot yet be credited to severity rather than to court; only within-court comparisons can separate the two:

- Hospital stay, days: AUC 0.04
- Number of surgical treatments received: AUC 0.10
- Number of operations received: AUC 0.12
- Number of different loss categories: AUC 0.14
- Number of losses claimed: AUC 0.15
- Highest VMIS surgery tier received (0 none, 1 Minor ... 4 Complex): AUC 0.17
- Injuries of type permanent injury: AUC 0.17
- Manifestations of type neurological deficit: AUC 0.18

## 2. Comparisons within a single court (the trustworthy ones)

Court cannot explain these differences, because every case in the comparison is from the same court.

#### serious vs non-serious [HKDC]  (n = 6 vs 155, court mix distance 0.00)
**No feature passes the false-discovery threshold (q < 0.1).** Showing the 8 smallest p-values as exploratory hints only.

| Feature | In serious | In non-serious | Effect | p | q | Notes |
|---|---|---|---|---|---|---|
| Hospital stay, days | median 31 (stated in 50%) | median 1 (stated in 41%) | +0.94 | 0.005 | 0.331 |  |
| ICD-11 NA82: Fracture of rib, sternum or thoracic spine | 33% (2/6) | 1% (2/155) | +0.32 | 0.007 | 0.331 |  |
| Number of Minor VMIS surgeries received | median 0.5, mean 0.67 | median 0, mean 0.17 | +0.37 | 0.018 | 0.440 |  |
| ICD-11 NB52: Fracture of lumbar spine or pelvis | 33% (2/6) | 3% (5/155) | +0.30 | 0.022 | 0.440 |  |
| ICD-11 NA02: Fracture of skull or facial bones | 33% (2/6) | 3% (5/155) | +0.30 | 0.022 | 0.440 |  |
| ICD-11 NA22: Fracture of neck | 17% (1/6) | 0% (0/155) | +0.17 | 0.037 | 0.609 |  |
| Treatment includes icu | 17% (1/6) | 1% (1/155) | +0.16 | 0.073 | 0.982 |  |
| Number of symptoms / functional deficits | median 4, mean 4.8 | median 7, mean 7.2 | -0.42 | 0.080 | 0.982 |  |

## 3. All cases pooled

Read the court mix distance of each comparison: close to 1 means the two groups come from different courts, and differences may reflect the court, not the injury.

#### substantial+ vs serious [ALL]  (n = 24 vs 6, court mix distance 1.00)
**No feature passes the false-discovery threshold (q < 0.1).** Showing the 8 smallest p-values as exploratory hints only.

| Feature | In substantial+ | In serious | Effect | p | q | Notes |
|---|---|---|---|---|---|---|
| Manifestations of type neurological deficit | median 1.5, mean 1.9 | median 0, mean 0 | +0.83 | 0.001 | 0.121 | court proxy (AUC 0.18) |
| Number of symptoms / functional deficits | median 10, mean 10 | median 4, mean 4.8 | +0.73 | 0.007 | 0.172 |  |
| Highest VMIS surgery tier received (0 none, 1 Minor ... 4 Complex) | median 3, mean 2.7 | median 0.5, mean 0.5 | +0.67 | 0.010 | 0.172 | court proxy (AUC 0.17) |
| Injuries of type permanent injury | median 2, mean 2.7 | median 0, mean 0.33 | +0.67 | 0.011 | 0.172 | court proxy (AUC 0.17) |
| Number of different loss categories | median 6, mean 5 | median 2, mean 2.3 | +0.68 | 0.011 | 0.172 | court proxy (AUC 0.14) |
| Number of surgical treatments received | median 3, mean 3.5 | median 0.5, mean 1 | +0.65 | 0.014 | 0.172 | court proxy (AUC 0.10) |
| Number of losses claimed | median 6, mean 5 | median 2, mean 2.5 | +0.65 | 0.015 | 0.172 | court proxy (AUC 0.15) |
| Manifestations of type functional deficit | median 3, mean 3.3 | median 1.5, mean 1.5 | +0.64 | 0.016 | 0.172 |  |

#### non-serious vs all other tiers [ALL]  (n = 155 vs 30, court mix distance 0.80)
43 feature(s) pass the false-discovery threshold (q < 0.1); strongest shown.

| Feature | In non-serious | In other tiers | Effect | p | q | Notes |
|---|---|---|---|---|---|---|
| Hospital stay, days | median 1 (stated in 41%) | median 38 (stated in 23%) | -0.94 | <0.001 | <0.001 | court proxy (AUC 0.04) |
| Number of operations received | median 0 (stated in 54%) | median 3 (stated in 63%) | -0.76 | <0.001 | <0.001 | court proxy (AUC 0.12) |
| Number of surgical treatments received | median 0, mean 0.5 | median 3, mean 3 | -0.69 | <0.001 | <0.001 | court proxy (AUC 0.10) |
| Number of different loss categories | median 2, mean 1.9 | median 5, mean 4.4 | -0.62 | <0.001 | <0.001 | court proxy (AUC 0.14) |
| Number of losses claimed | median 2, mean 2.1 | median 5, mean 4.5 | -0.61 | <0.001 | <0.001 | court proxy (AUC 0.15) |
| Highest VMIS surgery tier received (0 none, 1 Minor ... 4 Complex) | median 0, mean 0.55 | median 3, mean 2.2 | -0.56 | <0.001 | <0.001 | court proxy (AUC 0.17) |
| Sick leave taken, days | median 142 (stated in 60%) | median 472 (stated in 20%) | -0.54 | 0.026 | 0.064 |  |
| Injuries of type permanent injury | median 0, mean 0.43 | median 2, mean 2.2 | -0.51 | <0.001 | <0.001 | court proxy (AUC 0.17) |
| Surgeries received that are not in the VMIS schedule | median 0, mean 0.11 | median 1, mean 1.2 | -0.46 | <0.001 | <0.001 |  |
| Number of injuries | median 2, mean 2.9 | median 4, mean 5 | -0.46 | <0.001 | <0.001 |  |
| Loss claimed: loss of congenial employment | 21% (33/155) | 67% (20/30) | -0.45 | <0.001 | <0.001 |  |
| Manifestations of type neurological deficit | median 0, mean 0.41 | median 1, mean 1.5 | -0.45 | <0.001 | <0.001 | court proxy (AUC 0.18) |

#### serious vs all other tiers [ALL]  (n = 6 vs 179, court mix distance 0.13)
**No feature passes the false-discovery threshold (q < 0.1).** Showing the 8 smallest p-values as exploratory hints only.

| Feature | In serious | In other tiers | Effect | p | q | Notes |
|---|---|---|---|---|---|---|
| Hospital stay, days | median 31 (stated in 50%) | median 1 (stated in 38%) | +0.85 | 0.011 | 0.611 |  |
| ICD-11 NA82: Fracture of rib, sternum or thoracic spine | 33% (2/6) | 2% (4/179) | +0.31 | 0.012 | 0.611 |  |
| ICD-11 NA02: Fracture of skull or facial bones | 33% (2/6) | 5% (9/179) | +0.28 | 0.042 | 1.000 |  |
| Number of Minor VMIS surgeries received | median 0.5, mean 0.67 | median 0, mean 0.25 | +0.32 | 0.050 | 1.000 |  |
| Number of symptoms / functional deficits | median 4, mean 4.8 | median 7, mean 7.6 | -0.46 | 0.054 | 1.000 |  |
| ICD-11 NB52: Fracture of lumbar spine or pelvis | 33% (2/6) | 7% (12/179) | +0.27 | 0.067 | 1.000 |  |
| Manifestations of type neurological deficit | median 0, mean 0 | median 0, mean 0.61 | -0.35 | 0.082 | 1.000 |  |
| Manifestations of type symptom | median 1, mean 2.3 | median 3, mean 3.1 | -0.39 | 0.098 | 1.000 |  |

#### substantial+ vs all other tiers [ALL]  (n = 24 vs 161, court mix distance 1.00)
43 feature(s) pass the false-discovery threshold (q < 0.1); strongest shown.

| Feature | In substantial+ | In other tiers | Effect | p | q | Notes |
|---|---|---|---|---|---|---|
| Hospital stay, days | median 55 (stated in 17%) | median 1 (stated in 42%) | +0.91 | 0.002 | 0.006 | court proxy (AUC 0.04) |
| Number of surgical treatments received | median 3, mean 3.5 | median 0, mean 0.52 | +0.79 | <0.001 | <0.001 | court proxy (AUC 0.10) |
| Number of operations received | median 3 (stated in 71%) | median 0 (stated in 53%) | +0.76 | <0.001 | <0.001 | court proxy (AUC 0.12) |
| Number of different loss categories | median 6, mean 5 | median 2, mean 2 | +0.73 | <0.001 | <0.001 | court proxy (AUC 0.14) |
| Number of losses claimed | median 6, mean 5 | median 2, mean 2.1 | +0.70 | <0.001 | <0.001 | court proxy (AUC 0.15) |
| Highest VMIS surgery tier received (0 none, 1 Minor ... 4 Complex) | median 3, mean 2.7 | median 0, mean 0.55 | +0.66 | <0.001 | <0.001 | court proxy (AUC 0.17) |
| Injuries of type permanent injury | median 2, mean 2.7 | median 0, mean 0.43 | +0.66 | <0.001 | <0.001 | court proxy (AUC 0.17) |
| Manifestations of type neurological deficit | median 1.5, mean 1.9 | median 0, mean 0.4 | +0.64 | <0.001 | <0.001 | court proxy (AUC 0.18) |
| Surgeries received that are not in the VMIS schedule | median 1, mean 1.4 | median 0, mean 0.12 | +0.55 | <0.001 | <0.001 |  |
| Sick leave taken, days | median 472 (stated in 25%) | median 142 (stated in 58%) | +0.54 | 0.026 | 0.068 |  |
| Number of symptoms / functional deficits | median 10, mean 10 | median 7, mean 7.1 | +0.52 | <0.001 | <0.001 |  |
| Number of injuries | median 4.5, mean 5.3 | median 2, mean 3 | +0.52 | <0.001 | <0.001 |  |

## 4. Features that rise or fall with severity

Spearman correlation of each feature with the tier order (+1 = always higher in higher tiers).

**HKDC** — 6 feature(s) pass q < 0.1.

| Feature | correlation | p | q | cases |
|---|---|---|---|---|
| ICD-11 NA22: Fracture of neck | +0.40 | <0.001 | <0.001 | 161 |
| ICD-11 NA82: Fracture of rib, sternum or thoracic spine | +0.39 | <0.001 | <0.001 | 161 |
| Hospital stay, days | +0.35 | 0.004 | 0.066 | 67 |
| ICD-11 NB52: Fracture of lumbar spine or pelvis | +0.28 | <0.001 | 0.008 | 161 |
| ICD-11 NA02: Fracture of skull or facial bones | +0.28 | <0.001 | 0.008 | 161 |
| Treatment includes icu | +0.27 | <0.001 | 0.009 | 161 |

**ALL** — 43 feature(s) pass q < 0.1.

| Feature | correlation | p | q | cases |
|---|---|---|---|---|
| Number of operations received | +0.54 | <0.001 | <0.001 | 103 |
| Number of surgical treatments received | +0.51 | <0.001 | <0.001 | 185 |
| Number of Complex VMIS surgeries received | +0.50 | <0.001 | <0.001 | 185 |
| Hospital stay, days | +0.50 | <0.001 | <0.001 | 71 |
| Surgeries received that are not in the VMIS schedule | +0.48 | <0.001 | <0.001 | 185 |
| Highest VMIS surgery tier received (0 none, 1 Minor ... 4 Complex) | +0.44 | <0.001 | <0.001 | 185 |
| Loss claimed: loss of independence | +0.43 | <0.001 | <0.001 | 185 |
| Loss claimed: loss of sexual function and sexual life | +0.43 | <0.001 | <0.001 | 185 |
| Injuries of type permanent injury | +0.42 | <0.001 | <0.001 | 185 |
| Number of different loss categories | +0.41 | <0.001 | <0.001 | 185 |
| Number of losses claimed | +0.41 | <0.001 | <0.001 | 185 |
| Loss claimed: loss of the senses | +0.39 | <0.001 | <0.001 | 185 |

## 5. Decision rules

A shallow, class-balanced decision tree gives if/then rules for the tiers. Rules from the pooled data inherit any court confound.

### Decision rules: HKDC (161 cases)

| If ... | predict | cases | actual tiers in this group | right |
|---|---|---|---|---|
| [Sick leave taken, days] stated AND [Age at accident] > 19.5 (or not stated) | non-serious | 88 | non-serious 88, serious 0 | 100% |
| [Sick leave taken, days] not stated AND NOT [Plaintiff is male] | non-serious | 20 | non-serious 20, serious 0 | 100% |
| [Sick leave taken, days] not stated AND [Plaintiff is male] (or not stated) AND [Hospital stay, days] <= 19 | non-serious | 14 | non-serious 14, serious 0 | 100% |
| [Sick leave taken, days] stated AND [Age at accident] <= 19.5 | non-serious | 5 | non-serious 5, serious 0 | 100% |
| [Sick leave taken, days] not stated AND [Plaintiff is male] (or not stated) AND [Hospital stay, days] > 19 (or not stated) | serious | 34 | non-serious 28, serious 6 | 18% |

**How far to trust the rules** (same tree refit on 50 cross-validation splits, scored on cases it did not see):

|  | macro-F1 | QWK | balanced accuracy | accuracy |
|---|---|---|---|---|
| decision tree | 0.48 ± 0.06 | 0.02 ± 0.12 | 0.55 ± 0.19 | 0.81 ± 0.08 |
| always predict the biggest tier | 0.49 ± 0.00 | 0.00 ± 0.00 | 0.50 ± 0.00 | 0.96 ± 0.01 |

Recall by tier on unseen cases: non-serious 83%, serious 27%.

Most stable first splits across the cross-validation trees: Sick leave taken, days (82%), Occupation changed after accident (12%), Hospital stay, days (4%), Manifestations of type symptom (2%).

### Decision rules: ALL (185 cases)

| If ... | predict | cases | actual tiers in this group | right |
|---|---|---|---|---|
| NOT [Manifestations of type neurological deficit] AND [Sick leave taken, days] stated AND NOT [Surgeries received that are not in the VMIS schedule] | non-serious | 64 | non-serious 64, serious 0, substantial+ 0 | 100% |
| [Manifestations of type neurological deficit] AND [Hospital stay, days] <= 35.5 | non-serious | 19 | non-serious 19, serious 0, substantial+ 0 | 100% |
| [Manifestations of type neurological deficit] AND [Hospital stay, days] > 35.5 (or not stated) AND [Sick leave taken, days] <= 642 | non-serious | 16 | non-serious 15, serious 0, substantial+ 1 | 94% |
| NOT [Manifestations of type neurological deficit] AND [Sick leave taken, days] not stated AND [Highest VMIS surgery tier received] <= 1.5 | serious | 46 | non-serious 40, serious 6, substantial+ 0 | 13% |
| [Manifestations of type neurological deficit] AND [Hospital stay, days] > 35.5 (or not stated) AND [Sick leave taken, days] > 642 (or not stated) | substantial+ | 28 | non-serious 9, serious 0, substantial+ 19 | 68% |
| NOT [Manifestations of type neurological deficit] AND [Sick leave taken, days] not stated AND [Highest VMIS surgery tier received] > 1.5 | substantial+ | 7 | non-serious 5, serious 0, substantial+ 2 | 29% |
| NOT [Manifestations of type neurological deficit] AND [Sick leave taken, days] stated AND [Surgeries received that are not in the VMIS schedule] | substantial+ | 5 | non-serious 3, serious 0, substantial+ 2 | 40% |

**How far to trust the rules** (same tree refit on 50 cross-validation splits, scored on cases it did not see):

|  | macro-F1 | QWK | balanced accuracy | accuracy |
|---|---|---|---|---|
| decision tree | 0.47 ± 0.07 | 0.37 ± 0.15 | 0.64 ± 0.17 | 0.65 ± 0.09 |
| always predict the biggest tier | 0.30 ± 0.00 | 0.00 ± 0.00 | 0.33 ± 0.00 | 0.84 ± 0.00 |

Recall by tier on unseen cases: non-serious 64%, serious 56%, substantial+ 73%.

Most stable first splits across the cross-validation trees: Highest VMIS surgery tier received (0 none, 1 Minor ... 4 Complex) (34%), Manifestations of type neurological deficit (32%), Number of symptoms / functional deficits (20%), Number of different loss categories (6%), Injuries of type permanent injury (4%).

## 6. How to read the numbers

- **Effect** is the difference in share of cases having the feature (flags), or Cliff's delta (numbers: the chance a random case from the first group is higher than one from the second, minus the reverse). Both run from -1 to +1; positive means more in the first group.
- **p** is from Fisher's exact test (flags) or the Mann-Whitney test (numbers). **q** adjusts p for testing many features at once (Benjamini-Hochberg); trust q, not p.
- A feature that is "stated" or not is a property of the judgment text as well as the injury.
- Small tiers give wide uncertainty. A pattern in fewer than about 10 cases is a lead to check, not a finding.
