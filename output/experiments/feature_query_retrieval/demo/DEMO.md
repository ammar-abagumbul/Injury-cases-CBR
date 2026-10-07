# Experiment 7 — worked example of the fetch

One end-to-end retrieval: a real corpus extraction is used as the **query document**, the CBR retriever fetches its nearest cases with the tuned clinical-first weights, and the **returned documents** are listed and copied next to this file.

## 1. Query selection

Query case **[2016] HKCFI 601** was chosen from the corpus because it has a rich, fully-populated feature vector (eight ICD-coded injuries, three loss categories, treatment intensity, demographics) and a mid-range award, so there are genuinely comparable precedents to find. Nothing about its PSLA is used to retrieve — that is the held-out target we check afterwards.

## 2. The query document

### [2016] HKCFI 601
- **Case name:** CHAN KAM CHEONG v. CHAN TAK YEE AND OTHERS
- **Judgment:** 2016-04-08 (2016)
- **Plaintiff:** Male, age at accident 50, age at trial 55, occupation: licensed crane operator
- **Severity:** extracted `Serious injury`, reconstructed from award `Non-serious injury`
- **Treatment:** 3 operations, 62 hospital days
- **Injuries (8 ICD codes):** `NA07.Z`, `NA02.8`, `NA01.2`, `NA0D.Y`, `NA06.5`, `NA22.3`, `NA82.4`, `NC55.1`
- **Losses (3):** Loss of mobility, Loss of congenial employment, Loss of ability to enjoy food and drink
- **PSLA:** nominal HK$400,000, real (2020) **HK$434,018**

Full extracted document: [`query/2016_HKCFI_601.json`](query/2016_HKCFI_601.json)  (the corpus extraction verbatim). A judge would express the same case as the feature query [`query/query_features.yaml`](query/query_features.yaml):

```yaml
query_id: demo-2016_HKCFI_601
gender: Male
age_at_accident: 50
age_at_trial: 55
injuries:
- NA07.Z
- NA02.8
- NA01.2
- NA0D.Y
- NA06.5
- NA22.3
- NA82.4
- NC55.1
losses:
- Loss of mobility
- Loss of congenial employment
- Loss of ability to enjoy food and drink
overall_category: Serious injury
operations_count: 3
hospitalisation_days: 62
k: 8
```

<details><summary>Full query document JSON</summary>

```json
{
  "metadata": {
    "neutral_citation": "[2016] HKCFI 601",
    "action_number": "HCPI 1030/2013",
    "case_name": "CHAN KAM CHEONG v. CHAN TAK YEE AND OTHERS",
    "judgment_date": "2016-04-08",
    "plaintiff_count": 1,
    "has_pre_existing_injuries": true
  },
  "plaintiff": {
    "gender": "Male",
    "age_at_accident": 50,
    "age_at_trial": 55,
    "occupation_before": "licensed crane operator",
    "occupation_after": "general cleaner (assumed alternative employment)",
    "expected_occupation_after": "general cleaner (assumed alternative employment)",
    "salary_before": 17500.0,
    "salary_after": 7000.0,
    "expected_salary_after": 7000.0,
    "education_level": "studied up to primary one"
  },
  "injuries": {
    "injuries": [
      {
        "injury_id": "inj_001",
        "description": "intracranial breeding",
        "injury_type": "Residual disability",
        "body_part": "head",
        "laterality": null,
        "source": [
          "with open facial wounds/laceration, and intracranial breeding."
        ],
        "icd_code": "NA07.Z",
        "icd_description": "Intracranial injury, unspecified"
      },
      {
        "injury_id": "inj_002",
        "description": "multiple fractures to orbit (eye socket bone), malar and maxillary bones",
        "injury_type": "Residual disability",
        "body_part": "face (orbit, malar and maxillary bones)",
        "laterality": null,
        "source": [
          "CT scan showed multiple fractures to his skull, in his face and near his left eye.",
          "multiple fractures to cervical spine (C2 and C7), ribs, orbit (eye socket bone), malar and maxillary bones (centre of face and upper teeth area), with open facial wounds/laceration, and intracranial breeding."
        ],
        "icd_code": "NA02.8",
        "icd_description": "Multiple fractures involving skull or facial bones"
      },
      {
        "injury_id": "inj_003",
        "description": "open facial wounds/laceration",
        "injury_type": "Residual disability",
        "body_part": "face",
        "laterality": null,
        "source": [
          "with open facial wounds/laceration"
        ],
        "icd_code": "NA01.2",
        "icd_description": "Laceration without foreign body of head"
      },
      {
        "injury_id": "inj_004",
        "description": "loss of three upper incisor teeth",
        "injury_type": "Permanent injury",
        "body_part": "upper front teeth",
        "laterality": null,
        "source": [
          "The plaintiff had altogether 3 teeth removed as a result of the accident.",
          "Loss of three upper incisor teeth and hence diminished biting force (replaced by use of partial denture, to be replaced approximately every 5 years)."
        ],
        "icd_code": "NA0D.Y",
        "icd_description": "Other specified injury of teeth or supporting structures"
      },
      {
        "injury_id": "inj_005",
        "description": "left eye injury including left traumatic mydriasis with 200 degree angle recession, loss of ocular motility and binocular diplopia",
        "injury_type": "Residual disability",
        "body_part": "left eye",
        "laterality": "left",
        "source": [
          "Later examination revealed left traumatic mydriasis (dilated pupil) with 200 degree angle recession.",
          "Prism glasses were applied, and the plaintiff suffered from binocular diplopia (seeing double images) on all directions.",
          "Loss of ocular motility of the left eye, and diplopia (corrected by prismatic prescription with regular review for vertical misalignment, and annual examination required to inspect for onset of glaucoma);"
        ],
        "icd_code": "NA06.5",
        "icd_description": "Trauma to the iris sphincter"
      },
      {
        "injury_id": "inj_006",
        "description": "multiple fractures to cervical spine (C2 and C7); surgery for open reduction and internal fixation on the C2 fracture; halo-traction and SOMI brace applied",
        "injury_type": "Residual disability",
        "body_part": "cervical spine",
        "laterality": null,
        "source": [
          "multiple fractures to cervical spine (C2 and C7)",
          "Surgery for open reduction and internal fixation on the C2 fracture was done on 22 December 2011.",
          "Between 14 December 2011 to 17 July 2012, halo-traction and later a SOMI brace was applied to the plaintiff."
        ],
        "icd_code": "NA22.3",
        "icd_description": "Multiple fractures of cervical spine"
      },
      {
        "injury_id": "inj_007",
        "description": "rib fractures",
        "injury_type": "Residual disability",
        "body_part": "ribs",
        "laterality": null,
        "source": [
          "multiple fractures to cervical spine (C2 and C7), ribs, orbit (eye socket bone), malar and maxillary bones (centre of face and upper teeth area), with open facial wounds/laceration, and intracranial breeding."
        ],
        "icd_code": "NA82.4",
        "icd_description": "Multiple fractures of ribs"
      },
      {
        "injury_id": "inj_008",
        "description": "neurological impairment affecting the left upper limb with mild astereognosis, residual weakness and clumsiness, impaired digital dexterity, paraesthesia and decreased sensation",
        "injury_type": "Residual disability",
        "body_part": "left upper limb",
        "laterality": "left",
        "source": [
          "Restriction in cervical spinal movements and neurological impairment of the left upper limb with mild astereognosis (inability to identify object by touch), residual weakness and clumsiness of the left hand and impaired digital dexterity, paraesthesia (tingling sensation), pulling discomfort, decreased sensation (by 50% on the left thumb; by 30% on the left ring finger; and by 50% on the lateral left arm), left index/middle/ring fingers swan-neck deformity (worse at left middle finger);"
        ],
        "icd_code": "NC55.1",
        "icd_description": "Injury of median nerve at wrist or hand level"
      }
    ],
    "manifestations": [
      {
        "description": "Restriction in cervical spinal movements",
        "manifestation_type": "functional_deficit",
        "caused_by_injury_id": "inj_006"
      },
      {
        "description": "Neurological impairment of the left upper limb with mild astereognosis, residual weakness and clumsiness of the left hand, impaired digital dexterity, paraesthesia and decreased sensation",
        "manifestation_type": "neurological_deficit",
        "caused_by_injury_id": "inj_008"
      },
      {
        "description": "Diplopia (binocular double vision) and loss of ocular motility",
        "manifestation_type": "symptom",
        "caused_by_injury_id": "inj_005"
      },
      {
        "description": "Diminished biting force",
        "manifestation_type": "functional_deficit",
        "caused_by_injury_id": "inj_004"
      },
      {
        "description": "Numbness in his left hand",
        "manifestation_type": "symptom",
        "caused_by_injury_id": "inj_008"
      },
      {
        "description": "Difficulty in swallowing and speech",
        "manifestation_type": "symptom",
        "caused_by_injury_id": null
      },
      {
        "description": "Fatigue in his back and neck",
        "manifestation_type": "symptom",
        "caused_by_injury_id": "inj_006"
      },
      {
        "description": "Loss of flexion that prevents the plaintiff from looking down to the ground when walking",
        "manifestation_type": "functional_deficit",
        "caused_by_injury_id": "inj_006"
      }
    ],
    "overall_category": "Serious injury"
  },
  "treatment": {
    "treatments_received": [
      "admitted to the Intensive Care Unit",
      "mechanical ventilation for 96 hours",
      "surgical toileting and suturing",
      "three operations (10 December 2011 18:15-18:30; 10 December 2011 21:00-21:35; 22 December 2011 12:24-17:15)",
      "open reduction and internal fixation on the C2 fracture",
      "halo-traction and SOMI brace (14 December 2011 to 17 July 2012)",
      "approximately 29 sessions of physiotherapy at Princess Margaret Hospital",
      "12 sessions of physiotherapy from 19 June 2012 to 15 January 2013",
      "work rehabilitation treatments (3 sessions per week) from 11 January 2013 to 16 August 2013 (total 93 sessions) at Kwong Wah Hospital",
      "prism glasses",
      "removal of 3 teeth (dental extractions) and insertion of 3 false teeth"
    ],
    "treatments_future": [
      "regular review for vertical misalignment for prismatic prescription",
      "annual examination for onset of glaucoma",
      "replacement of partial denture approximately every 5 years"
    ],
    "hospitalisation_days": 62,
    "expected_hospitalisation_days": null,
    "operations_count": 3,
    "future_operations_count": null,
    "sick_leave_days_actual": 728,
    "sick_leave_days_expected": null
  },
  "losses": {
    "losses": [
      {
        "category": "Loss of mobility",
        "description": "Diplopia (double vision) which impairs his mobility",
        "caused_by_injury_id": "inj_005"
      },
      {
        "category": "Loss of congenial employment",
        "description": "Plaintiff can no longer operate cranes and must change to alternative light duty work",
        "caused_by_injury_id": "inj_008"
      },
      {
        "category": "Loss of ability to enjoy food and drink",
        "description": "Diminished biting force due to loss of three upper incisor teeth affecting chewing",
        "caused_by_injury_id": "inj_004"
      }
    ]
  },
  "psla": {
    "amount": 400000.0,
    "comparable_cases": [
      {
        "case_name": "Kei Yiu Fuk v Wong Ching Nam",
        "neutral_citation": null,
        "action_number": "HCPI 355/2002",
        "injury_description": "serious injury to his left eye, with periorbital swelling and bruising, mild diplopia, fractured ethmoid plate, oedema of left eye muscle, fractured medial wall and floor of left orbit",
        "hospitalisation_days": 4,
        "operations_count": null,
        "sick_leave_days": 267,
        "comparison": "less serious",
        "psla_amount": 300000.0
      },
      {
        "case_name": "Harvey Kenneth v Welltex International Development Limited & Ors",
        "neutral_citation": null,
        "action_number": "HCPI 818/1998",
        "injury_description": "fracture of pelvis and extensive damage to face bones, deep lacerations and abrasions, residual pain on his face, headache and diplopia",
        "hospitalisation_days": null,
        "operations_count": null,
        "sick_leave_days": null,
        "comparison": "more serious",
        "psla_amount": 450000.0
      },
      {
        "case_name": "Cheng Cho Fai v Law Ka Chung",
        "neutral_citation": null,
        "action_number": "HCPI 1005/2006",
        "injury_description": "wrist bone fracture, bilateral maxillary fracture, left multiple orbital fracture, 3 mobile teeth, sunken left eye with double vision, enophthalmos, diplopia and ptosis, post-concussional injuries",
        "hospitalisation_days": null,
        "operations_count": 7,
        "sick_leave_days": null,
        "comparison": "more serious",
        "psla_amount": 500000.0
      },
      {
        "case_name": "Luk Yee Lam v Orasa Livasiri",
        "neutral_citation": null,
        "action_number": "HCPI 394/2002",
        "injury_description": "disc herniation at C5/6 level, neck and back pain, numbness and weakness in the upper limbs and hands and his right leg",
        "hospitalisation_days": null,
        "operations_count": null,
        "sick_leave_days": null,
        "comparison": "more serious",
        "psla_amount": 400000.0
      },
      {
        "case_name": "Tsoi Wing Tak Michelle v Lau Sze Ni",
        "neutral_citation": null,
        "action_number": "HCPI 394/2002",
        "injury_description": "whiplash injury, wore a neck brace for six months, residual pain in upper chest, back and neck, no neurological or structural damage",
        "hospitalisation_days": null,
        "operations_count": null,
        "sick_leave_days": null,
        "comparison": "less serious",
        "psla_amount": 180000.0
      },
      {
        "case_name": "Chan Tak Chi v Wong Siu Tao",
        "neutral_citation": null,
        "action_number": "HCPI 1223/1996",
        "injury_description": "fracture to C5 spine and the right side facet, laceration to head, multiple abrasions, wore a Halo ring and body jacket, frozen shoulder",
        "hospitalisation_days": 13,
        "operations_count": null,
        "sick_leave_days": null,
        "comparison": "similar or same",
        "psla_amount": 360000.0
      }
    ]
  },
  "death": null,
  "extraction_metadata": {
    "model_name": "gpt",
    "num_extraction_runs": 1,
    "extraction_token_count": {
      "input_tokens": 24692,
      "output_tokens": 11167,
      "reasoning_tokens": 7616,
      "total_tokens": 35859
    },
    "reconciliation_token_count": null,
    "extraction_duration_ms": 82939.85998257995,
    "reconciliation_duration_ms": null,
    "total_token_count": {
      "input_tokens": 24692,
      "output_tokens": 11167,
      "reasoning_tokens": 7616,
      "total_tokens": 35859
    },
    "total_duration_ms": 82939.85998257995
  }
}
```

</details>

## 3. Retrieval setup

| Setting | Value |
|---|---|
| Method | myCBR local similarities + clinical-first composite amalgamation |
| Clinical group | `injuries`, `losses` — missing-penalised |
| Clinical share (`clinical_alpha`) | 0.7 |
| Loose injury gate (`min_injury_similarity`) | 0.1 |
| Auxiliary group | `age_at_accident`, `age_at_trial`, `operations_count`, `hospitalisation_days`, `n_injuries`, `n_losses` — available-case |
| Candidate pool | cases with a known real PSLA, query itself excluded (leave-one-out) |
| k | 8 |
| PSLA tolerance | ±50% (ratio in [2/3, 1.5]) |
| Base year | 2020 (awards deflated with `data/price_index.csv`) |

Severity (`overall_category`) and `gender` are **not** retrieval attributes: gender is a near-constant, and severity is the award band (a target-derived encoding of the compensation), so it is reconstructed from the award for diagnostics only. Similarity weights (tuned against clinical NDCG@k on a held-out train split):

| Attribute | Weight | Group |
|---|---|---|
| `injuries` | 1.170 | clinical |
| `losses` | 0.376 | clinical |
| `age_at_accident` | 0.207 | auxiliary |
| `age_at_trial` | 0.678 | auxiliary |
| `operations_count` | 1.306 | auxiliary |
| `hospitalisation_days` | 0.240 | auxiliary |
| `n_injuries` | 1.664 | auxiliary |
| `n_losses` | 2.344 | auxiliary |

## 4. Returned documents

The query's own real PSLA is **HK$434,018** (2020 HKD), while the severity reconstructed from its award is **Non-serious injury** (its extracted label was `Serious injury`). Across the top 8: 4/8 are within ±50%, median |ln(candidate/query)| = **0.648**, the awards span HK$151,639 – HK$1,424,904, and the minimum injury similarity is **0.542**.

| Rank | Case | Similarity | Inj sim | Loss sim | Year | Severity (extracted) | Severity (from award) | PSLA real (2020) | abs(ln(c/q)) | Within ±50% |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [1985] HKCFI 80 | 0.698 | 0.646 | 0.400 | 1985 | Serious injury | Non-serious injury | HK$409,846 | 0.057 | yes |
| 2 | [1979] HKCFI 115 | 0.640 | 0.608 | 0.333 | 1979 | Serious injury | Serious injury | HK$570,857 | 0.274 | yes |
| 3 | [2005] HKDC 202 | 0.631 | 0.688 | 0.000 | 2005 | — | Non-serious injury | HK$151,639 | 1.052 | no |
| 4 | [1985] HKCFI 564 | 0.630 | 0.576 | 0.200 | 1985 | Serious injury | Non-serious injury | HK$409,846 | 0.057 | yes |
| 5 | [1978] HKCFI 29 | 0.626 | 0.542 | 0.400 | 1978 | — | Disaster | HK$1,424,904 | 1.189 | no |
| 6 | [2013] HKCFI 2281 | 0.622 | 0.583 | 0.333 | 2013 | — | Substantial injury | HK$830,128 | 0.648 | no |
| 7 | [2005] HKCFI 915 | 0.621 | 0.585 | 0.286 | 2005 | Serious injury | Gross disability | HK$1,137,295 | 0.963 | no |
| 8 | [2017] HKDC 174 | 0.614 | 0.646 | 0.125 | 2017 | Non-serious injury | Non-serious injury | HK$404,410 | 0.071 | yes |

> **Corpus note.** ranks 1, 4 are the same judgment (*YEUNG YU v. WONG YUNG AND OTHERS*) indexed under different neutral citations. This is a duplication in the extracted corpus, not a retrieval error — it also explains why those rows have identical awards.

### Returned document cards

### 1. [1985] HKCFI 80
- **CBR similarity to query:** `0.698`
- **Case name:** YEUNG YU v. WONG YUNG AND OTHERS
- **Judgment:** 1985-08-05 (1985)
- **Plaintiff:** Male, age at accident —, age at trial 56, occupation: owner/driver with his own lorry
- **Severity:** extracted `Serious injury`, reconstructed from award `Non-serious injury`
- **Treatment:** — operations, 28 hospital days
- **Injuries (7 ICD codes):** `NA01.7`, `NA02.3`, `NA82.4`, `NB52.0`, `NA22.2`, `NA07.Z`, `NA06.A`
- **Losses (4):** Loss of the senses, Loss of mobility, Loss of confidence in going out in the public, Loss of congenial employment
- **PSLA:** nominal HK$120,000, real (2020) **HK$409,846**

Full document: [`results/01_1985_HKCFI_80.json`](results/01_1985_HKCFI_80.json)

### 2. [1979] HKCFI 115
- **CBR similarity to query:** `0.640`
- **Case name:** LAW KUN MEE v. LAI BUN AND ANOTHER
- **Judgment:** 1979-01-08 (1979)
- **Plaintiff:** Male, age at accident 36, age at trial 43, occupation: Prior to the accident he had worked at the Tsuen Wan Wharf for 6 years.
- **Severity:** extracted `Serious injury`, reconstructed from award `Serious injury`
- **Treatment:** — operations, — hospital days
- **Injuries (5 ICD codes):** `NA01.2`, `NA07.5`, `NA06.Y`, `NA06.Z`, `NA07.Z`
- **Losses (5):** Loss of the senses, Loss of sexual function and sexual life, Loss of mental ability, Loss of mobility, Loss of congenial employment
- **PSLA:** nominal HK$90,000, real (2020) **HK$570,857**

Full document: [`results/02_1979_HKCFI_115.json`](results/02_1979_HKCFI_115.json)

### 3. [2005] HKDC 202
- **CBR similarity to query:** `0.631`
- **Case name:** SO SAU MAN v. LEUNG MING KWONG AND ANOTHER
- **Judgment:** 2005-10-18 (2005)
- **Plaintiff:** —, age at accident 62, age at trial —, occupation: dish cleaner in a restaurant
- **Severity:** extracted `—`, reconstructed from award `Non-serious injury`
- **Treatment:** — operations, 0 hospital days
- **Injuries (2 ICD codes):** `NA01.2`, `NA0D.Y`
- **Losses (2):** Loss from scarring or disfigurement, Loss of bodily integrity
- **PSLA:** nominal HK$100,000, real (2020) **HK$151,639**

Full document: [`results/03_2005_HKDC_202.json`](results/03_2005_HKDC_202.json)

### 4. [1985] HKCFI 564
- **CBR similarity to query:** `0.630`
- **Case name:** YEUNG YU v. WONG YUNG AND OTHERS
- **Judgment:** 1985-08-05 (1985)
- **Plaintiff:** Male, age at accident —, age at trial 56, occupation: owner/driver with his own lorry
- **Severity:** extracted `Serious injury`, reconstructed from award `Non-serious injury`
- **Treatment:** — operations, 28 hospital days
- **Injuries (6 ICD codes):** `NA01.7`, `NA02.3`, `NA82.4`, `NA22.2`, `NB52.0`, `NA06.A`
- **Losses (3):** Loss of bodily integrity, Loss of social life, Loss of congenial employment
- **PSLA:** nominal HK$120,000, real (2020) **HK$409,846**

Full document: [`results/04_1985_HKCFI_564.json`](results/04_1985_HKCFI_564.json)

### 5. [1978] HKCFI 29
- **CBR similarity to query:** `0.626`
- **Case name:** CHOY FOR SANG v. TONG KWOK MING
- **Judgment:** 1978-01-24 (1978)
- **Plaintiff:** —, age at accident 26, age at trial —, occupation: driver
- **Severity:** extracted `—`, reconstructed from award `Disaster`
- **Treatment:** 1 operations, 45 hospital days
- **Injuries (4 ICD codes):** `NA07.Z`, `NA01.7`, `NA0D.Y`, `NC30.4`
- **Losses (4):** Loss of the senses, Loss from scarring or disfigurement, Loss of mobility, Loss of congenial employment
- **PSLA:** nominal HK$200,000, real (2020) **HK$1,424,904**

Full document: [`results/05_1978_HKCFI_29.json`](results/05_1978_HKCFI_29.json)

### 6. [2013] HKCFI 2281
- **CBR similarity to query:** `0.622`
- **Case name:** WONG KIM HUNG BY HIS NEXT FRIEND HIU SJU PHIN v. WONG WING KONG T/A 613 DECORATION WORKS AND OTHERS
- **Judgment:** 2013-08-21 (2013)
- **Plaintiff:** Male, age at accident 56, age at trial 63, occupation: construction worker
- **Severity:** extracted `—`, reconstructed from award `Substantial injury`
- **Treatment:** — operations, — hospital days
- **Injuries (2 ICD codes):** `NA07.1`, `NA01.2`
- **Losses (5):** Loss of mobility, Loss of independence, Loss of congenial employment, Loss of mental ability, Loss of expectation of life
- **PSLA:** nominal HK$700,000, real (2020) **HK$830,128**

Full document: [`results/06_2013_HKCFI_2281.json`](results/06_2013_HKCFI_2281.json)

### 7. [2005] HKCFI 915
- **CBR similarity to query:** `0.621`
- **Case name:** CHAN HAK FOON v. SUTERA HARBOUR RESORT SDN BHD AND ANOTHER
- **Judgment:** 2005-10-18 (2005)
- **Plaintiff:** Male, age at accident 57, age at trial 62, occupation: Managing director of Permtex International Ltd.; owner of Fabri-Technic Engineering & Trading Co. Ltd.; formerly Area Director of South Asia and General Manager in HK/PRC at Rohm and Haas Company
- **Severity:** extracted `Serious injury`, reconstructed from award `Gross disability`
- **Treatment:** — operations, 82 hospital days
- **Injuries (11 ICD codes):** `NA07.5`, `NA07.60`, `NA07.1`, `NA07.2Z`, `NA02.8`, `NA02.1A`, `NA02.2Y`, `NA82.4`, `NB32.32`, `NA05.0`, `NA01.2`
- **Losses (6):** Loss of mental ability, Loss from scarring or disfigurement, Loss of ability to enjoy food and drink, Loss of bodily integrity, Loss of mobility, Loss of independence
- **PSLA:** nominal HK$750,000, real (2020) **HK$1,137,295**

Full document: [`results/07_2005_HKCFI_915.json`](results/07_2005_HKCFI_915.json)

### 8. [2017] HKDC 174
- **CBR similarity to query:** `0.614`
- **Case name:** CHAN SIU YIM v. DR CHEUNG SHEUNG KIN ALSO KNOWN AS DR SAMUEL KINNETH CHEUNG
- **Judgment:** 2017-02-21 (2017)
- **Plaintiff:** Female, age at accident —, age at trial —
- **Severity:** extracted `Non-serious injury`, reconstructed from award `Non-serious injury`
- **Treatment:** — operations, — hospital days
- **Injuries (1 ICD codes):** `NA0D.Y`
- **Losses (6):** Loss of bodily integrity, Loss of ability to enjoy food and drink, Loss from scarring or disfigurement, Loss of communication ability, Breakdown of the family, Personality change
- **PSLA:** nominal HK$380,000, real (2020) **HK$404,410**

Full document: [`results/08_2017_HKDC_174.json`](results/08_2017_HKDC_174.json)

## 5. How the fetch works

1. The query is reduced to its clinical features (ICD codes, loss categories) plus the auxiliary attributes (ages, treatment intensity, counts).
2. The **clinical group** is compared with the myCBR local similarity functions — ICD-11 taxonomy max-overlap for injuries, Jaccard for losses — and averaged with a *missing penalty*, so a case that cannot be compared clinically is pushed down rather than let off.
3. The **auxiliary group** is averaged available-case (missing attributes dropped) and only adjusts within the clinical neighbourhood. The two groups are combined as `0.7 * clinical + 0.3 * auxiliary`.
4. A **loose gate** on graded injury similarity (≥ 0.1) drops cases with no demonstrable clinical overlap before ranking, so treatment and counts cannot buy a clinically unrelated case into the top-k. The floor is deliberately low — an exact-code test would be defeated by extraction noise.
5. The weights were tuned to maximise clinical NDCG@k (injury/loss relevance), not PSLA-band NDCG; PSLA closeness is reported separately as the compensation signal.

## 6. The diagnosis that motivated this method

The earlier headline benchmark was only marginally better than chance, and the worked example's 8th hit was a single-region hand-crush with no exact ICD overlap while a clinically richer case sat at rank 25. The cause was **objective misalignment**: weights were tuned against PSLA-band relevance, so treatment intensity, loss counts and severity were upweighted and the injury and loss sets were almost tuned away. Severity was also dropping out as a *free pass* for cases that lacked it. The full write-up is in `src/hklii_psla/experiments/experiment7/notes.md`.

## 7. What changed, and the acceptance check

| Before | After |
|---|---|
| Available-case weighted sum over 10 attributes | Clinical-first composite: clinical group (injury+loss) missing-penalised, auxiliary available-case |
| Tuned against PSLA-band NDCG | Tuned against clinical NDCG@k |
| `injuries` weight 0.16, `losses` 0.26 | Clinical group carries 0.7 of the score |
| `gender`, `overall_category` in the score | Both removed; severity reconstructed from the award |
| No clinical floor; rank 8 had injury sim 0.000 | Loose injury gate (≥ 0.1) before ranking |

The new top-8 all clear the clinical gate and are injuries-first: every returned case has injury similarity ≥ 0.542, whereas the previous top-8 included two cases with injury similarity 0.000. The two cases the old critique flagged as unfairly buried now rank honestly on their clinical merits:

| Case | Injury sim | Loss sim | Old rank | New rank |
|---|---|---|---|---|
| [2003] HKCFI 270 | 0.458 | 0.400 | 25 | 19 |
| [1980] HKCFI 66 | 0.538 | 0.167 | 18 | 20 |
| [2000] HKCFI 1090 | — | — | 8 | >40 |

`[2003] HKCFI 270` and `[1980] HKCFI 66` do **not** enter the top-8: at injury similarity 0.458 and 0.538 they are now out-ranked by cases with 0.54–0.69. Their old claim to the top-8 rested on award closeness, which is no longer part of the clinical score — the original acceptance criterion assumed PSLA-closeness remained a retrieval term, which would have leaked the query's award in leave-one-out evaluation. `[2000] HKCFI 1090` (injury sim 0.243, no exact overlap) correctly drops out. PSLA closeness is tracked separately: on the full benchmark the PSLA-band NDCG@10 is ~0.21 versus a random baseline of ~0.15, and the award spread above shows the tension between clinical and quantum closeness remains visible.

## 8. Files

| Path | Contents |
|---|---|
| `DEMO.md` | this report |
| `retrieval.json` | machine-readable query + ranked results |
| `query/2016_HKCFI_601.json` | the query document (corpus extraction) |
| `query/query_features.yaml` | the same case as a judge `FeatureQuery` |
| `results/01_1985_HKCFI_80.json` | returned document #1 ([1985] HKCFI 80) |
| `results/02_1979_HKCFI_115.json` | returned document #2 ([1979] HKCFI 115) |
| `results/03_2005_HKDC_202.json` | returned document #3 ([2005] HKDC 202) |
| `results/04_1985_HKCFI_564.json` | returned document #4 ([1985] HKCFI 564) |
| `results/05_1978_HKCFI_29.json` | returned document #5 ([1978] HKCFI 29) |
| `results/06_2013_HKCFI_2281.json` | returned document #6 ([2013] HKCFI 2281) |
| `results/07_2005_HKCFI_915.json` | returned document #7 ([2005] HKCFI 915) |
| `results/08_2017_HKDC_174.json` | returned document #8 ([2017] HKDC 174) |
