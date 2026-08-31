| Path | Metric | Score | Passed | Reason | Matched | Missed | Spurious | P | R | F1 |
|------|--------|-------|--------|--------|---------|--------|----------|---|---|----| 
| metadata.neutral_citation | string_exact | 1.000 | Yes |  | | | | | | |
| metadata.action_number | string_exact | 1.000 | Yes |  | | | | | | |
| metadata.case_name | string_fuzzy | 1.000 | Yes |  | | | | | | |
| metadata.judgment_date | string_exact | 1.000 | Yes |  | | | | | | |
| metadata.plaintiff_count | number_exact | 1.000 | Yes |  | | | | | | |
| metadata.has_pre_existing_injuries | boolean_exact | 1.000 | Yes |  | | | | | | |
| plaintiff.gender | string_exact | 1.000 | Yes |  | | | | | | |
| plaintiff.age_at_accident | number_exact | 1.000 | Yes |  | | | | | | |
| plaintiff.age_at_trial | number_exact | 1.000 | Yes |  | | | | | | |
| plaintiff.occupation_after | string_semantic | 1.000 | Yes | both_null | | | | | | |
| plaintiff.expected_occupation_after | string_semantic | 1.000 | Yes | both_null | | | | | | |
| plaintiff.salary_before | number_exact | 1.000 | Yes |  | | | | | | |
| plaintiff.salary_after | number_exact | 1.000 | Yes | both_null | | | | | | |
| plaintiff.expected_salary_after | number_exact | 1.000 | Yes | both_null | | | | | | |
| plaintiff.education_level | string_semantic | 1.000 | Yes | both_null | | | | | | |
| injuries.overall_category | string_exact | 1.000 | Yes |  | | | | | | |
| treatment.hospitalisation_days | number_exact | 1.000 | Yes | both_null | | | | | | |
| treatment.expected_hospitalisation_days | number_exact | 1.000 | Yes | both_null | | | | | | |
| treatment.operations_count | number_exact | 1.000 | Yes |  | | | | | | |
| treatment.future_operations_count | number_exact | 1.000 | Yes | both_null | | | | | | |
| treatment.sick_leave_days_actual | number_exact | 1.000 | Yes | both_null | | | | | | |
| treatment.sick_leave_days_expected | number_exact | 1.000 | Yes | both_null | | | | | | |
| psla.amount | number_tolerance | 1.000 | Yes |  | | | | | | |
| injury_loss_relations.items.injury | string_semantic | 1.000 | Yes | both_null | | | | | | |
| injury_loss_relations.items.caused_by_injury_id | string_semantic | 1.000 | Yes | both_null | | | | | | |
| injury_loss_relations.items.loss | string_semantic | 1.000 | Yes | both_null | | | | | | |
| injury_loss_relations.items.evidence | array_llm | 1.000 | Yes | both_null | | | | | | |
| plaintiff.occupation_before | string_semantic | 0.900 | Yes |  | | | | | | |
| treatment.treatments_future | array_llm | 0.000 | No |  | 0 | 1 | 2 | 0.00 | 0.00 | 0.00 |
| losses.losses | string_semantic | 0.570 | No |  | | | | | | |
| treatment.treatments_received | array_llm | 0.875 | Yes |  | 7 | 1 | 0 | 1.00 | 0.88 | 0.93 |
| injuries.injuries | array_llm | 0.000 | No |  | 0 | 7 | 2 | 0.00 | 0.00 | 0.00 |
| psla.comparable_cases | array_llm | 0.500 | No |  | 1 | 1 | 4 | 0.20 | 0.50 | 0.29 |
| injuries.manifestations | array_llm | 0.308 | No |  | 4 | 9 | 6 | 0.40 | 0.31 | 0.35 |