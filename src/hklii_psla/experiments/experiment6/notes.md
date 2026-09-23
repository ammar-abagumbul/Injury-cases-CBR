## Major BUMMMMM
* We forgot to include Burns for the ICD-codes *


We have so far performed an extraction run on a corpus of legal documents (approx. 1400).
The extractions are according to the schema that you can find in **schema.py**.

However, the extraction remains imperfect at times due to the following reasons.

1. Some extractions report null values even though the features can be properly inferred/extracted from the legal documents. Whilst difficult to certainly identify which null fields are true negatives, we can use some heuristics to vet out the extractions that need to be redone. 
  a. We can specify certain fields which act as major red flags. For example, a null field for Gender indicates that the model could do better.
  b. We can measure the proportion of fields that are null. An overwelmingly null dominated extraction should raise a red flag. 

2. Some extractions fail to correctly find the correct neutral citation. For our use case, we should always assume that the neutral citation is inlcuded in the case. A simple regex change can identify these defects.

3. Some injuries have null icd codes. This should be treated differently from the case stated in 1. A null icd code indicates that the injury extracted by the model in fact is not an injury (on its own). But rather, a symptom, after effect, manifestation, etc ... 

4. Some injuries have nodes that pack in a lot of body parts at once on some nodes. A major example of that is the abdomen, pelvic, lumbar, etc ... We need to filter out some of these fields so that we further process them and force the LLM into a decision and go the tree at least down a level.

5.
