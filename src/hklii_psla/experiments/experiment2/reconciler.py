import random
import time

from copy import deepcopy
from dataclasses import dataclass
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from hklii_psla.extractor.base import ExtractionMetadata, ExtractionResult, TokenCount
from hklii_psla.extractor.utils import extract_token_counts, get_sub_schema


from typing import Any


SYSTEM_PROMPT = (
    "You are an expert Data Reconciler. Your task is to merge, resolve, and reconcile "
    "extracted data items from multiple independent extractors into a single, definitive result.\n\n"

    "### Core Context\n"
    "- Input items from different extractors may arrive in any order; reconcile them by matching content, not position.\n"
    "- Extractors generally agree but may have conflicting fields, missing data, or unique context.\n\n"

    "### Reconciliation Rules\n"
    "1. Resolving Conflicts & Overlaps: If extractors conflict or overlap on a data point, select the highest-quality, most complete source. You must preserve this selected text VERBATIM without altering its original wording.\n"
    "2. Handling Unique Extractions: If an item appears in only one extractor, include it if it is accurate and relevant to the domain. Omit it if it looks like noise or an extraction error.\n"
    "3. Contextual Data: Retain additional context provided by an extractor only if it adds meaningful, relevant value to the final output. Otherwise, omit it.\n"
    "4. Text Modification: Do not summarize or paraphrase text unless explicitly instructed for a specific field. Lean toward verbatim accuracy for core data fields."
)

@dataclass(frozen=True)
class ReconciliationField:
    name: str
    dependencies: list[str]

@dataclass(frozen=True)
class ReconcilerConfig:
    fields: list[ReconciliationField]
    schema: type[BaseModel]
    additional_instructions: str = ""

class Reconciler:
    """
    Reconciler that compares and reconciles results from multiple extractors.

    Attributes:
        model: The chat model used for reconciliation.
        config: The configuration for the reconciler.
    """

    def __init__(self, model: BaseChatModel, config: ReconcilerConfig):
        self.model = model
        self.config = config
        self.schema = config.schema

    def reconcile(self, results: list[ExtractionResult]) -> ExtractionResult:

        # Randomly select a result to use as the base
        random_idx = random.randint(0, len(results) - 1)
        final_result: ExtractionResult = deepcopy(results[random_idx])
        reconcile_token_count = TokenCount()

        # Aggregate extraction-stage cost across all input runs
        extraction_token_count = sum(
            (r.token_count for r in results), TokenCount()
        )
        extraction_duration_ms = sum(r.duration_ms for r in results)

        t0 = time.perf_counter()

        for field in self.config.fields:
            sub_schema = get_sub_schema(field.name, self.schema)
            structured_llm = self.model.with_structured_output(sub_schema, include_raw=True)
            user_input = "Perform reconciliation on the following results:\n"
            if field.dependencies:
                dependencies: list[BaseModel] = [getattr(final_result.case, dep) for dep in field.dependencies]
                dependencies_json = [dep.model_dump_json() for dep in dependencies]
                user_input += (
                    f"You are to use the following dependencies as a "
                    f"reference and authoritative source for your reconciliation.\n"
                    f"Your reconciliation must not contradict the information in "
                    f"the dependencies.\n"
                    f"## Dependencies\n\n"
                    f"{'\n'.join(dependencies_json)}"
                    f"## End of Dependencies\n\n"
                )
            if self.config.additional_instructions:
                user_input += (
                    "Furthermore, below is some additional instructions for you "
                    "to follow.\n\n"
                    "## ADDITIONAL INSTRUCTIONS\n"
                    f"{self.config.additional_instructions}\n"
                    "## END OF ADDITIONAL INSTRUCTIONS\n\n"
                )
            for idx, result in enumerate(results):
                section: BaseModel = getattr(result.case, field.name)
                user_input += f"Result {idx + 1}:\n"
                user_input += section.model_dump_json() + "\n"
            call_result: Any = structured_llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_input)
            ])
            token_count = extract_token_counts(call_result)
            reconcile_token_count += token_count
            reconciliation = call_result.get("parsed")
            self.update_field(final_result, field.name, reconciliation)

        reconcile_duration_ms = (time.perf_counter() - t0) * 1000

        final_result.token_count = extraction_token_count + reconcile_token_count
        final_result.duration_ms = extraction_duration_ms + reconcile_duration_ms
        final_result.extraction_metadata = ExtractionMetadata(
            model_name=final_result.model_name,
            num_extraction_runs=len(results),
            extraction_token_count=extraction_token_count,
            reconciliation_token_count=reconcile_token_count,
            extraction_duration_ms=extraction_duration_ms,
            reconciliation_duration_ms=reconcile_duration_ms,
        )
        final_result.raw_response = None
        final_result.debug = None
        return final_result

    def update_field(self, result: ExtractionResult, field: str, reconciliation: BaseModel):
        update_section: BaseModel = getattr(result.case, field)

        if type(update_section) != type(reconciliation):
            raise RuntimeError(
                "Reconciliation type mismatch: " +
                f"{type(update_section)} != {type(reconciliation)}"
            )

        setattr(result.case, field, reconciliation)
