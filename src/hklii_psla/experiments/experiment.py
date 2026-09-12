from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, cast

import yaml
from pydantic import BaseModel, ValidationError

from hklii_psla.extractor.base import ExtractionResult


@dataclass
class ExperimentRun:
    """Record of a single extraction run."""

    provider: str
    model: str
    strategy: str  # "single-pass", "section-by-section", "multi-agent"
    prompt_style: str  # "zero-shot" or "few-shot"
    case_file: str
    result: ExtractionResult | None = None
    filename: str | None = None  # persisted filename for one-to-one mapping

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "strategy": self.strategy,
            "prompt_style": self.prompt_style,
            "case_file": self.case_file,
        }
        if self.result:
            d.update({
                "success": self.result.success,
                "duration_ms": self.result.duration_ms,
                "token_count": self.result.token_count.to_dict(),
                "extraction_metadata": (
                    self.result.extraction_metadata.to_dict()
                    if self.result.extraction_metadata
                    else None
                ),
                "error": self.result.error,
                "case_json": self.result.case.model_dump(mode="json") if self.result.case else None,
            })
        return d


@dataclass
class ExperimentBatch:
    """Collection of runs for a single experiment."""

    experiment_id: str
    runs: list[ExperimentRun] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        total = len(self.runs)
        successful = sum(1 for r in self.runs if r.result and r.result.success)
        total_duration = sum(r.result.duration_ms for r in self.runs if r.result)
        total_tokens = sum(r.result.token_count.total_tokens for r in self.runs if r.result)
        return {
            "experiment_id": self.experiment_id,
            "total_runs": total,
            "successful": successful,
            "failed": total - successful,
            "total_duration_ms": total_duration,
            "total_tokens": total_tokens,
        }



class BaseExperiment[ConfigT: BaseModel](ABC):

    _config: ConfigT
    _config_model: ClassVar[type[BaseModel]]
    _experiment_id: ClassVar[str]

    registry: ClassVar[dict[str, type["BaseExperiment[Any]"]]] = {}

    def __init__(self, config_path: Path) -> None:
        self._config = self._validate_config(config_path)

    def __init_subclass__(cls, experiment_id: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registry[experiment_id] = cls
        cls._experiment_id = experiment_id

    @classmethod
    def _validate_config(cls, config_path: Path) -> ConfigT:
        try:
            with config_path.open(mode="r", encoding="utf-8") as f:
                config = cast(dict[str, Any], yaml.safe_load(f))
                config = cls._config_model.model_validate(config)
                return cast(ConfigT, config)

        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path.name}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing config file: {e}")
        except ValidationError:
            raise ValidationError("Error validating config. Please check the correctness of the config file.")
        except Exception as e:
            raise e

    @abstractmethod
    def run(self) -> None:
        ...

    @abstractmethod
    def save_results(self) -> None:
        ...
