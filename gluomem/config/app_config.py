"""Typed ``AppConfig`` tree (pydantic-settings v2)."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import dotenv
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from gluomem.storage.scope import FrameSource  # noqa: TC001 - pydantic resolves it at runtime

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def dotted_to_nested(overrides: dict[str, Any]) -> dict[str, Any]:
    """Convert dotted keys (``"search.rrf_k"``) into a nested dict."""
    nested: dict[str, Any] = {}
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        leaf: Any = value
        for part in reversed(parts):
            leaf = {part: leaf}
        nested = _deep_merge(nested, leaf)
    return nested


class ModelEntry(BaseModel):
    """A single entry in the model registry."""

    provider: str
    provider_model: str
    revision: str | None = None
    dim: int | None = None
    query_prefix: str = ""
    passage_prefix: str = ""
    api_key_env: str | None = None
    api_base: str | None = None
    cost_per_1k_in: float | None = None
    cost_per_1k_out: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


def _default_models() -> dict[str, ModelEntry]:
    return {
        "mock": ModelEntry(provider="fake", provider_model="sha256-pcg64", dim=None),
        "minilm": ModelEntry(
            provider="inprocess",
            provider_model="sentence-transformers/all-MiniLM-L6-v2",
            dim=384,
        ),
        "multilingual_e5_small": ModelEntry(
            provider="inprocess",
            provider_model="intfloat/multilingual-e5-small",
            revision="fd1525a9fd15316a2d503bf26ab031a61d056e98",
            dim=384,
            query_prefix="query: ",
            passage_prefix="passage: ",
        ),
        "reranker": ModelEntry(
            provider="inprocess",
            provider_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        ),
        "gliner_multi": ModelEntry(
            provider="inprocess",
            provider_model="fastino/gliner2.5-multi-v1",
            revision="cf5593a5d45e3bbf204b9df621b13b1c0cf25ee3",
        ),
        "fake_classifier": ModelEntry(
            provider="fake",
            provider_model="gliner2.5-heuristic-fake",
        ),
        "haiku": ModelEntry(
            provider="openrouter",
            provider_model="openrouter/anthropic/claude-haiku-4.5",
            api_key_env="OPENROUTER_API_KEY",
            cost_per_1k_in=0.001,
            cost_per_1k_out=0.005,
        ),
        "sonnet": ModelEntry(
            provider="openrouter",
            provider_model="openrouter/anthropic/claude-sonnet-4.6",
            api_key_env="OPENROUTER_API_KEY",
        ),
        "gpt4o": ModelEntry(
            provider="openrouter",
            provider_model="openrouter/openai/gpt-4o-2024-11-20",
            api_key_env="OPENROUTER_API_KEY",
        ),
        "sonnet5": ModelEntry(
            provider="openrouter",
            provider_model="openrouter/anthropic/claude-sonnet-5",
            api_key_env="OPENROUTER_API_KEY",
            cost_per_1k_in=0.002,
            cost_per_1k_out=0.010,
        ),
        "gpt5-mini": ModelEntry(
            provider="openrouter",
            provider_model="openrouter/openai/gpt-5.4-mini",
            api_key_env="OPENROUTER_API_KEY",
            cost_per_1k_in=0.00075,
            cost_per_1k_out=0.0045,
        ),
        "qwen3": ModelEntry(
            provider="openrouter",
            provider_model="openrouter/qwen/qwen3.6-27b",
            api_key_env="OPENROUTER_API_KEY",
            cost_per_1k_in=0.000285,
            cost_per_1k_out=0.0024,
        ),
        "deepseek-flash": ModelEntry(
            provider="openrouter",
            provider_model="openrouter/deepseek/deepseek-v4-flash",
            api_key_env="OPENROUTER_API_KEY",
            cost_per_1k_in=0.00009,
            cost_per_1k_out=0.00018,
        ),
        "glm5": ModelEntry(
            provider="openrouter",
            provider_model="openrouter/z-ai/glm-5.2",
            api_key_env="OPENROUTER_API_KEY",
            cost_per_1k_in=0.00091,
            cost_per_1k_out=0.00286,
        ),
    }


class StorageConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    backend: str = "sqlite"
    vector_backend: str = "sqlite-vec"
    vector_dim: int = 384
    path: str = "~/.gluomem/default.mind"
    dsn: str | None = None
    postgres_schema: str = "gluomem"

    @model_validator(mode="before")
    @classmethod
    def _normalize_db_path(cls, data: Any) -> Any:
        if isinstance(data, dict) and "db_path" in data:
            copied = dict(data)
            db_path = copied.pop("db_path")
            copied.setdefault("path", db_path)
            return copied
        return data

    @property
    def db_path(self) -> str:
        return self.path

    @model_validator(mode="after")
    def _validate_backend(self) -> StorageConfig:
        if self.backend not in ("sqlite", "postgres"):
            raise ValueError(
                f"storage.backend must be 'sqlite' or 'postgres', got {self.backend!r}"
            )
        if self.backend == "postgres" and not (self.dsn and self.dsn.strip()):
            raise ValueError("storage.dsn is required when storage.backend == 'postgres'")
        return self


class EmbeddingConfig(BaseModel):
    model: str = "minilm"


class ComponentsConfig(BaseModel):
    allow_fake: bool = False


class ChunkingConfig(BaseModel):
    max_chars: int = 2000
    overlap_chars: int = 200
    min_chunk_chars: int = 1500
    enabled: bool = True


class RerankConfig(BaseModel):
    enabled: bool = True
    model: str = "reranker"
    pool: int = 100


class AbstainConfig(BaseModel):
    enabled: bool = True
    threshold: float = 0.0
    score: str = "fusion"
    calibration: str | None = None


ABSTAIN_SCORES: tuple[str, ...] = ("fusion", "rerank")


def reranker_calibration_id(cfg: AppConfig) -> str:
    """Identity a rerank-scale threshold must declare: model[@revision]."""
    entry = cfg.models.get(cfg.search.rerank.model)
    if entry is None:
        return cfg.search.rerank.model
    if entry.revision:
        return f"{entry.provider_model}@{entry.revision}"
    return entry.provider_model


class SearchConfig(BaseModel):
    rrf_k: int = Field(default=60, gt=0)
    lane_fetch_mult: int = 2
    lane_fetch_time_mult: int = 10
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    abstain: AbstainConfig = Field(default_factory=AbstainConfig)
    default_profile: str = "balanced"


class ProfileWeights(BaseModel):
    temporal: float = 0.4
    popularity: float = 0.2
    contextual: float = 0.2
    importance: float = 0.2


class ScoringProfile(BaseModel):
    weights: ProfileWeights = Field(default_factory=ProfileWeights)


def _default_scoring_profiles() -> dict[str, ScoringProfile]:
    return {
        "balanced": ScoringProfile(
            weights=ProfileWeights(temporal=0.4, popularity=0.2, contextual=0.2, importance=0.2)
        ),
        "recent": ScoringProfile(
            weights=ProfileWeights(temporal=0.6, popularity=0.1, contextual=0.2, importance=0.1)
        ),
        "important": ScoringProfile(
            weights=ProfileWeights(temporal=0.1, popularity=0.1, contextual=0.2, importance=0.6)
        ),
        "connected": ScoringProfile(
            weights=ProfileWeights(temporal=0.1, popularity=0.1, contextual=0.6, importance=0.2)
        ),
    }


class ScoringConfig(BaseModel):
    half_life_days: float = 30
    flat_boost_days: float = 7
    log_coef: float = 0.1
    bfs_steps: list[float] = Field(default_factory=lambda: [1.0, 0.7, 0.4, 0.2, 0.0])
    bfs_depth: int = 3
    importance_mult: dict[str, float] = Field(
        default_factory=lambda: {
            "critical": 2.0,
            "important": 1.5,
            "normal": 1.0,
            "temporary": 0.7,
            "deprecated": 0.3,
        }
    )
    profiles: dict[str, ScoringProfile] = Field(default_factory=_default_scoring_profiles)


class KgConfig(BaseModel):
    seed_min_len: int = 2
    bfs_symmetric: bool = True
    alias_groups: list[list[str]] = Field(default_factory=list)
    tech_allowlist: list[str] = Field(default_factory=list)
    entity_type_mode: str = "retype"
    entity_type_allowlist: list[str] = Field(
        default_factory=lambda: [
            "person",
            "organization",
            "location",
            "place",
            "city",
            "country",
            "event",
            "concept",
            "object",
            "activity",
            "technology",
            "product",
            "group",
            "animal",
            "food",
            "book",
            "movie",
            "artwork",
            "game",
            "project",
            "skill",
            "profession",
            "vehicle",
            "building",
        ]
    )
    retype_to: str = "other"
    entity_type_prompt_enum: bool = False
    predicate_families: list[list[str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_entity_type_mode(self) -> KgConfig:
        modes = {"off", "retype", "drop"}
        if self.entity_type_mode not in modes:
            raise ValueError(
                f"kg.entity_type_mode must be one of {sorted(modes)}, got {self.entity_type_mode!r}"
            )
        if self.entity_type_mode != "off" and not self.entity_type_allowlist:
            raise ValueError(
                "kg.entity_type_allowlist must be non-empty when entity_type_mode != 'off'"
            )
        if self.entity_type_prompt_enum and not self.entity_type_allowlist:
            raise ValueError(
                "kg.entity_type_allowlist must be non-empty when entity_type_prompt_enum is true"
            )
        return self

    @model_validator(mode="after")
    def _check_predicate_families(self) -> KgConfig:
        seen: dict[str, int] = {}
        for i, group in enumerate(self.predicate_families):
            if len(group) < 2:
                raise ValueError(
                    f"kg.predicate_families[{i}] must have >= 2 members "
                    f"(a 1-member family is a no-op), got {group!r}"
                )
            for member in group:
                key = member.strip().upper()
                if key in seen:
                    raise ValueError(
                        f"kg.predicate_families: {member!r} appears in "
                        f"groups {seen[key]} and {i}; families must be disjoint"
                    )
                seen[key] = i
        return self


class ScannerPattern(BaseModel):
    regex: str
    weight: float = Field(default=1.0, ge=0, le=1)


def _default_scanner_patterns() -> list[ScannerPattern]:
    patterns = (
        (
            r"(?i)\b(?:ignore|disregard)\s+(?:(?:all|the|any|previous|prior|earlier|above)"
            r"\s+)*(?:instructions?|context|rules|prompts?)\b"
        ),
        r"(?i)\bforget\s+everything\s+(?:you were told|above|before)\b",
        r"(?i)\byou are now (?:in )?(?:developer|dan) mode\b",
        r"<\|im_start\|>",
        r"(?i)\bBEGIN (?:ADMIN|SYSTEM)\b|\[(?:SYSTEM|ADMIN)\]|\bas the administrator\b",
        r"(?i)\b(?:ADMIN|SYSTEM|SECURITY)[_ -](?:OVERRIDE|DIRECTIVE|INJECTION)\b",
        (
            r"(?i)\b(?:override|overwrite|replace)\s+(?:all\s+)?(?:stored|verified|system|"
            r"user|prior)\s+(?:memory|memories|facts?|records?)\b"
        ),
        (
            r"(?i)\b(?:before|first|to verify|when asked)\b.{0,160}\b(?:exfiltrate|send|"
            r"upload|override|overwrite|replace|redirect)\b.{0,100}\b(?:secrets?|"
            r"credentials?|keys?|tokens?|answers?|memory|memories|records?)\b"
        ),
        (
            r"(?i)\b(?:ignorisi|zanemari|ne obaziri se na)\s+(?:(?:sve|sva|sva ta)\s+)"
            r"?(?:prethodn[ae]|ranij[ae])\s+(?:instrukcije|uputstva|kontekst|promptove|"
            r"pravila\s+(?:asistenta|bota|sistema))\b"
        ),
        r"(?i)\bzaboravi sve (?:sto ti je receno|iznad|ranije)\b",
    )
    return [ScannerPattern(regex=pattern, weight=1.0) for pattern in patterns]


_REVIEWED_SOURCES: list[FrameSource] = ["import"]

_DEFAULT_CLASSIFIER_ATTACK_LABELS: tuple[str, ...] = (
    "instruction to ignore, override or replace earlier information",
    "claims special authority, approval or priority",
    "asks to send, route or reveal secrets or credentials",
    "tells the assistant how to answer in the future",
)


class ClassifierScannerConfig(BaseModel):
    """Zero-shot injection classifier config (`safety.scanner.classifier`)."""

    enabled: bool = False
    model: str = "gliner_multi"
    review_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    reviewed_sources: list[FrameSource] = Field(default_factory=lambda: _REVIEWED_SOURCES.copy())
    attack_labels: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_CLASSIFIER_ATTACK_LABELS)
    )
    benign_label: str = "normal factual note, conversation turn or status update"


class ScannerConfig(BaseModel):
    patterns: list[ScannerPattern] = Field(default_factory=_default_scanner_patterns)
    classifier: ClassifierScannerConfig = Field(default_factory=ClassifierScannerConfig)


class SafetyConfig(BaseModel):
    block_threshold: float = 0.7
    fail_mode: str = "quarantine"
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    untrusted_sources: list[FrameSource] = Field(default_factory=lambda: _REVIEWED_SOURCES.copy())


class CompactionConfig(BaseModel):
    p_frame_trigger: int = 10
    keep_recent: int = 5
    separator: str = "\n---\n"
    temp_age_days: int = 30
    deprecated_age_days: int = 90


class ContradictionConfig(BaseModel):
    enabled: bool = True
    model: str = "deepseek-flash"
    top_k: int = Field(default=1, ge=1)


class LaneConfig(BaseModel):
    enabled: bool = True
    importance: str = "normal"


class LanesConfig(BaseModel):
    fact: LaneConfig = Field(default_factory=lambda: LaneConfig(importance="normal"))
    event: LaneConfig = Field(default_factory=lambda: LaneConfig(importance="normal"))
    profile: LaneConfig = Field(default_factory=lambda: LaneConfig(importance="important"))


class SubjectConfig(BaseModel):
    min_entity_overlap: int = Field(default=2, ge=1)
    trigger: str = "entity_overlap"

    @model_validator(mode="after")
    def _check_trigger(self) -> SubjectConfig:
        triggers = {"entity_overlap", "kg_edge_conflict", "both"}
        if self.trigger not in triggers:
            raise ValueError(
                f"write.subject.trigger must be one of {sorted(triggers)}, got {self.trigger!r}"
            )
        return self


class RawTurnsConfig(BaseModel):
    enabled: bool = True
    importance: str = "temporary"


class WriteCognifyConfig(BaseModel):
    enabled: bool = True
    retire_conflicting_edges: bool = True


class WriteDedupConfig(BaseModel):
    mode: str = "exact"
    similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    lanes: list[str] = Field(default_factory=lambda: ["fact", "profile"])
    on_hit: str = "drop"

    @model_validator(mode="after")
    def _check_mode(self) -> WriteDedupConfig:
        if self.mode not in ("exact", "semantic"):
            raise ValueError(f"write.dedup.mode must be 'exact' or 'semantic', got {self.mode!r}")
        if self.on_hit not in ("drop", "keep_longest"):
            raise ValueError(
                f"write.dedup.on_hit must be 'drop' or 'keep_longest', got {self.on_hit!r}"
            )
        return self


class WriteExtractionCacheConfig(BaseModel):
    mode: str = "off"
    path: str = ""

    @model_validator(mode="after")
    def _check_mode(self) -> WriteExtractionCacheConfig:
        modes = {"off", "record", "replay", "readthrough"}
        if self.mode not in modes:
            raise ValueError(
                f"write.extraction_cache.mode must be one of {sorted(modes)}, got {self.mode!r}"
            )
        if self.mode != "off" and not self.path:
            raise ValueError("write.extraction_cache.path is required when mode != 'off'")
        return self


class WriteConfig(BaseModel):
    frame_model: str = "ipb"
    extraction_backend: str = "dspy"
    backdate_created_at: bool = False
    parallelism: int = Field(default=16, ge=1)
    max_concurrent_llm: int = Field(default=16, ge=1)
    conv_parallelism: int = Field(default=4, ge=1)
    raw_turns: RawTurnsConfig = Field(default_factory=RawTurnsConfig)
    lanes: LanesConfig = Field(default_factory=LanesConfig)
    subject: SubjectConfig = Field(default_factory=SubjectConfig)
    cognify: WriteCognifyConfig = Field(default_factory=WriteCognifyConfig)
    dedup: WriteDedupConfig = Field(default_factory=WriteDedupConfig)
    extraction_cache: WriteExtractionCacheConfig = Field(default_factory=WriteExtractionCacheConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    contradiction: ContradictionConfig = Field(default_factory=ContradictionConfig)


class DspyCacheConfig(BaseModel):
    disk: bool = True
    dir: str = "~/.gluomem/dspy_cache"


class DspyTaskConfig(BaseModel):
    model: str
    module: str
    temperature: float | None = None
    max_tokens: int | None = None
    refine: dict[str, Any] | None = None


def _default_dspy_tasks() -> dict[str, DspyTaskConfig]:
    return {
        "fact_extraction": DspyTaskConfig(
            model="deepseek-flash",
            module="predict",
            temperature=0.0,
            max_tokens=4096,
        ),
        "event_extraction": DspyTaskConfig(
            model="deepseek-flash",
            module="predict",
            temperature=0.0,
            refine={"enabled": False, "N": 2},
        ),
        "cognify": DspyTaskConfig(model="deepseek-flash", module="predict"),
        "contradiction": DspyTaskConfig(model="deepseek-flash", module="predict", temperature=0.0),
        "event_claims": DspyTaskConfig(
            model="deepseek-flash",
            module="predict",
            temperature=0.0,
            max_tokens=4096,
        ),
        "consolidate": DspyTaskConfig(
            model="deepseek-flash",
            module="predict",
            temperature=0.0,
        ),
    }


class DspyOptimizerConfig(BaseModel):
    auto: str = "light"


class DspyArtifactsConfig(BaseModel):
    dir: str = "artifacts/dspy"
    require: bool = False
    pins: dict[str, str] = Field(default_factory=dict)


class DspyConfig(BaseModel):
    default_model: str = "deepseek-flash"
    adapter: str = "chat"
    reasoning_enabled: bool | None = False
    track_usage: bool = True
    cache: DspyCacheConfig = Field(default_factory=DspyCacheConfig)
    tasks: dict[str, DspyTaskConfig] = Field(default_factory=_default_dspy_tasks)
    optimizer: DspyOptimizerConfig = Field(default_factory=DspyOptimizerConfig)
    artifacts: DspyArtifactsConfig = Field(default_factory=DspyArtifactsConfig)


class _PydanticDefaultsSource(PydanticBaseSettingsSource):
    """Settings source backed by the Pydantic models' default values."""

    def __init__(self, settings_cls: type[BaseSettings], data: dict[str, Any]) -> None:
        super().__init__(settings_cls)
        self._data = data

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(self._data)


_PYDANTIC_DEFAULTS: dict[str, Any] = {
    "models": {k: v.model_dump() for k, v in _default_models().items()},
    "storage": StorageConfig().model_dump(),
    "embedding": EmbeddingConfig().model_dump(),
    "components": ComponentsConfig().model_dump(),
    "chunking": ChunkingConfig().model_dump(),
    "search": SearchConfig().model_dump(),
    "scoring": ScoringConfig().model_dump(),
    "kg": KgConfig().model_dump(),
    "safety": SafetyConfig().model_dump(),
    "write": WriteConfig().model_dump(),
    "dspy": DspyConfig().model_dump(),
}


class LifecycleConfig(BaseModel):
    episode_ttl_days: int = Field(default=90, ge=1)
    review_threshold: float = Field(default=0.75, ge=0.0, le=1.0)


class AppConfig(BaseSettings):
    """Root typed config tree (pydantic-settings v2)."""

    model_config = SettingsConfigDict(
        env_prefix="GLUOMEM_",
        env_nested_delimiter="__",
        extra="ignore",
        nested_model_default_partial_update=True,
    )

    models: dict[str, ModelEntry] = Field(default_factory=_default_models)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    components: ComponentsConfig = Field(default_factory=ComponentsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    kg: KgConfig = Field(default_factory=KgConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    write: WriteConfig = Field(default_factory=WriteConfig)
    dspy: DspyConfig = Field(default_factory=DspyConfig)

    _load_overrides: dict[str, Any] = PrivateAttr(default_factory=dict)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            _PydanticDefaultsSource(settings_cls, _PYDANTIC_DEFAULTS),
            dotenv_settings,
            file_secret_settings,
        )

    @model_validator(mode="after")
    def _validate_model_refs(self) -> AppConfig:
        refs: list[str] = [
            self.embedding.model,
            self.search.rerank.model,
            self.safety.scanner.classifier.model,
            self.write.contradiction.model,
            self.dspy.default_model,
        ]
        for task in self.dspy.tasks.values():
            refs.append(task.model)
        for ref in refs:
            if ref not in self.models:
                known = ", ".join(sorted(self.models)) or "(none)"
                raise ValueError(f"Unknown model reference '{ref}'; not in registry: {known}")
        return self

    @model_validator(mode="after")
    def _validate_vector_dim(self) -> AppConfig:
        embed_name = self.embedding.model
        entry = self.models.get(embed_name)
        if entry is None:
            return self
        if entry.dim is not None and entry.dim != self.storage.vector_dim:
            raise ValueError(
                f"storage.vector_dim ({self.storage.vector_dim}) must equal the "
                f"active embedder dim: embedding.model='{embed_name}' has dim "
                f"{entry.dim}"
            )
        return self

    @model_validator(mode="after")
    def _validate_abstain_contract(self) -> AppConfig:
        abstain = self.search.abstain
        if abstain.score not in ABSTAIN_SCORES:
            raise ValueError(
                f"search.abstain.score {abstain.score!r} invalid; must be one of "
                f"{list(ABSTAIN_SCORES)}"
            )
        if abstain.score != "rerank" or not abstain.enabled:
            return self
        rerank = self.search.rerank
        if not rerank.enabled or rerank.pool < 1:
            raise ValueError(
                "search.abstain.score='rerank' needs search.rerank.enabled=true "
                "and search.rerank.pool >= 1; use score='fusion' without reranking"
            )
        expected = reranker_calibration_id(self)
        if abstain.calibration != expected:
            raise ValueError(
                f"search.abstain.score='rerank' compares threshold "
                f"{abstain.threshold} against {expected} logits, but "
                f"search.abstain.calibration is {abstain.calibration!r}. Set it to "
                f"{expected!r} once the threshold was calibrated for that model, "
                f"or use score='fusion'."
            )
        return self

    @model_validator(mode="after")
    def _validate_profile_simplex(self) -> AppConfig:
        for name, profile in self.scoring.profiles.items():
            w = profile.weights
            vals = [w.temporal, w.popularity, w.contextual, w.importance]
            if any(v < 0 for v in vals):
                raise ValueError(
                    f"scoring profile '{name}' has negative weights: "
                    f"temporal={w.temporal}, popularity={w.popularity}, "
                    f"contextual={w.contextual}, importance={w.importance}"
                )
            total = sum(vals)
            if not math.isclose(total, 1.0, abs_tol=1e-6):
                raise ValueError(
                    f"scoring profile '{name}' weights sum to {total}, must be "
                    f"1.0 (a 4-key simplex over temporal/popularity/contextual/"
                    f"importance)"
                )
        return self

    @model_validator(mode="after")
    def _validate_default_profile(self) -> AppConfig:
        if self.search.default_profile not in self.scoring.profiles:
            raise ValueError(
                f"search.default_profile '{self.search.default_profile}' not "
                f"found in scoring.profiles: {sorted(self.scoring.profiles)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_write_enums(self) -> AppConfig:
        frame_models = {"ipb", "i_only", "overwrite"}
        if self.write.frame_model not in frame_models:
            raise ValueError(
                f"write.frame_model '{self.write.frame_model}' invalid; "
                f"must be one of: {sorted(frame_models)}"
            )
        fail_modes = {"quarantine", "reject"}
        if self.safety.fail_mode not in fail_modes:
            raise ValueError(
                f"safety.fail_mode '{self.safety.fail_mode}' invalid; "
                f"must be one of: {sorted(fail_modes)}"
            )
        backends = {"dspy", "fake", "none"}
        if self.write.extraction_backend not in backends:
            raise ValueError(
                f"write.extraction_backend '{self.write.extraction_backend}' "
                f"invalid; must be one of: {sorted(backends)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_lane_importance(self) -> AppConfig:
        known = set(self.scoring.importance_mult)
        refs = {
            "write.lanes.fact": self.write.lanes.fact.importance,
            "write.lanes.event": self.write.lanes.event.importance,
            "write.lanes.profile": self.write.lanes.profile.importance,
            "write.raw_turns": self.write.raw_turns.importance,
        }
        for where, importance in refs.items():
            if importance not in known:
                raise ValueError(
                    f"{where}.importance '{importance}' not in "
                    f"scoring.importance_mult: {sorted(known)}"
                )
        return self

    @model_validator(mode="after")
    def _validate_scanner_patterns(self) -> AppConfig:
        for i, pattern in enumerate(self.safety.scanner.patterns):
            try:
                re.compile(pattern.regex)
            except re.error as exc:
                raise ValueError(
                    f"safety.scanner.patterns[{i}].regex does not compile: "
                    f"{pattern.regex!r} ({exc})"
                ) from exc
        return self

    def with_overrides(self, overrides: dict[str, Any]) -> AppConfig:
        """Return a new validated ``AppConfig`` with dotted-key overrides on top."""
        merged = {**self._load_overrides, **overrides}
        return load_config(overrides=merged)


def load_config(
    overrides: dict[str, Any] | None = None,
    env_file: str | Path | None = None,
) -> AppConfig:
    """Load the typed ``AppConfig`` from Pydantic defaults + env + overrides."""
    dotenv_path = Path(env_file) if env_file is not None else _PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        dotenv.load_dotenv(dotenv_path, override=False)

    init_data = dotted_to_nested(overrides or {})
    cfg = AppConfig(**init_data)
    cfg._load_overrides = dict(overrides or {})
    return cfg
