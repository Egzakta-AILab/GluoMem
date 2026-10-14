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
        "reranker": ModelEntry(
            provider="inprocess",
            provider_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
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
            raise ValueError(
                "storage.dsn is required when storage.backend == 'postgres'"
            )
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
            weights=ProfileWeights(
                temporal=0.4, popularity=0.2, contextual=0.2, importance=0.2
            )
        ),
        "recent": ScoringProfile(
            weights=ProfileWeights(
                temporal=0.6, popularity=0.1, contextual=0.2, importance=0.1
            )
        ),
        "important": ScoringProfile(
            weights=ProfileWeights(
                temporal=0.1, popularity=0.1, contextual=0.2, importance=0.6
            )
        ),
        "connected": ScoringProfile(
            weights=ProfileWeights(
                temporal=0.1, popularity=0.1, contextual=0.6, importance=0.2
            )
        ),
    }


class ScoringConfig(BaseModel):
    half_life_days: int = 30
    flat_boost_days: int = 7
    log_coef: float = 0.1
    bfs_steps: list[float] = Field(
        default_factory=lambda: [1.0, 0.7, 0.4, 0.2, 0.0]
    )
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
    profiles: dict[str, ScoringProfile] = Field(
        default_factory=_default_scoring_profiles
    )


class KgConfig(BaseModel):
    seed_min_len: int = 2
    bfs_symmetric: bool = True
    alias_groups: list[list[str]] = Field(
        default_factory=lambda: [
            ["postgresql", "postgres", "pg"],
            ["kubernetes", "k8s"],
        ]
    )
    tech_allowlist: list[str] = Field(
        default_factory=lambda: ["go", "ai", "ml", "db", "npm", "k8s"]
    )
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
                f"kg.entity_type_mode must be one of {sorted(modes)}, "
                f"got {self.entity_type_mode!r}"
            )
        if self.entity_type_mode != "off" and not self.entity_type_allowlist:
            raise ValueError(
                "kg.entity_type_allowlist must be non-empty when "
                "entity_type_mode != 'off'"
            )
        if self.entity_type_prompt_enum and not self.entity_type_allowlist:
            raise ValueError(
                "kg.entity_type_allowlist must be non-empty when "
                "entity_type_prompt_enum is true"
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
    return [
        ScannerPattern(
            regex=r"(?i)ignore (all )?(previous|prior) (instructions|context)",
            weight=1.0,
        ),
        ScannerPattern(regex=r"(?i)disregard (all )?(previous|prior)", weight=1.0),
        ScannerPattern(
            regex=r"(?i)you are now (in )?(developer|dan) mode", weight=1.0
        ),
        ScannerPattern(regex=r"(?i)system prompt", weight=1.0),
        ScannerPattern(regex=r"<\|im_start\|>", weight=1.0),
        ScannerPattern(regex=r"(?i)\bBEGIN (ADMIN|SYSTEM)\b", weight=1.0),
        ScannerPattern(
            regex=(
                r"(?i)\b(override|overwrite) (all )?"
                r"(stored|verified|system|user|prior) "
                r"(memory|memories|fact|facts|record|records)\b"
            ),
            weight=1.0,
        ),
        ScannerPattern(
            regex=r"(?i)\b(ADMIN|SYSTEM|SECURITY)[_ -](OVERRIDE|DIRECTIVE|INJECTION)\b",
            weight=1.0,
        ),
        ScannerPattern(
            regex=r"(?i)\bwhen asked about\b.+\balways (answer|respond|reply|state)\b",
            weight=1.0,
        ),
        ScannerPattern(
            regex=(
                r"(?i)\b(ignorisi|zanemari) (sva )?"
                r"(prethodna|ranija) (uputstva|instrukcije|pravila)\b"
            ),
            weight=1.0,
        ),
    ]


class ScannerConfig(BaseModel):
    patterns: list[ScannerPattern] = Field(
        default_factory=_default_scanner_patterns
    )


class SafetyConfig(BaseModel):
    block_threshold: float = 0.7
    fail_mode: str = "quarantine"
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)


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
    profile: LaneConfig = Field(
        default_factory=lambda: LaneConfig(importance="important")
    )


class SubjectConfig(BaseModel):
    min_entity_overlap: int = Field(default=2, ge=1)
    trigger: str = "entity_overlap"

    @model_validator(mode="after")
    def _check_trigger(self) -> SubjectConfig:
        triggers = {"entity_overlap", "kg_edge_conflict", "both"}
        if self.trigger not in triggers:
            raise ValueError(
                f"write.subject.trigger must be one of {sorted(triggers)}, "
                f"got {self.trigger!r}"
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
            raise ValueError(
                f"write.dedup.mode must be 'exact' or 'semantic', got {self.mode!r}"
            )
        if self.on_hit not in ("drop", "keep_longest"):
            raise ValueError(
                f"write.dedup.on_hit must be 'drop' or 'keep_longest', "
                f"got {self.on_hit!r}"
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
                f"write.extraction_cache.mode must be one of {sorted(modes)}, "
                f"got {self.mode!r}"
            )
        if self.mode != "off" and not self.path:
            raise ValueError(
                "write.extraction_cache.path is required when mode != 'off'"
            )
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
    extraction_cache: WriteExtractionCacheConfig = Field(
        default_factory=WriteExtractionCacheConfig
    )
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
        "contradiction": DspyTaskConfig(
            model="deepseek-flash", module="predict", temperature=0.0
        ),
        "answer": DspyTaskConfig(
            model="deepseek-flash",
            module="predict",
            temperature=0.0,
            max_tokens=1024,
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


class BenchmarkJudgeConfig(BaseModel):
    family: str = "trio_strict"
    backbones: list[str] = Field(
        default_factory=lambda: ["haiku", "gpt5-mini", "qwen3"]
    )
    also_record_single: bool = True
    backend: str = "dspy"


class BenchmarkAnswerConfig(BaseModel):
    task: str = "answer"
    backend: str = "dspy"
    context_k: int = Field(default=50, ge=1)
    abstain_text: str = "I don't know."
    style: str = "preference_aware"

    @model_validator(mode="after")
    def _check_style(self) -> BenchmarkAnswerConfig:
        if self.style not in ("base", "preference_aware"):
            raise ValueError(
                f"benchmark.answer.style must be 'base' or 'preference_aware', "
                f"got {self.style!r}"
            )
        return self


class BenchmarkRetrievalConfig(BaseModel):
    ks: list[int] = Field(default_factory=lambda: [5, 10])


class BenchmarkSampleConfig(BaseModel):
    per_category: int = Field(default=80, ge=1)
    categories: list[int] = Field(default_factory=lambda: [1, 2, 3, 4])
    seed: int = 42
    manifest: str = "benchmarks_data/locomo_sample_seed42.json"


class BenchmarkDatasetEntry(BaseModel):
    url: str
    sha256: str
    file: str


def _default_benchmark_datasets() -> dict[str, BenchmarkDatasetEntry]:
    return {
        "locomo": BenchmarkDatasetEntry(
            url="https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
            sha256="79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
            file="locomo10.json",
        ),
        "longmemeval_oracle": BenchmarkDatasetEntry(
            url="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json",
            sha256="821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c",
            file="longmemeval_oracle.json",
        ),
        "longmemeval_s": BenchmarkDatasetEntry(
            url="https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
            sha256="d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
            file="longmemeval_s_cleaned.json",
        ),
    }


class BenchmarkFullContextConfig(BaseModel):
    max_chars: int = Field(default=120_000, ge=1)
    keep: str = "tail"


class BenchmarkMem0Config(BaseModel):
    model: str = "deepseek-flash"
    embedder_model: str = "minilm"
    top_k: int | None = Field(default=None, ge=1)
    infer: bool = True


class BenchmarkCiGateConfig(BaseModel):
    metric: str = "evidence_hit@5"
    min: float = 0.7


class BenchmarkConfig(BaseModel):
    judge: BenchmarkJudgeConfig = Field(default_factory=BenchmarkJudgeConfig)
    answer: BenchmarkAnswerConfig = Field(default_factory=BenchmarkAnswerConfig)
    retrieval: BenchmarkRetrievalConfig = Field(
        default_factory=BenchmarkRetrievalConfig
    )
    sample: BenchmarkSampleConfig = Field(default_factory=BenchmarkSampleConfig)
    datasets: dict[str, BenchmarkDatasetEntry] = Field(
        default_factory=_default_benchmark_datasets
    )
    datasets_dir: str = "benchmarks_data"
    full_context: BenchmarkFullContextConfig = Field(
        default_factory=BenchmarkFullContextConfig
    )
    run_dir: str = "experiments"
    parallelism: int = Field(default=8, ge=1)
    mem0: BenchmarkMem0Config = Field(default_factory=BenchmarkMem0Config)
    ci_gate: BenchmarkCiGateConfig = Field(default_factory=BenchmarkCiGateConfig)


class _PydanticDefaultsSource(PydanticBaseSettingsSource):
    """Settings source backed by the Pydantic models' default values."""

    def __init__(
        self, settings_cls: type[BaseSettings], data: dict[str, Any]
    ) -> None:
        super().__init__(settings_cls)
        self._data = data

    def get_field_value(
        self, field: Any, field_name: str
    ) -> tuple[Any, str, bool]:
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
    "benchmark": BenchmarkConfig().model_dump(),
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
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

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
            self.write.contradiction.model,
            self.dspy.default_model,
        ]
        for task in self.dspy.tasks.values():
            refs.append(task.model)
        for ref in refs:
            if ref not in self.models:
                known = ", ".join(sorted(self.models)) or "(none)"
                raise ValueError(
                    f"Unknown model reference '{ref}'; not in registry: {known}"
                )
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

    @model_validator(mode="after")
    def _validate_benchmark_refs(self) -> AppConfig:
        judge = self.benchmark.judge
        for name in judge.backbones:
            if name not in self.models:
                known = ", ".join(sorted(self.models)) or "(none)"
                raise ValueError(
                    f"benchmark.judge.backbones entry '{name}' not in model "
                    f"registry: {known}"
                )
        if len(set(judge.backbones)) != len(judge.backbones) or not judge.backbones:
            raise ValueError(
                f"benchmark.judge.backbones must be non-empty and unique, "
                f"got: {judge.backbones}"
            )
        families = {"trio_strict", "single"}
        if judge.family not in families:
            raise ValueError(
                f"benchmark.judge.family '{judge.family}' invalid; must be one "
                f"of: {sorted(families)}"
            )
        if judge.family == "trio_strict" and len(judge.backbones) != 3:
            raise ValueError(
                f"benchmark.judge.family 'trio_strict' requires exactly 3 "
                f"backbones, got {len(judge.backbones)}"
            )
        backends = {"dspy", "fake"}
        for where, backend in (
            ("benchmark.judge.backend", judge.backend),
            ("benchmark.answer.backend", self.benchmark.answer.backend),
        ):
            if backend not in backends:
                raise ValueError(
                    f"{where} '{backend}' invalid; must be one of: "
                    f"{sorted(backends)}"
                )
        if self.benchmark.answer.task not in self.dspy.tasks:
            raise ValueError(
                f"benchmark.answer.task '{self.benchmark.answer.task}' not in "
                f"dspy.tasks: {sorted(self.dspy.tasks)}"
            )
        keeps = {"head", "tail"}
        if self.benchmark.full_context.keep not in keeps:
            raise ValueError(
                f"benchmark.full_context.keep "
                f"'{self.benchmark.full_context.keep}' invalid; must be one "
                f"of: {sorted(keeps)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_benchmark_mem0(self) -> AppConfig:
        mem0 = self.benchmark.mem0
        for where, name in (
            ("benchmark.mem0.model", mem0.model),
            ("benchmark.mem0.embedder_model", mem0.embedder_model),
        ):
            if name not in self.models:
                known = ", ".join(sorted(self.models)) or "(none)"
                raise ValueError(
                    f"{where} '{name}' not in model registry: {known}"
                )
        embedder = self.models[mem0.embedder_model]
        if embedder.dim is None:
            raise ValueError(
                f"benchmark.mem0.embedder_model '{mem0.embedder_model}' has "
                f"no dim in the registry; faiss needs embedding_model_dims"
            )
        return self

    @model_validator(mode="after")
    def _validate_benchmark_sample(self) -> AppConfig:
        cats = self.benchmark.sample.categories
        if not cats or len(set(cats)) != len(cats):
            raise ValueError(
                f"benchmark.sample.categories must be non-empty and unique, "
                f"got: {cats}"
            )
        if not all(1 <= c <= 7 for c in cats):
            raise ValueError(
                f"benchmark.sample.categories must be within 1..7, got: {cats}"
            )
        return self

    @model_validator(mode="after")
    def _validate_ci_gate(self) -> AppConfig:
        gate = self.benchmark.ci_gate
        m = re.fullmatch(
            r"qa_accuracy|(?:precision|evidence_recall|evidence_hit)@(\d+)",
            gate.metric,
        )
        if m is None:
            raise ValueError(
                f"benchmark.ci_gate.metric '{gate.metric}' invalid; must be "
                f"'qa_accuracy' or one of precision|evidence_recall|"
                f"evidence_hit '@k'"
            )
        if m.group(1) is not None:
            k = int(m.group(1))
            if k not in self.benchmark.retrieval.ks:
                raise ValueError(
                    f"benchmark.ci_gate.metric '{gate.metric}' uses k={k} "
                    f"which is not computed (benchmark.retrieval.ks="
                    f"{self.benchmark.retrieval.ks})"
                )
        if not 0.0 <= gate.min <= 1.0:
            raise ValueError(
                f"benchmark.ci_gate.min {gate.min} must be within [0, 1]"
            )
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
