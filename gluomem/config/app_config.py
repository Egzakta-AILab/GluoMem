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

from gluomem.config.variants import (
    deep_merge,
    dotted_to_nested,
    merge_variant,
    read_default_config,
    read_yaml,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("default_config.yaml")
_VARIANTS_DIR = _PROJECT_ROOT / "conf" / "variants"


def _section_default(section: str, key: str) -> Any:
    return read_default_config()[section][key]


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


class ScoringConfig(BaseModel):
    half_life_days: int = 30
    flat_boost_days: int = 7
    log_coef: float = 0.1
    bfs_steps: list[float] = Field(
        default_factory=lambda: list(_section_default("scoring", "bfs_steps"))
    )
    bfs_depth: int = 3
    importance_mult: dict[str, float] = Field(
        default_factory=lambda: dict(_section_default("scoring", "importance_mult"))
    )
    profiles: dict[str, ScoringProfile] = Field(
        default_factory=lambda: {
            k: ScoringProfile.model_validate(v)
            for k, v in _section_default("scoring", "profiles").items()
        }
    )


class KgConfig(BaseModel):
    seed_min_len: int = 2
    bfs_symmetric: bool = True
    alias_groups: list[list[str]] = Field(
        default_factory=lambda: list(_section_default("kg", "alias_groups"))
    )
    tech_allowlist: list[str] = Field(
        default_factory=lambda: list(_section_default("kg", "tech_allowlist"))
    )
    entity_type_mode: str = "retype"
    entity_type_allowlist: list[str] = Field(
        default_factory=lambda: list(_section_default("kg", "entity_type_allowlist"))
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


class ScannerConfig(BaseModel):
    patterns: list[ScannerPattern] = Field(
        default_factory=lambda: [
            ScannerPattern.model_validate(p)
            for p in _section_default("safety", "scanner")["patterns"]
        ]
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
    tasks: dict[str, DspyTaskConfig] = Field(
        default_factory=lambda: {
            k: DspyTaskConfig.model_validate(v)
            for k, v in _section_default("dspy", "tasks").items()
        }
    )
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
        default_factory=lambda: {
            k: BenchmarkDatasetEntry.model_validate(v)
            for k, v in _section_default("benchmark", "datasets").items()
        }
    )
    datasets_dir: str = "benchmarks_data"
    full_context: BenchmarkFullContextConfig = Field(
        default_factory=BenchmarkFullContextConfig
    )
    run_dir: str = "experiments"
    parallelism: int = Field(default=8, ge=1)
    mem0: BenchmarkMem0Config = Field(default_factory=BenchmarkMem0Config)
    ci_gate: BenchmarkCiGateConfig = Field(default_factory=BenchmarkCiGateConfig)


class _DictSettingsSource(PydanticBaseSettingsSource):
    """Settings source backed by a plain dict (used for the file/variant layer)."""

    def __init__(self, settings_cls: type[BaseSettings], data: dict[str, Any]) -> None:
        super().__init__(settings_cls)
        self._data = data

    def get_field_value(
        self, field: Any, field_name: str
    ) -> tuple[Any, str, bool]:
        value = self._data.get(field_name)
        return value, field_name, value is not None

    def prepare_field_value(
        self,
        field_name: str,
        field: Any,
        value: Any,
        value_is_complex: bool,  # noqa: ARG002
    ) -> Any:
        return value

    def __call__(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for field_name in self.settings_cls.model_fields:
            value = self._data.get(field_name)
            if value is not None:
                out[field_name] = value
        return out


class AppConfig(BaseSettings):
    """Root typed config tree (pydantic-settings v2)."""

    model_config = SettingsConfigDict(
        env_prefix="GLUOMEM_",
        env_nested_delimiter="__",
        extra="ignore",
        nested_model_default_partial_update=True,
    )

    models: dict[str, ModelEntry] = Field(
        default_factory=lambda: {
            k: ModelEntry.model_validate(v)
            for k, v in read_default_config()["models"].items()
        }
    )
    storage: StorageConfig = Field(default_factory=StorageConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    components: ComponentsConfig = Field(default_factory=ComponentsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    kg: KgConfig = Field(default_factory=KgConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    write: WriteConfig = Field(default_factory=WriteConfig)
    dspy: DspyConfig = Field(default_factory=DspyConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    _load_path: Path | None = PrivateAttr(default=None)
    _load_variant: str | None = PrivateAttr(default=None)
    _load_variants_dir: Path | None = PrivateAttr(default=None)
    _load_overrides: dict[str, Any] = PrivateAttr(default_factory=dict)

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
        return load_config(
            path=self._load_path,
            variant=self._load_variant,
            variants_dir=self._load_variants_dir,
            overrides=merged,
        )


def load_config(
    path: str | Path | None = None,
    variant: str | None = None,
    overrides: dict[str, Any] | None = None,
    variants_dir: str | Path | None = None,
    env_file: str | Path | None = None,
) -> AppConfig:
    """Load the typed ``AppConfig`` from the file layer + env + overrides."""
    dotenv_path = Path(env_file) if env_file is not None else _PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        dotenv.load_dotenv(dotenv_path, override=False)

    yaml_path = Path(path) if path is not None else None
    file_layer = read_default_config()
    if yaml_path is not None:
        file_layer = deep_merge(file_layer, read_yaml(yaml_path))

    vdir = Path(variants_dir) if variants_dir is not None else _VARIANTS_DIR
    if variant is not None:
        file_layer = merge_variant(file_layer, variant, vdir)

    init_data = dotted_to_nested(overrides or {})
    file_layer_final = file_layer

    class _Cfg(AppConfig):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            sources: list[PydanticBaseSettingsSource] = [
                init_settings,
                env_settings,
            ]
            if file_layer_final:
                sources.append(_DictSettingsSource(settings_cls, file_layer_final))
            sources.append(dotenv_settings)
            sources.append(file_secret_settings)
            return tuple(sources)

    cfg = _Cfg(**init_data)
    cfg._load_path = yaml_path
    cfg._load_variant = variant
    cfg._load_variants_dir = vdir
    cfg._load_overrides = dict(overrides or {})
    return cfg
