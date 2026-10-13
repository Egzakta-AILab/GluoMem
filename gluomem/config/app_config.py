"""gluomem.config.app_config — typed AppConfig tree (pydantic-settings v2).

Every engine constant flows from here; nothing is hardcoded in code paths.
The ``dspy`` section is plain typed config consumed by ``gluomem/llm/``;
this module never imports dspy.

Precedence (lowest → highest): field defaults < conf/config.yaml < variant
overlay < GLUOMEM_-prefixed env < with_overrides()/overrides arg.

Cross-validators: ``storage.vector_dim == active embedder dim``; every
scoring profile is a 4-key simplex; referenced model names resolve in the
registry; ``search.rrf_k > 0``. ``with_overrides`` re-runs full validation.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import dotenv
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from gluomem.config.variants import (
    dotted_to_nested,
    merge_variant,
    read_yaml,
)

# ---------------------------------------------------------------------------
# Default file path (the file layer).
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "conf" / "config.yaml"
_VARIANTS_DIR = _PROJECT_ROOT / "conf" / "variants"


# ---------------------------------------------------------------------------
# Nested section models.
# ---------------------------------------------------------------------------


class ModelEntry(BaseModel):
    """A single entry in the model registry.

    Carries provider routing, the provider-specific model id, an optional
    native embedding dim, the env var name holding the API key, and per-1k-token
    costs. ``dim`` is None for non-embedding models (LLMs, rerankers).

    ``provider_model`` is a LiteLLM-format string (``openrouter/<vendor>/<model>``
    for OpenRouter-routed LLMs). ``extra`` is passed verbatim as extra kwargs to
    ``dspy.LM`` (→ LiteLLM), e.g. ``{"extra_body": {"provider": {"order": [...],
    "allow_fallbacks": false, "require_parameters": true}}}`` to pin OpenRouter
    backend routing for reproducible runs.
    """

    provider: str
    provider_model: str
    # Pinned upstream revision (e.g. a Hugging Face commit). Part of the
    # embedding fingerprint; ``None`` means "whatever the hub serves".
    revision: str | None = None
    dim: int | None = None
    api_key_env: str | None = None
    api_base: str | None = None
    cost_per_1k_in: float | None = None
    cost_per_1k_out: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class StorageConfig(BaseModel):
    vector_backend: str = "sqlite-vec"
    vector_dim: int = 384
    path: str = "~/.gluomem/default.mind"


class EmbeddingConfig(BaseModel):
    model: str = "minilm"
    pad_to: int | None = None


class ComponentsConfig(BaseModel):
    """Which component implementations a config may build.

    ``allow_fake`` admits the deterministic test doubles (``provider: fake``
    embedders, ``fake`` extraction/answer/judge backends). It is off by
    default so a production config can never run on them; tests and offline
    demos opt in explicitly.
    """

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
    """Evidence-sufficiency gate, separate from ranking.

    ``score`` names the scale ``threshold`` is compared against:

    - ``fusion`` — the top ``rrf * relevance`` among eligible hits. Computed
      before any rerank, so the decision is the same with reranking on or off.
    - ``rerank`` — the top cross-encoder logit. Requires reranking enabled and
      ``calibration`` equal to the reranker's ``provider_model`` (with
      ``@revision`` when pinned), so a threshold chosen for one scale or model
      cannot silently be read on another.

    ``calibration`` identifies where the threshold came from; ``None`` means
    uncalibrated (the default 0.0 fusion threshold only abstains on no hits).
    """

    enabled: bool = True
    threshold: float = 0.0
    score: str = "fusion"
    calibration: str | None = None


#: Scores the abstention gate can read (see :class:`AbstainConfig`).
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


def _default_profiles() -> dict[str, ScoringProfile]:
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
    half_life_days: int = 30
    flat_boost_days: int = 7
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
    profiles: dict[str, ScoringProfile] = Field(default_factory=_default_profiles)


def _default_entity_type_allowlist() -> list[str]:
    # Curated semantic core, data-derived from LoCoMo ingests (698 distinct
    # extracted types over 9,191 entities, 336 of them singletons). This
    # ~24-type set covers the head (~70% of entities) and drops the
    # fragmented tail junk (`coffee scene`, `golden retriever puppy`, ...).
    return [
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


class KgConfig(BaseModel):
    seed_min_len: int = 2
    bfs_symmetric: bool = True
    alias_groups: list[list[str]] = Field(
        default_factory=lambda: [["postgresql", "postgres", "pg"], ["kubernetes", "k8s"]]
    )
    tech_allowlist: list[str] = Field(
        default_factory=lambda: ["go", "ai", "ml", "db", "npm", "k8s"]
    )
    # Entity-type pollution gate (resolution-side). ``off`` = any extracted
    # type inserted verbatim (unbounded type sprawl). ``retype`` (default)
    # relabels a non-allowlisted type to ``retype_to`` — node + relations
    # kept, type vocabulary collapsed to the allowlist. ``drop`` additionally
    # skips the entity AND prunes relations touching a non-allowlisted node —
    # aggressive graph shrink; opt-in (severs real compositional edges).
    entity_type_mode: str = "retype"
    entity_type_allowlist: list[str] = Field(
        default_factory=_default_entity_type_allowlist
    )
    retype_to: str = "other"
    # Prompt-side entity-type constraint: render ``KGEntity.entity_type`` as
    # a closed JSON-schema enum (the allowlist) in the cognify extraction
    # signature, so the extractor cannot invent types at all. CHANGES
    # EXTRACTION ITSELF — a frozen extraction cache recorded with the enum off
    # will replay the old unconstrained outputs; flip this only with a fresh
    # cache path.
    entity_type_prompt_enum: bool = False
    # Predicate-family alias groups for functional-edge semantics: members of
    # a group (e.g. [[LIVES_IN, MOVED_TO, RESIDES_IN]]) are treated as the
    # SAME functional predicate by edge retirement and the kg_edge_conflict
    # contradiction trigger, mirroring ``alias_groups`` for entities. Empty
    # (default) = exact-type matching.
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


class SemanticClassifierConfig(BaseModel):
    enabled: bool = False


class ScannerPattern(BaseModel):
    """One regex injection marker; ``weight`` is the score it contributes."""

    regex: str
    weight: float = Field(default=1.0, ge=0, le=1)


def _default_injection_patterns() -> list[ScannerPattern]:
    """Minimal prompt-injection marker list (cheap regex lane)."""
    return [
        ScannerPattern(
            regex=r"(?i)ignore (all )?(previous|prior) (instructions|context)"
        ),
        ScannerPattern(regex=r"(?i)disregard (all )?(previous|prior)"),
        ScannerPattern(regex=r"(?i)you are now (in )?(developer|dan) mode"),
        ScannerPattern(regex=r"(?i)system prompt"),
        ScannerPattern(regex=r"<\|im_start\|>"),
        ScannerPattern(regex=r"(?i)\bBEGIN (ADMIN|SYSTEM)\b"),
    ]


class ScannerConfig(BaseModel):
    patterns: list[ScannerPattern] = Field(default_factory=_default_injection_patterns)


class SafetyConfig(BaseModel):
    block_threshold: float = 0.7
    fail_mode: str = "quarantine"
    semantic_classifier: SemanticClassifierConfig = Field(
        default_factory=SemanticClassifierConfig
    )
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
    # How many candidate base frames get a contradiction check per
    # observation. 1 = single best candidate; raising it trades extra
    # contradiction calls for recall on updates whose best entity-overlap
    # frame is not the contradicted one.
    top_k: int = Field(default=1, ge=1)


class LaneConfig(BaseModel):
    """One extraction lane (fact/event/profile); importance applied at admission."""

    enabled: bool = True
    importance: str = "normal"


class LanesConfig(BaseModel):
    fact: LaneConfig = Field(default_factory=lambda: LaneConfig(importance="normal"))
    event: LaneConfig = Field(default_factory=lambda: LaneConfig(importance="normal"))
    profile: LaneConfig = Field(
        default_factory=lambda: LaneConfig(importance="important")
    )


class SubjectConfig(BaseModel):
    """Same-subject detection for P-frame attachment (KG entity overlap) and
    contradiction-candidate nomination (``trigger``)."""

    min_entity_overlap: int = Field(default=2, ge=1)
    # Which mechanism nominates frames for the contradiction check.
    # ``entity_overlap`` (default) = the same-subject base
    # frames. ``kg_edge_conflict`` = a freshly-resolved
    # relation (s, T, t_new) whose source already holds a valid same-family
    # edge to a DIFFERENT target flags the old edge's frames — this catches
    # update-type facts ({Bob, Seattle} vs {Bob, Boston} share only Bob, so
    # entity overlap structurally misses them). ``both`` = union, overlap
    # base first. P-frame attachment always uses entity overlap regardless
    # (the trigger only nominates contradiction candidates), so flipping the
    # trigger cannot inflate P-frames.
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
    """Verbatim dialog turns; de-prioritized vs extracted knowledge."""

    enabled: bool = True
    importance: str = "temporary"


class WriteCognifyConfig(BaseModel):
    enabled: bool = True
    retire_conflicting_edges: bool = True


class WriteDedupConfig(BaseModel):
    """Near-duplicate (paraphrase) dedup at admission.

    ``mode='exact'`` (default) is content-hash only.
    ``mode='semantic'`` additionally drops a candidate whose embedding
    cosine to an existing SAME-GOP frame is >= ``similarity_threshold`` (restated
    facts across sessions land in one ``{conv}:facts`` GOP and paraphrases slip
    past the exact hash). ``lanes`` restricts which lanes probe — events are
    excluded by default because same-text events at different times are distinct
    (protecting the temporal signal); raw turns and B-frames are never probed.
    """

    mode: str = "exact"
    similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    lanes: list[str] = Field(default_factory=lambda: ["fact", "profile"])
    # What happens when a candidate hits a stored near-duplicate.
    # ``drop`` (default) = first-writer-wins. ``keep_longest`` = when the
    # candidate is LONGER than the stored match, admit it and demote the
    # stored frame to ``deprecated`` (bitemporal: demote, never delete);
    # this keeps date-specific restatements that first-writer-wins discards.
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
    """Freeze one extraction and replay it across ingest-time ablation arms.

    ``mode='off'`` (default): every extraction call hits the live backend.
    Otherwise the backend is wrapped in ``CachedExtractors``
    (``gluomem.llm.cache``), keyed on the extraction INPUTS so a post-extraction
    ablation (KG resolution, dedup, frame model) replays the identical
    extraction across arms — deepseek-flash over OpenRouter at temp 0 is not
    reproducible run-to-run, so this is what makes an ingest A/B noise-free.
    ``record``/``readthrough`` build/reuse the cache at ``path``; ``replay`` is
    strict (a miss raises rather than calling the backend).
    """

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
    # Benchmark mode: anchor recency decay (created_at) on the SESSION
    # datetime instead of ingest time — without this, ingesting a 2023
    # benchmark today makes every memory identically fresh and the temporal
    # scoring signal pure noise. Default false = live semantics.
    backdate_created_at: bool = False
    # Max concurrent extraction LLM calls (lane extraction + cognify batch).
    # 1 = fully sequential (no threads). DB writes stay single-threaded
    # regardless; results are applied in deterministic input order.
    # 16: extraction is LLM-bound and OpenRouter absorbs the concurrency.
    parallelism: int = Field(default=16, ge=1)
    # Two-dials-one-budget concurrency model for BATCH ingest (one .mind per
    # conversation). ``conv_parallelism`` is how many conversations ingest
    # concurrently; ``max_concurrent_llm`` is a PROCESS-WIDE cap on simultaneous
    # extraction LLM calls (the real OpenRouter throttle). Thread count is bounded
    # by conv_parallelism * parallelism, but only max_concurrent_llm calls ever
    # hit the API — so cross-conversation fan-out can't multiply into a rate-limit
    # storm. Default 16 == the within-session peak, so single-conversation
    # ingest is unaffected; multi-conversation SUSTAINS 16 instead of bursting
    # then idling between sessions. Raise toward the account rate limit for more
    # speed. conv_parallelism=1 → fully sequential batch (exact per-conv telemetry).
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


# --- dspy section: typed config only (never `import dspy` here). -----------


class DspyCacheConfig(BaseModel):
    disk: bool = True
    dir: str = "~/.gluomem/dspy_cache"


class DspyTaskConfig(BaseModel):
    """A single DSPy task routing entry."""

    model: str
    module: str
    temperature: float | None = None
    max_tokens: int | None = None
    refine: dict[str, Any] | None = None


class DspyOptimizerConfig(BaseModel):
    auto: str = "light"


class DspyArtifactsConfig(BaseModel):
    """Compiled-program artifact store (offline optimization outputs).

    ``pins`` maps task name → expected sha256 of the saved ``program.pkl``;
    when set, loading an artifact whose hash mismatches fails loudly.
    ``require=True`` makes a missing artifact an error instead of falling
    back to the zero-shot module.
    """

    dir: str = "artifacts/dspy"
    require: bool = False
    pins: dict[str, str] = Field(default_factory=dict)


def _default_dspy_tasks() -> dict[str, DspyTaskConfig]:
    return {
        # 4096: long sessions (up to ~35 turns) produce fact lists that
        # overflow smaller caps — truncation silently loses facts. Profile
        # extraction reuses this task and inherits the same headroom.
        "fact_extraction": DspyTaskConfig(
            model="deepseek-flash", module="predict", temperature=0.0, max_tokens=4096
        ),
        # refine is OFF by default: dspy.Refine forces temperature=1.0 on
        # every attempt (breaks determinism for the task doing date math);
        # adapter-level JSONAdapter fallback already retries parse failures
        # at the configured temperature. Flip refine.enabled for ablation.
        "event_extraction": DspyTaskConfig(
            model="deepseek-flash",
            module="predict",
            temperature=0.0,
            refine={"enabled": False, "N": 2},
        ),
        "cognify": DspyTaskConfig(model="deepseek-flash", module="predict"),
        # predict for the experiment baseline; cot is the ablation flip.
        "contradiction": DspyTaskConfig(
            model="deepseek-flash", module="predict", temperature=0.0
        ),
        # Benchmark answerer (the experiment SUBJECT). Model swap per run:
        # --override dspy.tasks.answer.model=sonnet5|gpt5-mini|qwen3|glm5.
        # Deliberately NOT in TASK_SIGNATURES (optimize --task all skips it).
        # 1024: deepseek at temp 0 occasionally loops on a question; a
        # smaller cap truncates mid-marker and breaks adapter parsing.
        "answer": DspyTaskConfig(
            model="deepseek-flash", module="predict", temperature=0.0, max_tokens=1024
        ),
    }


class DspyConfig(BaseModel):
    """The ``dspy`` config section (task routing, cache, optimizer, artifacts).

    Typed config only — do NOT import dspy here.
    """

    default_model: str = "deepseek-flash"
    adapter: str = "chat"
    # Hidden-reasoning policy for OpenRouter-routed hybrid reasoning models:
    # reasoning bills 5-12x the visible output and eats max_tokens
    # unpredictably, with no extraction-quality gain.
    # False -> send {reasoning: {enabled: false}}; True -> explicitly enable;
    # None -> send nothing (provider default). A per-model
    # `models.<name>.extra.extra_body.reasoning` always wins over this.
    reasoning_enabled: bool | None = False
    track_usage: bool = True
    cache: DspyCacheConfig = Field(default_factory=DspyCacheConfig)
    tasks: dict[str, DspyTaskConfig] = Field(default_factory=_default_dspy_tasks)
    optimizer: DspyOptimizerConfig = Field(default_factory=DspyOptimizerConfig)
    artifacts: DspyArtifactsConfig = Field(default_factory=DspyArtifactsConfig)


# --- tuning + benchmark sections. -------------------------------------------


class MultiFidelityConfig(BaseModel):
    min_resource: float = 0.1
    max_resource: float = 1.0
    reduction_factor: int = 3


class LtrConfig(BaseModel):
    enabled: bool = False
    min_labeled_queries: int = 200


class TuningConfig(BaseModel):
    k: int = 10
    n_trials: int = 20000
    seed: int = 42
    multi_fidelity: MultiFidelityConfig = Field(default_factory=MultiFidelityConfig)
    top_n: int = 8
    ltr: LtrConfig = Field(default_factory=LtrConfig)


class BenchmarkJudgeConfig(BaseModel):
    """Trio-strict = 3 independent backbone families (Anthropic + OpenAI +
    open); unanimous CORRECT required. Singles always recorded alongside to
    measure self-judge inflation."""

    family: str = "trio_strict"
    backbones: list[str] = Field(
        default_factory=lambda: ["haiku", "gpt5-mini", "qwen3"]
    )
    also_record_single: bool = True
    backend: str = "dspy"  # dspy | fake (fake: deterministic offline, loud banner)


class BenchmarkAnswerConfig(BaseModel):
    """The answerer is the experiment SUBJECT; its model comes from
    ``dspy.tasks[<task>].model`` so subject swaps are one override."""

    task: str = "answer"
    backend: str = "dspy"  # dspy | fake
    context_k: int = Field(default=50, ge=1)
    abstain_text: str = "I don't know."
    # Answer-prompt variant. ``preference_aware`` (default) adds one
    # instruction to the ``base`` prompt: acknowledge the user's OWN prior
    # attempts/preferences instead of re-suggesting them as new. ``base`` is
    # the unmodified prompt; the style is recorded in run manifests.
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
    """Stratified QA sample: per_category per category, seeded shuffle."""

    per_category: int = Field(default=80, ge=1)
    categories: list[int] = Field(default_factory=lambda: [1, 2, 3, 4])
    seed: int = 42
    manifest: str = "benchmarks_data/locomo_sample_seed42.json"


class BenchmarkDatasetEntry(BaseModel):
    """One SHA-pinned benchmark dataset source (tamper-evident fetch/load)."""

    url: str
    sha256: str
    file: str


class BenchmarkFullContextConfig(BaseModel):
    max_chars: int = Field(default=120_000, ge=1)
    keep: str = "tail"  # head | tail (tail = most recent sessions survive)


def _default_benchmark_datasets() -> dict[str, BenchmarkDatasetEntry]:
    return {
        "locomo": BenchmarkDatasetEntry(
            url=(
                "https://raw.githubusercontent.com/snap-research/locomo/"
                "main/data/locomo10.json"
            ),
            sha256=(
                "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
            ),
            file="locomo10.json",
        ),
        # LongMemEval (arXiv:2410.10813), "cleaned" release on HF. Oracle =
        # evidence-only haystacks (ceiling variant); S = ~48 sessions/question.
        "longmemeval_oracle": BenchmarkDatasetEntry(
            url=(
                "https://huggingface.co/datasets/xiaowu0162/"
                "longmemeval-cleaned/resolve/main/longmemeval_oracle.json"
            ),
            sha256=(
                "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c"
            ),
            file="longmemeval_oracle.json",
        ),
        "longmemeval_s": BenchmarkDatasetEntry(
            url=(
                "https://huggingface.co/datasets/xiaowu0162/"
                "longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
            ),
            sha256=(
                "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
            ),
            file="longmemeval_s_cleaned.json",
        ),
    }


class BenchmarkMem0Config(BaseModel):
    """Head-to-head competitor (mem0 OSS, base/vector variant).

    Fairness contract: extraction backbone = same registry model as ours;
    everything else mem0 defaults. ``embedder_model``
    must carry a dim (faiss needs ``embedding_model_dims``) — local MiniLM
    replaces mem0's OpenAI-embeddings default (no OpenAI key in this
    project; also matches gluomem's embedder). ``top_k`` None falls back to
    ``benchmark.answer.context_k``.
    """

    model: str = "deepseek-flash"
    embedder_model: str = "minilm"
    top_k: int | None = Field(default=None, ge=1)
    infer: bool = True


class BenchmarkCiGateConfig(BaseModel):
    # evidence_hit@5, not precision@5: LoCoMo evidence sets are 1-3 dia_ids,
    # so strict precision@5 is capped at |evidence|/5 and a 0.7 floor could
    # never pass. Both families are still computed and reported.
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


# ---------------------------------------------------------------------------
# Root settings.
# ---------------------------------------------------------------------------


def _default_models() -> dict[str, ModelEntry]:
    return {
        "mock": ModelEntry(
            provider="fake",
            provider_model="sha256-pcg64",
            dim=None,
        ),
        "minilm": ModelEntry(
            provider="inprocess",
            provider_model="sentence-transformers/all-MiniLM-L6-v2",
            dim=384,
        ),
        "ollama": ModelEntry(
            provider="ollama",
            provider_model="ollama/nomic-embed-text",
            dim=768,
        ),
        "reranker": ModelEntry(
            provider="inprocess",
            provider_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        ),
        # --- LLMs: ALL routed via OpenRouter (one gateway, one key). --------
        # OpenRouter's routable slugs are undated; reproducibility pinning is
        # done per-experiment via ``extra.extra_body.provider`` + recording the
        # served model version, not via dated slugs.
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
        # --- Experiment matrix (answer/extraction backbones). ---------------
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
        # Hidden-reasoning policy lives on dspy.reasoning_enabled (global,
        # router-injected); a per-entry `extra.extra_body.reasoning` here
        # would override it for this model only.
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


class _DictSettingsSource(PydanticBaseSettingsSource):
    """A settings source backed by a plain dict (used for the file/variant layer).

    Placed below ``env_settings`` in the priority order so GLUOMEM_ env vars beat
    the file layer, matching the documented precedence.
    """

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
    """Root typed config tree (pydantic-settings v2).

    Precedence (lowest → highest): field defaults < file layer (yaml + variant)
    < GLUOMEM_ env < init kwargs (overrides).
    """

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
    search: SearchConfig = Field(default_factory=SearchConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    kg: KgConfig = Field(default_factory=KgConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    write: WriteConfig = Field(default_factory=WriteConfig)
    dspy: DspyConfig = Field(default_factory=DspyConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    # --- load context (used by with_overrides to re-load + re-validate) -----
    _load_path: Path | None = PrivateAttr(default=None)
    _load_variant: str | None = PrivateAttr(default=None)
    _load_variants_dir: Path | None = PrivateAttr(default=None)
    _load_overrides: dict[str, Any] = PrivateAttr(default_factory=dict)

    # --- cross-validators (core config) ------------------------------------

    @model_validator(mode="after")
    def _validate_model_refs(self) -> AppConfig:
        """Every referenced model name resolves in the registry."""
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
        """storage.vector_dim MUST equal the active embedder's registry dim.

        The mock embedder (``dim is None``) adapts to any ``storage.vector_dim``
        — it has no fixed native dim, so the mismatch check is skipped for it.
        """
        embed_name = self.embedding.model
        entry = self.models.get(embed_name)
        if entry is None:
            # Unknown reference: skip here; ``_validate_model_refs`` owns the
            # clean "Unknown model reference" error (do not mask it with a
            # raw KeyError, regardless of validator ordering).
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
        """``search.abstain`` names a supported score with matching calibration."""
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
        """Every scoring profile's weights form a 4-key simplex (sum == 1.0)."""
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
        """``search.default_profile`` MUST resolve to a defined scoring profile."""
        if self.search.default_profile not in self.scoring.profiles:
            raise ValueError(
                f"search.default_profile '{self.search.default_profile}' not "
                f"found in scoring.profiles: {sorted(self.scoring.profiles)}"
            )
        return self

    # --- cross-validators (write/safety surface) -----------------------------

    @model_validator(mode="after")
    def _validate_write_enums(self) -> AppConfig:
        """Write/safety enums fail loudly on unknown values (no silent fallthrough)."""
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
        """Every lane/raw-turn importance MUST be a scoring.importance_mult key."""
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
        """Every safety.scanner pattern MUST compile at config load, not at ingest."""
        for i, pattern in enumerate(self.safety.scanner.patterns):
            try:
                re.compile(pattern.regex)
            except re.error as exc:
                raise ValueError(
                    f"safety.scanner.patterns[{i}].regex does not compile: "
                    f"{pattern.regex!r} ({exc})"
                ) from exc
        return self

    # --- cross-validators (benchmark surface) ---------------------------------

    @model_validator(mode="after")
    def _validate_benchmark_refs(self) -> AppConfig:
        """Judge backbones resolve in the registry; enums fail loudly."""
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
        """mem0 model refs resolve; embedder must carry a dim (faiss needs it)."""
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
        """Sample categories are unique ints within 1..7 (LoCoMo uses 1..5,
        LongMemEval 1..7; per-dataset range errors surface at sampling)."""
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
        """Gate metric parses and, when @k, k is a computed retrieval cutoff."""
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

    # --- with_overrides -----------------------------------------------------

    def with_overrides(self, overrides: dict[str, Any]) -> AppConfig:
        """Return a NEW validated AppConfig with dotted-key overrides on top.

        The overrides sit at the TOP of precedence (above env). The new config
        is rebuilt from the same file/variant layers and re-runs all validators.

        Example::

            cfg.with_overrides({"search.rrf_k": 99, "embedding.model": "mock"})
        """
        merged = {**self._load_overrides, **overrides}
        return load_config(
            path=self._load_path,
            variant=self._load_variant,
            variants_dir=self._load_variants_dir,
            overrides=merged,
        )


# ---------------------------------------------------------------------------
# load_config factory.
# ---------------------------------------------------------------------------


def load_config(
    path: str | Path | None = None,
    variant: str | None = None,
    overrides: dict[str, Any] | None = None,
    variants_dir: str | Path | None = None,
    env_file: str | Path | None = None,
) -> AppConfig:
    """Load the typed ``AppConfig`` from the file layer + env + overrides.

    Parameters
    ----------
    path:
        Path to the YAML file layer. Defaults to ``conf/config.yaml``.
    variant:
        Optional variant overlay name; looked up as
        ``{variants_dir}/{variant}.yaml`` and deep-merged over the file layer.
    overrides:
        Optional dotted-key overrides (e.g. ``{"search.rrf_k": 99}``); these
        sit at the TOP of precedence (above env).
    variants_dir:
        Directory holding variant YAMLs. Defaults to ``conf/variants``.
    env_file:
        Dotenv file loaded into ``os.environ`` BEFORE anything else (secrets
        like ``OPENROUTER_API_KEY`` — API keys are looked up via
        ``ModelEntry.api_key_env`` at LM construction, never stored in
        config). Defaults to ``<project root>/.env`` when that file exists.
        Already-set real environment variables are NEVER overridden.

    Precedence: field defaults < file < variant < GLUOMEM_ env < overrides.
    """
    dotenv_path = Path(env_file) if env_file is not None else _PROJECT_ROOT / ".env"
    if dotenv_path.exists():
        dotenv.load_dotenv(dotenv_path, override=False)

    yaml_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    file_layer = read_yaml(yaml_path)

    vdir = Path(variants_dir) if variants_dir is not None else _VARIANTS_DIR
    if variant is not None:
        file_layer = merge_variant(file_layer, variant, vdir)

    init_data = dotted_to_nested(overrides or {})
    file_layer_final = file_layer

    # A one-off subclass that injects the file layer as a low-priority source
    # (below env_settings). Overrides ride in via init kwargs (highest priority).
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
