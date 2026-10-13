# gluomem — Technical Design

> Companion to `SPEC.md` and [`docs/research/PAPER.md`](./docs/research/PAPER.md). Scope is the memory package plus evaluation; production surfaces are out of scope (§11).

> **Implementation status (2026-10-13, version 0.0.1).** The tuning engine (§9, `tune/` in the layout below, the `tune` and `ablate` commands) is designed but not implemented. `storage/vector_store.py` and `search/abstain.py` do not exist as separate modules; the only vector store is `SqliteVecStore` in `storage/sqlite_vec.py`, and abstention lives in `search/hybrid.py`. The `experiments/` run directories were never published. The README lists current scope and known limitations.

---

## 0. Architecture

```
   EXPERIMENT CLI:  ingest · bench run · bench rescore · tune · ablate
        │
   ┌────▼─────────────────────────────────────────────────────────────┐
   │ WRITE (DSPy)                       READ                            │
   │  admit() → dedup→scan→index        HybridSearch                    │
   │   ↑ ingest benchmark data           FTS5 BM25 ∥ vec KNN → RRF(k)   │
   │   ↑ lane extract (DSPy)             → ×relevance(profile)          │
   │   ↑ cognify (DSPy)                  → rerank? → abstain gate       │
   └────┬───────────────────────────────────────┬──────────────────────┘
        │  CORE: FrameStore(I/P/B) · KnowledgeGraph(bitemporal) · Scoring
        │  (protocols ↓)
   VectorStore(sqlite-vec, exact) · Embedder(MiniLM) · Reranker(opt) · LLM(DSPy/Anthropic)
        │
   STORAGE: SQLite .mind file  (tables + FTS5 + vec0)
   CROSS-CUTTING: Config (pydantic-settings) · Benchmark harness · Tuning engine
```

### Package layout
```
gluomem/
  config/    app_config.py  registry.py  variants.py
  storage/   schema.sql  db.py  vector_store.py  sqlite_vec.py
  frames/    store.py  sessions.py  content_hash.py  compaction.py        # I/P/B
  embedding/ base.py  providers.py  chunker.py  reranker.py  fingerprint.py  compat.py
  search/    hybrid.py  scoring.py  abstain.py  sanitize.py
  kg/        graph.py  normalizer.py  temporal.py
  write/     gate.py  ingest.py  lanes.py  cognify.py                      # benchmark ingest only
  llm/       signatures.py  modules.py  router.py  judge.py  bootstrap.py  optimize.py
  bench/     adapters/  retriever.py  judge.py  metrics.py  rescore.py  report.py  baselines.py
  tune/      candidate_cache.py  rescore.py  objective.py  two_tier.py  ltr.py
  cli.py  components.py  extras.py  timeutil.py
conf/  artifacts/  benchmarks_data/  experiments/  tests/
```
Out of scope: MCP server, lifecycle hooks, wiki compiler, enrichment/synth queue, identity/awareness layers, harvest adapters.

---

## 1. Storage

One `*.mind` SQLite file (WAL, FK on). The schema provides an 8-value `source` CHECK, a `trust` column, `event_time` split from `created_at`, an edge `UNIQUE` constraint, and a vector dimension that flows from the active embedder.

```sql
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE sessions (id INTEGER PK AUTOINCREMENT, gop_id TEXT UNIQUE NOT NULL,
  project_id TEXT, status TEXT DEFAULT 'active', started_at TEXT DEFAULT (datetime('now')),
  ended_at TEXT, summary TEXT);

CREATE TABLE memory_frames (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  frame_type TEXT NOT NULL CHECK(frame_type IN('I','P','B')),
  gop_id TEXT NOT NULL REFERENCES sessions(gop_id),
  t INTEGER NOT NULL DEFAULT 0,
  base_frame_id INTEGER REFERENCES memory_frames(id),         -- P/B → base I
  content TEXT NOT NULL,
  importance TEXT NOT NULL DEFAULT 'normal'
    CHECK(importance IN('critical','important','normal','temporary','deprecated')),
  source TEXT NOT NULL DEFAULT 'user_stated'
    CHECK(source IN('user_stated','tool_verified','agent_inferred','import','system','personal','workspace','team_sync')),
  trust  TEXT NOT NULL DEFAULT 'normal' CHECK(trust IN('trusted','normal','low','quarantined')),
  access_count INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),   -- INGEST time = recency-decay anchor
  event_time TEXT,                                       -- WORLD/event time = range-query anchor
  last_accessed TEXT NOT NULL DEFAULT (datetime('now')),
  content_hash TEXT);
CREATE INDEX idx_frames_gop_t ON memory_frames(gop_id,t);
CREATE INDEX idx_frames_type  ON memory_frames(frame_type,gop_id);
CREATE INDEX idx_frames_base  ON memory_frames(base_frame_id);
CREATE INDEX idx_frames_hash  ON memory_frames(content_hash);
CREATE INDEX idx_frames_event ON memory_frames(event_time);

CREATE VIRTUAL TABLE memory_frames_fts USING fts5(content, content_rowid='id', tokenize='porter unicode61');
CREATE TABLE memory_frame_chunks (id INTEGER PK AUTOINCREMENT,
  frame_id INTEGER NOT NULL REFERENCES memory_frames(id) ON DELETE CASCADE,
  chunk_idx INTEGER NOT NULL, content TEXT NOT NULL, char_start INTEGER, char_end INTEGER,
  UNIQUE(frame_id, chunk_idx));
-- vec0 tables created with dim = active embedder dim (never hardcoded):
--   memory_frames_vec(embedding float[<dim>]) ; memory_frame_chunks_vec(embedding float[<dim>])

CREATE TABLE knowledge_entities (id INTEGER PK AUTOINCREMENT, entity_type TEXT, name TEXT,
  properties TEXT DEFAULT '{}', valid_from TEXT DEFAULT (datetime('now')), valid_to TEXT,
  recorded_at TEXT DEFAULT (datetime('now')));
CREATE TABLE knowledge_relations (id INTEGER PK AUTOINCREMENT,
  source_id INTEGER REFERENCES knowledge_entities(id), target_id INTEGER REFERENCES knowledge_entities(id),
  relation_type TEXT, confidence REAL DEFAULT 1.0, properties TEXT DEFAULT '{}',
  valid_from TEXT DEFAULT (datetime('now')), valid_to TEXT, recorded_at TEXT DEFAULT (datetime('now')),
  UNIQUE(source_id, target_id, relation_type));
-- earlier validity intervals of a relation that was retired and re-asserted
CREATE TABLE knowledge_relation_history (id INTEGER PK AUTOINCREMENT,
  relation_id INTEGER REFERENCES knowledge_relations(id) ON DELETE CASCADE,
  valid_from TEXT NOT NULL, valid_to TEXT NOT NULL, recorded_at TEXT NOT NULL,
  closed_recorded_at TEXT NOT NULL DEFAULT (datetime('now')));
CREATE TABLE kg_entity_frames (entity_id INTEGER REFERENCES knowledge_entities(id) ON DELETE CASCADE,
  frame_id INTEGER REFERENCES memory_frames(id) ON DELETE CASCADE, PRIMARY KEY(entity_id, frame_id));
```

**VectorStore protocol** (only `sqlite-vec` implemented; exact KNN; the protocol keeps the door open for ANN as future work):
```python
class VectorStore(Protocol):
    dim: int
    def add(self, rowid: int, vec: np.ndarray) -> None: ...      # vec0: rowid must be SQL literal
    def add_batch(self, items: list[tuple[int, np.ndarray]]) -> None: ...
    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]: ...
    def delete(self, rowid: int) -> None: ...
```

---

## 2. Frames (I/P/B)

```python
@dataclass
class Frame:
    id: int; frame_type: Literal['I','P','B']; gop_id: str; t: int
    base_frame_id: int|None; content: str
    importance: Importance; source: FrameSource; trust: Trust
    access_count: int; created_at: str; event_time: str|None; last_accessed: str

class FrameStore:
    def create_i_frame(self, gop, content, importance='normal', source='user_stated',
                       event_time=None) -> Frame: ...          # dedup by content_hash
    def create_p_frame(self, gop, content, base_id, importance, source) -> Frame: ...   # delta on I
    def create_b_frame(self, gop, description, base_id, references,
                       importance=None) -> Frame: ...          # importance inherits from base when None
    def reconstruct_state(self, gop) -> tuple[Frame|None, list[Frame]]: ...  # latest I + P-since
    def compact(self, cfg) -> None: ...                        # re-embed merged I, referential delete
```

**Who writes P/B:**
- **P** — a new observation about the same subject as an existing I-frame (delta/refinement) during ingestion of a multi-session conversation.
- **B** — the contradiction detector finds a later statement that conflicts with a stored I/P; it writes a B correction `{description, references}` and may retire the superseded KG edge (bitemporal). This is the conflict-resolution mechanism.

Importance multipliers, compaction trigger/keep-window, and the merge separator are all config. The compaction trigger fires at 10 accumulated P-frames and keeps the most recent 5.

**Write atomicity.** Every `create_*` embeds first, then commits the frame row with its FTS, vector and chunk rows in one transaction. `FrameStore.index_state(id)` reports whether a frame's projections match its current text; dedup hits call `ensure_indexed`, and `repair_indexes()` is the recovery path for existing files. Compaction embeds the merged text first, then commits the content update, re-index, `memory_frame_lineage` rows (constituent id, type, hash, source, trust, timestamps; not text), KG-link carry-over and P-frame deletes together. `memory_frame_lineage` is created by `schema.sql` with `IF NOT EXISTS`, so older files gain it on open.

---

## 3. Embedding / chunking / reranker

Protocols with these defaults: in-process `sentence-transformers/all-MiniLM-L6-v2` (384 dims, native — no zero-pad) for embeddings; optional `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker. The chunker is a pure function with all sizes in config (default chunk size 2000, overlap 200, whole-frame threshold 1500). Factories (`make_embedder`, `make_reranker`, `make_extractors`, the bench answer/judge builders) build the implementation the registry entry's `provider` names (`inprocess` = sentence-transformers; `fake` = the hash mock) or raise `ProviderUnavailableError` / `MissingExtraError`; there is no mock fallback. Test doubles also need `components.allow_fake: true` (`FakeComponentError` otherwise). Every vector write is length-checked (`assert_dim`), and each `.mind` file records an `EmbedderFingerprint` (implementation, model, revision, dim, preprocessing, index schema) in `meta.embedding_fingerprint` on first bind (`gluomem/embedding/compat.py`). `FrameStore` and `HybridSearch` reject a different fingerprint, or a file holding vectors with none recorded, before touching a vector. `FrameStore.rebuild_index` re-embeds everything first and then swaps vectors and fingerprint in one transaction; `adopt_embedder` records a fingerprint for a pre-fingerprint file without re-embedding. `gluomem.components.component_report` (CLI `gluomem components`) lists effective implementations, fingerprints, API-key variable names and whether they are set, never their values.

---

## 4. Search (HybridSearch)

```
weights   = cfg.scoring.profiles[profile].weights
fts_ids   = fts_search(sanitize(query), lane_fetch)     # preserves dotted/hyphenated tokens; LIKE fallback for empty/short
vec_ids   = chunk_knn(query, lane_fetch) or whole_frame_knn(query, lane_fetch)
rrf[id]  += 1/(cfg.search.rrf_k + rank)                 # per lane, rrf_k = 60
frames    = load + event_time since/until filter
gdist     = kg.contextual(query, recent_entities)
final     = rrf · Σ w_x·feature_x
sort; rerank top cfg.search.rerank_pool (replace score w/ logit);
abstain if top_final < cfg.search.abstain_threshold     # gates recall
```
`final = rrf × relevance` keeps IR-quality separate from personalization — the property that enables the offline tuning in §9.

---

## 5. Scoring (all params config; ablatable)

```python
def temporal(days, c):    return 1.0 if days <= c.flat_boost_days else 0.5 ** (days / c.half_life)
def popularity(n, c):     return 1.0 + math.log10(1 + n) * c.log_coef
def contextual(d, c):     return c.bfs_steps[d] if d <= c.bfs_depth and d < len(c.bfs_steps) else 0.0
def importance(lv, c):    return c.importance_mult[lv]
relevance = w.t*temporal + w.p*popularity + w.c*contextual + w.i*importance
```
`temporal` anchors on `created_at` with a 30-day half-life and a 7-day flat boost; range queries use `event_time`. The `contextual` step vector is `[1.0, 0.7, 0.4, 0.2, 0.0]` over BFS distances 0–4. The `importance_mult` map is `{critical: 2.0, important: 1.5, normal: 1.0, temporary: 0.7, deprecated: 0.3}`. The `balanced` profile weights are `{temporal: 0.4, popularity: 0.2, contextual: 0.2, importance: 0.2}`. Each signal can be zeroed via config for ablations.

---

## 6. Knowledge graph + temporal

Bitemporal entities/relations, retire-not-delete, `as_of(T)`. Valid time (`valid_from`/`valid_to`) says when a fact held in the world; `recorded_at` says when GluoMem learned it. Re-asserting a retired relation after a gap archives the closed interval in `knowledge_relation_history` and starts a new one on the same row (a re-assertion valid from before the old `valid_to` just continues the interval); `as_of(T)` checks the current row and its history. Conflicting single-valued facts are resolved by valid time: a newer-valid fact closes the older edge at its own `valid_from`, and an older fact recorded late is closed at the newer edge's `valid_from` instead of retiring it. All timestamps follow one UTC rule (`gluomem/timeutil.py`): naive is UTC, aware is converted to naive UTC on write, date-only means midnight UTC. **Seeding** is a word-boundary/token match (config `seed_min_len` aligned with the technical-token allowlist). **BFS** is symmetric (in + out), with config depth and step vector. KG-lane errors emit a metric (never a silent zero). Backdating: DSPy resolves relative time cues to `event_time`; `created_at` stays ingest time.

```python
class KnowledgeGraph:
    def seed_entities(self, query) -> list[int]: ...     # word-boundary
    def bfs_distances(self, seeds, depth) -> dict[int,int]: ...  # symmetric
    def contextual(self, query, recent) -> dict[int,int]: ...    # frame_id → distance
    def retire_entity(self, eid): ... ; def as_of(self, iso): ...
```

---

## 7. Write path (benchmark ingest) + admission gate

All writes funnel through one `admit()` (dedup + index + minimal injection scan). LLM extraction is **DSPy** (§8).

```python
def admit(content, source, trust, *, scanner, store, kg, cfg) -> AdmitResult:
    v = scanner.scan(content, context=source)            # cheap regex; DSPy classifier optional
    if v.score >= cfg.safety.block_threshold and cfg.safety.fail_mode == 'quarantine':
        trust = 'quarantined'
    frame = store.create_i_frame(content, source=source, trust=trust)   # dedup inside
    # index FTS + vector + chunks ; link KG entities
    return AdmitResult(frame=frame)
```
**Ingest pipeline:** benchmark conversation → sessions/GOP → per-turn frames → lane extraction (fact/event/profile) → cognify (entities/relations) → P/B from deltas/contradictions.

**Eligibility** (`gluomem/frames/eligibility.py`): a quarantined frame is stored for inspection but is not recall-eligible. Search lanes, KG contextual scoring, the rerank pool and `reconstruct_state` skip it; the gate skips semantic dedup and KG linking for it; ingest does not cognify flagged observations and a flagged contradiction cannot write a B-frame, demote, or retire edges. Compaction merges a P-frame into its base only when same GOP, same source, both eligible, and the P's trust is at least the base's, so merged text is never promoted.

---

## 8. DSPy LLM layer (DSPy 3.x)

Typed outputs go directly on `dspy.OutputField` (no removed `TypedPredictor`/`Assert`/`Suggest`; use `dspy.Refine` for validity retries). **Default LM = `anthropic/claude-haiku-4-5-20251001`** (dated snapshot for reproducibility), routed per task via the registry.

```python
class ExtractedEvent(pydantic.BaseModel):
    text: str; event_date: datetime | None; confidence: float = Field(ge=0, le=1)
class EventExtraction(dspy.Signature):
    """Extract episodic events; resolve relative time cues to ISO dates."""
    source_text: str = dspy.InputField(); reference_date: datetime = dspy.InputField()
    events: list[ExtractedEvent] = dspy.OutputField()

class CognifyExtraction(dspy.Signature):
    """Extract typed entities + relations from a frame."""
    text: str = dspy.InputField()
    entities: list[KGEntity] = dspy.OutputField(); relations: list[KGRelation] = dspy.OutputField()

class JudgeQA(dspy.Signature):
    question: str = dspy.InputField(); gold: str = dspy.InputField(); prediction: str = dspy.InputField()
    verdict: Verdict = dspy.OutputField()      # Verdict{correct:bool, score:float, rationale:str}

class LMRouter:
    def lm_for(self, task) -> dspy.LM:
        e = self.registry.resolve(self.cfg.tasks[task].model)
        return dspy.LM(e.provider_model, api_base=e.api_base,
                       api_key=os.environ.get(e.api_key_env or ''),
                       temperature=self.cfg.tasks[task].temperature, cache=True)
    @contextmanager
    def bind(self, task):
        with dspy.context(lm=self.lm_for(task)): yield
```
- Extraction → `dspy.Predict` + Haiku. Judge/synthesis → `dspy.ChainOfThought` on a strong model; the QA judge uses `anthropic/claude-sonnet-4-6` (pinned to a dated snapshot).
- **Optimize offline** with `MIPROv2` (or `BootstrapFewShot`), `save(dir, save_program=True)` (writes `metadata.json` with versions), load read-only with a content-hash check. Production never calls `.compile()`.
- **TrioJudge**: 3 independent backbones → strict/majority; per-row `JudgeRecord` persisted (JSONL) for offline rescore. A single self-judge tends to score a few points above an independent trio-strict panel; report the measured gap per dataset.

---

## 9. Tuning subsystem

> **Not implemented.** This section is the design only; `gluomem/tune/__init__.py` is a placeholder.

`final = rrf × relevance` factors into an expensive retrieve and a cheap rescore. Retrieve each eval query's pool once; cache the raw per-feature inputs plus per-retriever ranks; then re-rank thousands of parameter vectors with zero model calls, scored by nDCG/Recall@k against benchmark-derived gold labels (reusing answer spans / MCQ keys, requiring no additional labeling).

```python
@dataclass
class CandidateRow:
    doc_id:str; fts_rank:int; vec_rank:int; days_since_created:float
    access_count:int; bfs_distance:int; importance_level:str; grade:int=0   # gold

def rescore(rows, p) -> list[str]:        # pure; shared by tuner AND runtime
    out=[]
    for r in rows:
        rrf = 1/(p.rrf_k+r.fts_rank) + 1/(p.rrf_k+r.vec_rank)
        rel = (p.w_t*_temporal(r.days_since_created,p) + p.w_p*(1+math.log10(1+r.access_count)*p.log_coef)
             + p.w_c*_contextual(r.bfs_distance,p)     + p.w_i*p.importance_mult[r.importance_level])
        out.append((r.doc_id, rrf*rel))
    out.sort(key=lambda x:x[1], reverse=True); return [d for d,_ in out[:p.pool_size]]

def objective(trial):                     # Optuna; no LLM, no retrieval
    raw=np.array([trial.suggest_float(n,1e-3,1) for n in ('w_t','w_p','w_c','w_i')]); w=raw/raw.sum()
    p=ParamVector(w_t=w[0],...,rrf_k=trial.suggest_int('rrf_k',*space.rrf_k), half_life=...)
    return float(np.mean([ndcg_at_k(rescore(cache.rows(q),p), qrels[q], k) for q in cache.query_ids()]))

study=optuna.create_study(direction='maximize',
        sampler=optuna.samplers.TPESampler(multivariate=True,group=True,seed=cfg.seed),
        pruner=optuna.pruners.HyperbandPruner(...))
study.enqueue_trial(cfg.incumbent_params)   # never regress
study.optimize(objective, n_trials=cfg.n_trials)
```
**Two-tier:** Tier-1 (no model calls, thousands of trials, Hyperband-pruned) → promote the top-N distinct configs to Tier-2 end-to-end LLM-judge QA eval → ship the Tier-2 winner (IR ranking ≠ QA accuracy). **Cheap params** (weights, `rrf_k`, half-life, BFS steps, importance multipliers) tune from one cache build; **expensive params** (chunk size, embedder) sit in an outer cache-generation grid. An optional **LambdaMART** ranker (LightGBM) trains over the same cache, gated by `min_labeled_queries` and Tier-2 validated.

---

## 10. Benchmark harness

```python
class DatasetAdapter(Protocol): name:str; sha:str; def load(self)->list[Sample]: ...
class Retriever(Protocol):      def answer(self, s: Sample)->Prediction: ...   # gluomem + baselines
class Judge(Protocol):          family:str; def score(self, s, p)->Score: ...

@dataclass(frozen=True)
class ProtocolTag: metric_family:str; judge_family:str; judge_count:int; backbone:str; dataset_sha:str
@dataclass(frozen=True)
class Score: value:float; tag:ProtocolTag; n:int

class MetricRegistry:
    def aggregate(self, scores) -> Score:           # REFUSES to mix families
        if len({(s.tag.metric_family,s.tag.judge_family,s.tag.backbone) for s in scores})>1:
            raise IncompatibleMetricsError(...)
```
**Baselines in-harness:** full-context (no memory), naive vector RAG, BM25-only. Offline rescore runs from committed `JudgeRecord`s plus the dataset SHA, offline against committed artifacts with no network or model calls. The report pivots on `ProtocolTag` (an F1 score cannot sit beside an LLM-judge accuracy). Tuning reuses these exact adapters and samples.

---

## 11. Out of scope
MCP server · lifecycle hooks · wiki compiler · enrichment/synth queue · identity/awareness layers · harvest adapters · LanceDB/ANN scale · multi-user/RBAC. The protocols leave the door open for these as future work, but nothing is implemented.

---

## 12. Config & stack
`pydantic-settings v2` `AppConfig` (precedence: defaults < file < variant < env < override); a **ModelRegistry** of by-name models; a **ScoringProfile** validated as a simplex; a cross-validator asserting `storage.vector_dim == embedder.dim`. Stack: Python 3.11+, `sqlite3` + `sqlite-vec`, `sentence-transformers`, **DSPy 3.x** (Anthropic via LiteLLM), `optuna` + `ranx` + `lightgbm`, `pytest`, `uv`. The full default configuration ships as `conf/config.yaml`.

---

## References

- **Reciprocal Rank Fusion** — Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods," SIGIR 2009.
- **DSPy** — dspy.ai documentation.
- **sqlite-vec** — github.com/asg017/sqlite-vec.
- **Optuna** — optuna.org (TPE sampler, Hyperband pruner).
- **pydantic-settings** — docs.pydantic.dev.
- **Anthropic model ids** — platform.claude.com model overview.
