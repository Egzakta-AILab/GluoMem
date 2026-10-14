# GluoMem Domain Context

## Domain Terms

| Term | Definition |
|---|---|
| Memory Scope | The three-part isolation boundary (`tenant`, `principal`, `namespace`) that partitions all stored frames, sessions, and knowledge graph nodes so one user or workspace never observes or mutates another's memory. |
| Memory Frame | A single unit of stored memory following GOP video-compression semantics: an independent keyframe (`I`), a forward delta (`P`), or a conflict-correcting bidirectional frame (`B`). |
| Group of Pictures (GOP) | A logical stream of memory frames within a Memory Scope (such as a conversation lane or topic session) anchored by an `I`-frame and refined by subsequent `P` and `B` frames. |
| Memory Kind | The epistemic category of a frame: `fact`, `preference`, `procedure`, `episode`, `correction`, or `summary`. |
| Frame Lifecycle Status | The visibility state of a frame: `active` (eligible for recall), `superseded` (replaced by a newer correction or revision), `expired` (past its retention window), or `tombstoned` (soft-deleted by user or policy). |
| Recall Eligibility | The invariant that a frame may enter retrieval, knowledge-graph seeding, reranking, or state reconstruction only when its trust is not `quarantined`, its status is `active`, and its expiration timestamp has not passed. |
| Bitemporal Knowledge Graph | A scoped graph of entities and relations tracking both world validity (`valid_from`, `valid_to`) and system ingestion time (`recorded_at`), preserving retired intervals in relation history. |
| Admission Gate | The single write boundary that scans candidate text for prompt injection, checks exact and semantic duplicates within the active Memory Scope, and indexes admitted frames across full-text and vector stores. |
| Hybrid Search | Recall pipeline combining scoped lexical search and scoped dense vector search via Reciprocal Rank Fusion (RRF), multiplied by a 4-signal relevance profile (`temporal`, `popularity`, `contextual`, `importance`) and gated by calibrated abstention. |
| Candidate Cache | An offline tuning artifact storing per-query candidate feature rows and retriever ranks so parameter optimization (`Optuna`) and ablations run without model or database calls. |

## Domain-to-Code Mapping

| Domain Term | Code Location |
|---|---|
| Memory Scope | `gluomem/storage/scope.py` (`MemoryScope`, `DEFAULT_SCOPE`, `resolve_scope`) |
| Memory Frame | `gluomem/frames/store.py` (`Frame`, `FrameStore`) |
| Group of Pictures (GOP) | `gluomem/frames/sessions.py` (`ensure_session`, `close_session`, `get_session`) |
| Memory Kind | `gluomem/storage/scope.py` (`MemoryKind`, `MEMORY_KINDS`) |
| Frame Lifecycle Status | `gluomem/storage/scope.py` (`FrameStatus`, `FRAME_STATUSES`) |
| Recall Eligibility | `gluomem/frames/eligibility.py` (`is_recall_eligible`, `RECALL_ELIGIBLE_SQL`) |
| Bitemporal Knowledge Graph | `gluomem/kg/graph.py` (`KnowledgeGraph`, `Entity`, `Relation`, `RelationInterval`) |
| Admission Gate | `gluomem/write/gate.py` (`admit`, `AdmitResult`) |
| Hybrid Search | `gluomem/search/hybrid.py` (`HybridSearch`, `SearchHit`, `SearchResult`) |
| Candidate Cache | `gluomem/tune/candidate_cache.py` (`CandidateRow`, `CandidateCache`, `build_candidate_cache`) |

## Context Boundaries

### Upstream
- Host applications (such as KVARK or CLI benchmarks) supply authenticated scope headers (`tenant_id`, `principal_id`, `namespace`), conversation turns, and recall queries.

### Downstream
- Embedded SQLite (`.mind` with `FTS5` + `sqlite-vec`) or PostgreSQL 18 (`pgvector` + `tsvector`), managed through `StorageBackend`, `VectorStore`, and `FtsStore` protocols and Alembic migrations.
- DSPy LM providers and sentence-transformer embedding/reranking models configured via `AppConfig`.

### Not in Scope
- Host-application RBAC policy evaluation, document ACL management, or UI chat orchestration (owned by the upstream host service).

## Architectural Decision Records

### ADR-001: Mandatory MemoryScope Isolation Across Storage, KG, and Search Lanes
**Date:** 2026-04-19
**Status:** Accepted

#### Context
GluoMem originally operated as a single-user embedded `.mind` store where `memory_frames`, `sessions`, `knowledge_entities`, and `knowledge_relations` had no tenant or principal partitioning. Running GluoMem as a shared PostgreSQL service requires strict isolation so that vector KNN (`LIMIT k`), full-text search, deduplication, and knowledge-graph traversal cannot leak across principals or starve a small principal's recall pool behind another principal's rows.

#### Decision
Introduce `MemoryScope(tenant_id, principal_id, namespace)` (defaulting to `MemoryScope("local", "default", "personal")` for embedded CLI/benchmark callers). Every stateful table (`sessions`, `memory_frames`, `knowledge_entities`, `knowledge_relations`) carries `(tenant_id, principal_id, namespace)` columns with composite indexes. Both `VectorStore.search` and `FtsStore.search`/`search_like` push scope and recall-eligibility predicates (`status = 'active'`, `trust != 'quarantined'`, `expires_at IS NULL OR expires_at > now`) into the SQL query before `ORDER BY ... LIMIT k`.

#### Consequences
- Zero cross-principal data or graph-distance leakage in shared SQLite or PostgreSQL databases.
- Small scopes are never starved by top-`k` vector cutoffs from larger scopes.
- Existing single-user tests and CLI workflows continue to work unchanged via `DEFAULT_SCOPE`.
