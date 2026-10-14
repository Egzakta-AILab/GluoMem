"""Offline candidate pool cache for zero-call retrieval parameter tuning.

Separates the expensive one-time retrieval pass (FTS + vector KNN + KG BFS +
frame metadata load) from the cheap parameter rescore pass (`DESIGN.md` §9).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gluomem.embedding.compat import stored_fingerprint
from gluomem.embedding.fingerprint import fingerprint_of
from gluomem.frames.eligibility import is_recall_eligible
from gluomem.frames.store import Frame, FrameStore
from gluomem.kg.graph import KnowledgeGraph
from gluomem.search.sanitize import sanitize
from gluomem.storage.db import MindDB
from gluomem.timeutil import Clock, parse_utc, utc_key, utc_now

if TYPE_CHECKING:
    from gluomem.config.app_config import AppConfig
    from gluomem.embedding.base import Embedder

__all__ = [
    "CacheManifest",
    "CandidateCache",
    "CandidateRow",
    "QuerySpec",
    "build_cache_for_benchmark",
    "build_cache_for_db",
]


@dataclass(frozen=True)
class CandidateRow:
    """One candidate document in a query's cached retrieval pool."""

    doc_id: str
    fts_rank: int | None
    vec_rank: int | None
    days_since_created: float
    access_count: int
    bfs_distance: int
    importance_level: str
    grade: int = 0
    frame_id: int | None = None
    dia_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateRow:
        fts_rank = data.get("fts_rank")
        vec_rank = data.get("vec_rank")
        frame_id = data.get("frame_id")
        dia_id = data.get("dia_id")
        return cls(
            doc_id=str(data["doc_id"]),
            fts_rank=int(fts_rank) if fts_rank is not None else None,
            vec_rank=int(vec_rank) if vec_rank is not None else None,
            days_since_created=float(data["days_since_created"]),
            access_count=int(data["access_count"]),
            bfs_distance=int(data["bfs_distance"]),
            importance_level=str(data["importance_level"]),
            grade=int(data.get("grade", 0)),
            frame_id=int(frame_id) if frame_id is not None else None,
            dia_id=str(dia_id) if dia_id is not None else None,
        )


@dataclass(frozen=True)
class QuerySpec:
    """A query with known gold relevant document IDs for cache construction."""

    query_id: str
    query: str
    relevant_doc_ids: tuple[str, ...] = ()
    relevant_grades: Mapping[str, int] = field(default_factory=dict)
    conv_id: str | None = None
    since: str | None = None
    until: str | None = None
    clock_iso: str | None = None

    def qrel(self) -> dict[str, int]:
        out = {doc_id: 1 for doc_id in self.relevant_doc_ids}
        for doc_id, grade in self.relevant_grades.items():
            if int(grade) > 0:
                out[str(doc_id)] = int(grade)
        return out


@dataclass(frozen=True)
class CacheManifest:
    """Provenance and fingerprint metadata for a saved :class:`CandidateCache`."""

    dataset: str
    dataset_sha: str
    embedder_fingerprint: dict[str, Any] | None
    lane_fetch: int
    bfs_depth: int
    query_count: int
    created_at: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CacheManifest:
        fp = data.get("embedder_fingerprint")
        return cls(
            dataset=str(data["dataset"]),
            dataset_sha=str(data["dataset_sha"]),
            embedder_fingerprint=dict(fp) if isinstance(fp, Mapping) else None,
            lane_fetch=int(data["lane_fetch"]),
            bfs_depth=int(data["bfs_depth"]),
            query_count=int(data["query_count"]),
            created_at=str(data["created_at"]),
            schema_version=int(data.get("schema_version", 1)),
        )


class CandidateCache:
    """In-memory and on-disk store of per-query candidate pools and qrels."""

    def __init__(
        self,
        queries: Mapping[str, Sequence[CandidateRow]],
        *,
        qrels: Mapping[str, Mapping[str, int]] | None = None,
        manifest: CacheManifest | None = None,
    ) -> None:
        self._queries: dict[str, list[CandidateRow]] = {
            str(qid): list(rows) for qid, rows in queries.items()
        }
        self._qrels: dict[str, dict[str, int]] = {}
        for qid, rows in self._queries.items():
            derived = {r.doc_id: int(r.grade) for r in rows if int(r.grade) > 0}
            if qrels and qid in qrels:
                for doc_id, grade in qrels[qid].items():
                    if int(grade) > 0:
                        derived[str(doc_id)] = int(grade)
            self._qrels[qid] = derived
        if qrels:
            for qid, qmap in qrels.items():
                if qid not in self._qrels:
                    self._qrels[str(qid)] = {
                        str(d): int(g) for d, g in qmap.items() if int(g) > 0
                    }

        self._manifest = manifest or CacheManifest(
            dataset="in_memory",
            dataset_sha="0" * 64,
            embedder_fingerprint=None,
            lane_fetch=max((len(r) for r in self._queries.values()), default=0),
            bfs_depth=3,
            query_count=len(self._queries),
            created_at=utc_key(utc_now()),
        )

    @property
    def manifest(self) -> CacheManifest:
        return self._manifest

    def query_ids(self) -> list[str]:
        return sorted(self._queries.keys())

    def rows(self, query_id: str) -> list[CandidateRow]:
        return list(self._queries.get(query_id, ()))

    def qrel(self, query_id: str) -> dict[str, int]:
        return dict(self._qrels.get(query_id, {}))

    def qrels(self) -> dict[str, dict[str, int]]:
        return {qid: dict(qmap) for qid, qmap in self._qrels.items()}

    def save(self, cache_dir: Path | str) -> Path:
        """Persist `manifest.json` and `candidates.jsonl` to `cache_dir`."""
        out = Path(cache_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest_path = out / "manifest.json"
        rows_path = out / "candidates.jsonl"
        manifest_path.write_text(
            json.dumps(self._manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with rows_path.open("w", encoding="utf-8") as fh:
            for qid in self.query_ids():
                record = {
                    "query_id": qid,
                    "qrel": self._qrels.get(qid, {}),
                    "candidates": [r.to_dict() for r in self._queries[qid]],
                }
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        return out

    @classmethod
    def load(cls, cache_dir: Path | str) -> CandidateCache:
        """Load a :class:`CandidateCache` from `cache_dir`."""
        root = Path(cache_dir)
        manifest_path = root / "manifest.json"
        rows_path = root / "candidates.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing candidate cache manifest: {manifest_path}")
        if not rows_path.is_file():
            raise FileNotFoundError(f"missing candidate cache rows: {rows_path}")
        manifest = CacheManifest.from_dict(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        queries: dict[str, list[CandidateRow]] = {}
        qrels: dict[str, dict[str, int]] = {}
        for line in rows_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            qid = str(payload["query_id"])
            queries[qid] = [
                CandidateRow.from_dict(item) for item in payload.get("candidates", [])
            ]
            raw_qrel = payload.get("qrel", {})
            if isinstance(raw_qrel, Mapping):
                qrels[qid] = {
                    str(doc_id): int(grade)
                    for doc_id, grade in raw_qrel.items()
                    if int(grade) > 0
                }
        return cls(queries, qrels=qrels, manifest=manifest)


def _days_since_created(created_at: str, now: datetime) -> float:
    try:
        dt = parse_utc(created_at)
    except ValueError:
        return 0.0
    delta = (parse_utc(now) - dt).total_seconds() / 86400.0
    return max(0.0, delta)


def _event_time_in_range(
    event_time: str | None, since: str | None, until: str | None
) -> bool:
    if since is None and until is None:
        return True
    if event_time is None:
        return False
    et = utc_key(event_time)
    if since is not None and et < utc_key(since):
        return False
    if until is not None and et > utc_key(until):
        return False
    return True


def _retrieve_query_pool(
    *,
    db: MindDB,
    store: FrameStore,
    embedder: Embedder,
    kg: KnowledgeGraph | None,
    cfg: AppConfig,
    spec: QuerySpec,
    lane_fetch: int,
    now: datetime,
    content_to_dias: Mapping[str, tuple[str, ...]] | None = None,
) -> list[CandidateRow]:
    sanitized = sanitize(spec.query)
    if not sanitized:
        fts_ids = db.fts.search_like(spec.query, int(lane_fetch))
    else:
        fts_ids = db.fts.search(sanitized, int(lane_fetch))
    eligible_fts = store.filter_eligible_ids(fts_ids)
    fts_ids = [fid for fid in fts_ids if fid in eligible_fts]

    qvec = embedder.embed(spec.query)
    vec_ids: list[int] = []
    chunk_hits = db.chunk_vec.search(qvec, int(lane_fetch))
    if chunk_hits:
        chunk_ids = [int(cid) for cid, _dist in chunk_hits]
        chunk_to_frame = store.resolve_chunk_frames(chunk_ids)
        seen: set[int] = set()
        for cid in chunk_ids:
            fid = chunk_to_frame.get(cid)
            if fid is not None and fid not in seen:
                seen.add(fid)
                vec_ids.append(fid)
        if vec_ids:
            eligible_vec = store.filter_eligible_ids(vec_ids)
            vec_ids = [fid for fid in vec_ids if fid in eligible_vec]
    if not vec_ids:
        frame_hits = db.frame_vec.search(qvec, int(lane_fetch))
        raw_frame_ids = [int(fid) for fid, _dist in frame_hits]
        eligible_vec = store.filter_eligible_ids(raw_frame_ids)
        vec_ids = [fid for fid in raw_frame_ids if fid in eligible_vec]

    fts_rank_map = {fid: rank for rank, fid in enumerate(fts_ids)}
    vec_rank_map = {fid: rank for rank, fid in enumerate(vec_ids)}
    all_fids = list(dict.fromkeys([*fts_ids, *vec_ids]))

    ctx_map: dict[int, int] = {}
    if kg is not None:
        try:
            ctx_map = kg.contextual(spec.query, cfg.kg, depth=cfg.scoring.bfs_depth)
        except Exception:
            ctx_map = {}

    beyond_depth = cfg.scoring.bfs_depth + 1
    frames_by_id = store.get_frames(all_fids)
    qrel = spec.qrel()
    rows: list[CandidateRow] = []

    for fid in all_fids:
        frame = frames_by_id.get(fid)
        if frame is None or not is_recall_eligible(frame.trust):
            continue
        if not _event_time_in_range(frame.event_time, spec.since, spec.until):
            continue
        dia_id = _resolve_dia_id(frame, content_to_dias, qrel)
        doc_id, grade = _resolve_doc_id_and_grade(frame, dia_id, qrel)
        rows.append(
            CandidateRow(
                doc_id=doc_id,
                fts_rank=fts_rank_map.get(fid),
                vec_rank=vec_rank_map.get(fid),
                days_since_created=_days_since_created(frame.created_at, now),
                access_count=frame.access_count,
                bfs_distance=ctx_map.get(fid, beyond_depth),
                importance_level=frame.importance,
                grade=grade,
                frame_id=frame.id,
                dia_id=dia_id,
            )
        )
    return rows


def _resolve_dia_id(
    frame: Frame,
    content_to_dias: Mapping[str, tuple[str, ...]] | None,
    qrel: Mapping[str, int],
) -> str | None:
    if content_to_dias is None:
        return None
    if ":session:" not in frame.gop_id or frame.source != "import":
        return None
    dias = content_to_dias.get(frame.content)
    if not dias:
        return None
    return next((d for d in dias if d in qrel), dias[0])


def _resolve_doc_id_and_grade(
    frame: Frame,
    dia_id: str | None,
    qrel: Mapping[str, int],
) -> tuple[str, int]:
    if dia_id is not None and dia_id in qrel:
        return dia_id, int(qrel[dia_id])
    frame_key = f"frame:{frame.id}"
    if frame_key in qrel:
        return frame_key, int(qrel[frame_key])
    id_str = str(frame.id)
    if id_str in qrel:
        return id_str, int(qrel[id_str])
    if dia_id is not None:
        return dia_id, int(qrel.get(dia_id, 0))
    return frame_key, 0


def build_cache_for_db(
    db: MindDB,
    store: FrameStore,
    cfg: AppConfig,
    queries: Sequence[QuerySpec],
    *,
    embedder: Embedder | None = None,
    kg: KnowledgeGraph | None = None,
    lane_fetch: int = 100,
    clock: Clock | None = None,
    dataset_name: str = "custom",
    dataset_sha: str | None = None,
) -> CandidateCache:
    """Build a :class:`CandidateCache` against a single :class:`MindDB`."""
    resolved_embedder = embedder or store.embedder
    resolved_kg = kg or KnowledgeGraph(db)
    resolved_clock = clock or utc_now
    pool_by_query: dict[str, list[CandidateRow]] = {}
    qrels_by_query: dict[str, dict[str, int]] = {}

    for spec in queries:
        qrel = spec.qrel()
        if not qrel:
            continue
        now = parse_utc(spec.clock_iso) if spec.clock_iso else resolved_clock()
        pool_by_query[spec.query_id] = _retrieve_query_pool(
            db=db,
            store=store,
            embedder=resolved_embedder,
            kg=resolved_kg,
            cfg=cfg,
            spec=spec,
            lane_fetch=lane_fetch,
            now=now,
        )
        qrels_by_query[spec.query_id] = qrel

    fp = stored_fingerprint(db) or fingerprint_of(resolved_embedder)
    sha = dataset_sha or hashlib.sha256(
        json.dumps(
            [(q.query_id, q.query, sorted(q.qrel().items())) for q in queries],
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    manifest = CacheManifest(
        dataset=dataset_name,
        dataset_sha=sha,
        embedder_fingerprint=fp.as_dict(),
        lane_fetch=int(lane_fetch),
        bfs_depth=int(cfg.scoring.bfs_depth),
        query_count=len(pool_by_query),
        created_at=utc_key(resolved_clock()),
    )
    return CandidateCache(pool_by_query, qrels=qrels_by_query, manifest=manifest)


def build_cache_for_benchmark(
    cfg: AppConfig,
    *,
    db_dir: Path | str,
    dataset_name: str = "locomo",
    sample_manifest: Path | str | None = None,
    lane_fetch: int = 100,
    embedder: Embedder | None = None,
    clock: Clock | None = None,
) -> CandidateCache:
    """Build a :class:`CandidateCache` over per-conversation benchmark `.mind` DBs."""
    from gluomem.bench.adapters import make_adapter
    from gluomem.bench.adapters.locomo import load_sample_manifest, stratified_sample
    from gluomem.embedding.providers import make_embedder

    adapter = make_adapter(dataset_name, cfg)
    if sample_manifest is not None:
        samples = load_sample_manifest(Path(sample_manifest), adapter=adapter)
    else:
        all_samples = adapter.load()
        try:
            samples = stratified_sample(
                all_samples,
                per_category=cfg.benchmark.sample.per_category,
                categories=cfg.benchmark.sample.categories,
                seed=cfg.benchmark.sample.seed,
            )
        except Exception:
            samples = list(all_samples)

    content_to_dias = adapter.content_to_dias()
    resolved_embedder = embedder or make_embedder(cfg)
    resolved_clock = clock or utc_now
    now = resolved_clock()

    root = Path(db_dir)
    pool_by_query: dict[str, list[CandidateRow]] = {}
    qrels_by_query: dict[str, dict[str, int]] = {}
    by_conv: dict[str, list[Any]] = {}
    for s in samples:
        if not s.evidence:
            continue
        by_conv.setdefault(s.conv_id, []).append(s)

    fp_dict: dict[str, Any] | None = None
    for conv_id, conv_samples in sorted(by_conv.items()):
        db_path = root / f"{conv_id}.mind"
        if not db_path.is_file():
            raise FileNotFoundError(
                f"no memory DB for conversation '{conv_id}' at {db_path}"
            )
        db = MindDB(db_path, dim=cfg.storage.vector_dim)
        try:
            store = FrameStore(db, embedder=resolved_embedder, chunking=cfg.chunking)
            kg = KnowledgeGraph(db)
            if fp_dict is None:
                fp = stored_fingerprint(db) or fingerprint_of(resolved_embedder)
                fp_dict = fp.as_dict()
            for s in conv_samples:
                spec = QuerySpec(
                    query_id=s.id,
                    query=s.question,
                    relevant_doc_ids=tuple(s.evidence),
                    conv_id=s.conv_id,
                )
                pool_by_query[s.id] = _retrieve_query_pool(
                    db=db,
                    store=store,
                    embedder=resolved_embedder,
                    kg=kg,
                    cfg=cfg,
                    spec=spec,
                    lane_fetch=lane_fetch,
                    now=now,
                    content_to_dias=content_to_dias,
                )
                qrels_by_query[s.id] = spec.qrel()
        finally:
            db.close()

    manifest = CacheManifest(
        dataset=dataset_name,
        dataset_sha=adapter.sha,
        embedder_fingerprint=fp_dict,
        lane_fetch=int(lane_fetch),
        bfs_depth=int(cfg.scoring.bfs_depth),
        query_count=len(pool_by_query),
        created_at=utc_key(now),
    )
    return CandidateCache(pool_by_query, qrels=qrels_by_query, manifest=manifest)
