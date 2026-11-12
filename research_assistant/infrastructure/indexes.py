"""
Builds and persists the two search indexes used by the retriever:

  1. FAISS flat index  – dense semantic search via sentence-transformers
  2. BM25 index        – sparse keyword search via rank_bm25


add_chunks(chunks: list[dict]) -> None
    Embed chunks, add them to both indexes and persist to CACHE_DIR

load_indexes() -> tuple[FAISSIndex, BM25Index] | tuple[None, None]
    Load persisted indexes from CACHE_DIR and returns (None, None) if not found
"""


import hashlib
import json
import logging
import os
import pickle
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional
import numpy as np
from research_assistant import config
from research_assistant.infrastructure.embedding import embed as _embed, get_embedder as _get_embedder
logger = logging.getLogger(__name__)


# cache paths (respect both dynamic config.CACHE_DIR overrides and legacy constant patches)
_INITIAL_CACHE_DIR = config.CACHE_DIR
_DEFAULT_FAISS_INDEX_PATH = _INITIAL_CACHE_DIR / "faiss.index"
_DEFAULT_FAISS_META_PATH  = _INITIAL_CACHE_DIR / "faiss_meta.pkl"
_DEFAULT_BM25_PATH        = _INITIAL_CACHE_DIR / "bm25.pkl"

_FAISS_INDEX_PATH  = _DEFAULT_FAISS_INDEX_PATH
_FAISS_META_PATH   = _DEFAULT_FAISS_META_PATH
_BM25_PATH         = _DEFAULT_BM25_PATH


def _index_paths() -> tuple[Path, Path, Path, Path]:
    if (
        _FAISS_INDEX_PATH != _DEFAULT_FAISS_INDEX_PATH
        or _FAISS_META_PATH != _DEFAULT_FAISS_META_PATH
        or _BM25_PATH != _DEFAULT_BM25_PATH
    ):
        faiss_path = Path(_FAISS_INDEX_PATH)
        meta_path = Path(_FAISS_META_PATH)
        bm25_path = Path(_BM25_PATH)
    else:
        cache_dir = Path(config.CACHE_DIR)
        faiss_path = cache_dir / "faiss.index"
        meta_path = cache_dir / "faiss_meta.pkl"
        bm25_path = cache_dir / "bm25.pkl"
    return faiss_path, meta_path, bm25_path, faiss_path.parent / "manifest.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def _install_staged_file(src: Path, dst: Path, tx_id: str) -> None:
    """Copy a verified staged file to a same-dir temp file and atomically replace dst."""
    tmp_path = dst.with_name(f".{dst.name}.tmp-{tx_id}")
    with open(src, "rb") as src_f, open(tmp_path, "wb") as dst_f:
        shutil.copyfileobj(src_f, dst_f)
        dst_f.flush()
        os.fsync(dst_f.fileno())
    os.replace(tmp_path, dst)


def _files_match_checksums(targets: dict[str, Path], checksums: dict[str, str]) -> bool:
    return all(
        path.is_file() and checksums.get(name) == _file_sha256(path)
        for name, path in targets.items()
    )


def recover_pending_transaction(cache_dir: Optional[Path] = None) -> Optional[str]:
    """
    Reconcile an interrupted multi-file index commit if .commit_tx.json is present.
    Returns 'already_committed', 'rolled_forward', 'rolled_back', or None.
    """
    faiss_path, meta_path, bm25_path, manifest_path = _index_paths()
    root = cache_dir or faiss_path.parent
    tx_path = root / ".commit_tx.json"
    if not tx_path.exists():
        return None

    tx = json.loads(tx_path.read_text(encoding="utf-8"))
    staging_dir = Path(tx["staging_dir"])
    rollback_dir = Path(tx["rollback_dir"])
    target_map = {
        "faiss": Path(tx["targets"]["faiss"]),
        "meta": Path(tx["targets"]["meta"]),
        "bm25": Path(tx["targets"]["bm25"]),
    }
    target_manifest = tx["manifest"]
    target_checksums = target_manifest["checksums"]
    tx_id = tx["tx_id"]

    if _files_match_checksums(target_map, target_checksums) and Path(tx["manifest_path"]).is_file():
        outcome = "already_committed"
    else:
        staged_map = {
            "faiss": staging_dir / "faiss.index",
            "meta": staging_dir / "faiss_meta.pkl",
            "bm25": staging_dir / "bm25.pkl",
        }
        if _files_match_checksums(staged_map, target_checksums):
            for key, staged in staged_map.items():
                _install_staged_file(staged, target_map[key], tx_id)
            committed = dict(target_manifest, status="COMMITTED")
            _atomic_write_bytes(Path(tx["manifest_path"]), json.dumps(committed, indent=2).encode("utf-8"))
            outcome = "rolled_forward"
        elif tx.get("had_previous"):
            rb_checksums = tx["rollback_checksums"]
            rb_map = {
                "faiss": rollback_dir / "faiss.index",
                "meta": rollback_dir / "faiss_meta.pkl",
                "bm25": rollback_dir / "bm25.pkl",
            }
            if not _files_match_checksums(rb_map, rb_checksums):
                raise RuntimeError("Rollback snapshot corrupted during recovery")
            for key, rb_file in rb_map.items():
                _install_staged_file(rb_file, target_map[key], tx_id)
            rb_manifest = rollback_dir / "manifest.json"
            m_dst = Path(tx["manifest_path"])
            if rb_manifest.is_file():
                _install_staged_file(rb_manifest, m_dst, tx_id)
            elif m_dst.exists():
                m_dst.unlink()
            outcome = "rolled_back"
        else:
            for path in (*target_map.values(), Path(tx["manifest_path"])):
                if path.exists():
                    path.unlink()
            outcome = "rolled_back"

    tx_path.unlink(missing_ok=True)
    shutil.rmtree(staging_dir, ignore_errors=True)
    shutil.rmtree(rollback_dir, ignore_errors=True)
    return outcome


# FAISS index 
@dataclass
class FAISSIndex:
    index: object          
    meta:  list[dict] = field(default_factory = list)   

    def save(self) -> None:
        import faiss
        faiss_path, meta_path, _, _ = _index_paths()
        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("wb", dir=faiss_path.parent, prefix=".faiss.", delete=False) as f_tmp:
            f_tmp_path = f_tmp.name
        faiss.write_index(self.index, f_tmp_path)
        os.replace(f_tmp_path, faiss_path)
        _atomic_write_bytes(meta_path, pickle.dumps(self.meta))
        logger.info("FAISS index saved (%d vectors)", self.index.ntotal)



    @classmethod
    def load(cls) -> Optional["FAISSIndex"]:
        recover_pending_transaction()
        faiss_path, meta_path, _, _ = _index_paths()
        if not faiss_path.exists() or not meta_path.exists():
            return None
        import faiss
        index = faiss.read_index(str(faiss_path))
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        logger.info("FAISS index loaded (%d vectors)", index.ntotal)
        return cls(index=index, meta=meta)



    def add(self, vecs: np.ndarray, chunks: list[dict]) -> None:
        import faiss
        # Normalize for cosine similarity (IndexFlatIP on unit vectors = cosine)
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.meta.extend(chunks)



    def search(
        self,
        query_vec: np.ndarray,
        k: int = config.TOP_K,
        paper_id: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> list[dict]:
        """
        Return up to k chunk dicts
        If paper_id is given only chunks from that paper are returned
        Results below SCORE_THRESHOLD are discarded
        """
        import faiss
        if self.index.ntotal == 0:
            return []

        q = query_vec.copy().astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)

        # Over-fetch so we can filter by paper_id and threshold
        fetch_k = self.index.ntotal if paper_id else min(self.index.ntotal, k * 2)
        threshold = config.SCORE_THRESHOLD if min_score is None else min_score
        while True:
            scores, indices = self.index.search(q, fetch_k)
            results, seen = [], set()
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or score < threshold:
                    continue
                chunk = self.meta[idx].copy()
                if paper_id and chunk.get("paper_id") != paper_id:
                    continue
                key = (chunk.get("paper_id"), chunk["chunk_idx"]) if "chunk_idx" in chunk else ("index", int(idx))
                if key in seen:
                    continue
                seen.add(key)
                chunk["_faiss_score"] = float(score)
                results.append(chunk)
                if len(results) >= k:
                    return results
            if fetch_k >= self.index.ntotal:
                return results
            fetch_k = min(self.index.ntotal, max(fetch_k * 2, 1))

    def cosine_scores(self, query_vec: np.ndarray, chunks: list[dict]) -> dict[tuple, float]:
        """Compute exact cosine similarity against persisted L2-normalized chunk vectors."""
        import faiss

        if not chunks or self.index.ntotal == 0 or not self.meta:
            return {}
        target_keys = {
            (c.get("paper_id"), c.get("chunk_idx"))
            for c in chunks
            if "paper_id" in c and "chunk_idx" in c
        }
        if not target_keys:
            return {}

        q = query_vec.copy().astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        q_flat = q.reshape(-1)
        scores: dict[tuple, float] = {}
        for row_idx, meta in enumerate(self.meta):
            key = (meta.get("paper_id"), meta.get("chunk_idx"))
            if key in target_keys and key not in scores:
                vec = self.index.reconstruct(row_idx)
                scores[key] = float(np.dot(q_flat, vec))
                if len(scores) == len(target_keys):
                    break
        return scores


# BM25 index
@dataclass
class BM25Index:
    bm25:   object              # BM25Okapi instance
    tokens: list[list[str]]    
    chunks: list[dict]

    def save(self) -> None:
        _, _, bm25_path, _ = _index_paths()
        _atomic_write_bytes(bm25_path, pickle.dumps((self.bm25, self.tokens, self.chunks)))
        logger.info("BM25 index saved (%d documents)", len(self.chunks))



    @classmethod
    def load(cls) -> Optional["BM25Index"]:
        recover_pending_transaction()
        _, _, bm25_path, _ = _index_paths()
        if not bm25_path.exists():
            return None
        with open(bm25_path, "rb") as f:
            bm25, tokens, chunks = pickle.load(f)
        logger.info("BM25 index loaded (%d documents)", len(chunks))
        return cls(bm25 = bm25, tokens = tokens, chunks = chunks)



    @classmethod
    def build(cls, all_tokens: list[list[str]], all_chunks: list[dict]) -> "BM25Index":
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(all_tokens)
        return cls(bm25 = bm25, tokens = all_tokens, chunks = all_chunks)



    def search(
        self,
        query: str,
        k: int = config.TOP_K,
        paper_id: Optional[str] = None,
    ) -> list[dict]:
        """Return up to k chunk dicts ranked by BM25 score"""
        if not self.chunks:
            return []

        q_tokens = _tokenize(query)
        scores   = self.bm25.get_scores(q_tokens)
        ranked = sorted(
            ((score, i) for i, score in enumerate(scores) if score > 0),   # Pair each chunk with its score, filter, sort
            reverse=True,
        )

        results, seen = [], set()
        for score, i in ranked:
            chunk = self.chunks[i].copy()
            if paper_id and chunk.get("paper_id") != paper_id:
                continue
            key = (chunk.get("paper_id"), chunk["chunk_idx"]) if "chunk_idx" in chunk else ("index", i)
            if key in seen:
                continue
            seen.add(key)
            chunk["_bm25_score"] = float(score)
            results.append(chunk)
            if len(results) >= k:
                break
        return results
    

# Tokenizer 
def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric and drop empty tokens"""
    import re
    return [t for t in re.split(r"\W+", text.lower()) if t]


def _chunk_key(chunk: dict, fallback_idx: int) -> tuple:
    if "paper_id" in chunk and "chunk_idx" in chunk:
        return (chunk["paper_id"], chunk["chunk_idx"])
    return ("__pos__", fallback_idx)


def _fsync_path(path: Path) -> None:
    with open(path, "r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())


def commit_indexes(faiss_idx: FAISSIndex, bm25_idx: BM25Index) -> dict:
    """
    Atomically persist synchronized FAISS and BM25 indexes via staging,
    rollback snapshots, and a write-ahead transaction journal.
    """
    import faiss

    ntotal = int(faiss_idx.index.ntotal)
    if not (ntotal == len(faiss_idx.meta) == len(bm25_idx.chunks) == len(bm25_idx.tokens)):
        raise ValueError(
            f"Index length mismatch before commit: faiss={ntotal}, meta={len(faiss_idx.meta)}, "
            f"bm25_chunks={len(bm25_idx.chunks)}, bm25_tokens={len(bm25_idx.tokens)}"
        )
    faiss_keys = [_chunk_key(m, i) for i, m in enumerate(faiss_idx.meta)]
    bm25_keys = [_chunk_key(c, i) for i, c in enumerate(bm25_idx.chunks)]
    if faiss_keys != bm25_keys:
        raise ValueError("FAISS metadata and BM25 chunk key ordering diverged before commit")

    faiss_path, meta_path, bm25_path, manifest_path = _index_paths()
    cache_dir = faiss_path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    recover_pending_transaction(cache_dir)

    tx_id = uuid.uuid4().hex[:12]
    staging_dir = cache_dir / f".staging-{tx_id}"
    rollback_dir = cache_dir / f".rollback-{tx_id}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    staged_faiss = staging_dir / "faiss.index"
    staged_meta = staging_dir / "faiss_meta.pkl"
    staged_bm25 = staging_dir / "bm25.pkl"

    faiss.write_index(faiss_idx.index, str(staged_faiss))
    _fsync_path(staged_faiss)
    _atomic_write_bytes(staged_meta, pickle.dumps(faiss_idx.meta))
    _atomic_write_bytes(staged_bm25, pickle.dumps((bm25_idx.bm25, bm25_idx.tokens, bm25_idx.chunks)))

    target_checksums = {
        "faiss": _file_sha256(staged_faiss),
        "meta": _file_sha256(staged_meta),
        "bm25": _file_sha256(staged_bm25),
    }
    manifest = {
        "version": tx_id,
        "vector_count": ntotal,
        "chunk_count": len(bm25_idx.chunks),
        "unique_keys": len(set(faiss_keys)),
        "paper_ids": sorted({m["paper_id"] for m in faiss_idx.meta if "paper_id" in m}),
        "checksums": target_checksums,
        "status": "PREPARED",
    }
    _atomic_write_bytes(staging_dir / "manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))

    existing_targets = {"faiss": faiss_path, "meta": meta_path, "bm25": bm25_path}
    had_previous = all(p.is_file() for p in existing_targets.values())
    rollback_checksums: dict[str, str] = {}
    if had_previous:
        rollback_dir.mkdir(parents=True, exist_ok=True)
        rb_names = {"faiss": "faiss.index", "meta": "faiss_meta.pkl", "bm25": "bm25.pkl"}
        for key, src in existing_targets.items():
            dst = rollback_dir / rb_names[key]
            shutil.copyfile(src, dst)
            _fsync_path(dst)
            rollback_checksums[key] = _file_sha256(dst)
        if manifest_path.is_file():
            shutil.copyfile(manifest_path, rollback_dir / "manifest.json")
            _fsync_path(rollback_dir / "manifest.json")

    tx_record = {
        "tx_id": tx_id,
        "staging_dir": str(staging_dir),
        "rollback_dir": str(rollback_dir),
        "targets": {
            "faiss": str(faiss_path),
            "meta": str(meta_path),
            "bm25": str(bm25_path),
        },
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "had_previous": had_previous,
        "rollback_checksums": rollback_checksums,
    }
    tx_path = cache_dir / ".commit_tx.json"
    _atomic_write_bytes(tx_path, json.dumps(tx_record, indent=2).encode("utf-8"))

    _install_staged_file(staged_faiss, faiss_path, tx_id)
    _install_staged_file(staged_meta, meta_path, tx_id)
    _install_staged_file(staged_bm25, bm25_path, tx_id)

    committed = dict(manifest, status="COMMITTED")
    _atomic_write_bytes(manifest_path, json.dumps(committed, indent=2).encode("utf-8"))

    tx_path.unlink(missing_ok=True)
    shutil.rmtree(staging_dir, ignore_errors=True)
    shutil.rmtree(rollback_dir, ignore_errors=True)
    return committed


def rebuild_deduplicated_indexes(
    chunk_overrides: Optional[dict[tuple[str, int], dict]] = None,
) -> dict:
    """Rebuild existing persisted indexes with unique (paper_id, chunk_idx) entries."""
    import faiss

    recover_pending_transaction()
    faiss_idx = FAISSIndex.load()
    bm25_idx = BM25Index.load()
    if faiss_idx is None or bm25_idx is None:
        raise FileNotFoundError("Cannot rebuild indexes before both FAISS and BM25 exist")
    if faiss_idx.index.ntotal != len(faiss_idx.meta) or len(faiss_idx.meta) != len(bm25_idx.chunks):
        raise ValueError("Existing indexes have inconsistent row counts")

    seen: set[tuple] = set()
    clean_vecs: list[np.ndarray] = []
    clean_meta: list[dict] = []
    clean_tokens: list[list[str]] = []

    for i, meta in enumerate(faiss_idx.meta):
        key = _chunk_key(meta, i)
        if key in seen:
            continue
        seen.add(key)
        chunk = meta.copy()
        if chunk_overrides and key in chunk_overrides:
            override = chunk_overrides[key]
            if override.get("text") != chunk.get("text"):
                raise ValueError(f"Chunk text mismatch while enriching metadata for {key}")
            chunk.update(override)
        clean_vecs.append(faiss_idx.index.reconstruct(i))
        clean_meta.append(chunk)
        clean_tokens.append(_tokenize(chunk.get("text", "")))

    dim = faiss_idx.index.d
    new_index = faiss.IndexFlatIP(dim)
    if clean_vecs:
        vec_arr = np.asarray(clean_vecs, dtype=np.float32)
        faiss.normalize_L2(vec_arr)
        new_index.add(vec_arr)

    clean_faiss = FAISSIndex(index=new_index, meta=clean_meta)
    clean_bm25 = BM25Index.build(clean_tokens, clean_meta)
    return commit_indexes(clean_faiss, clean_bm25)


# Public API
def add_chunks(chunks: list[dict]) -> None:
    """
    Embed chunks and atomically add unique entries to both FAISS and BM25 indexes.
    Existing index data is preserved; duplicate (paper_id, chunk_idx) keys are skipped.
    """

    if not chunks:
        logger.warning("add_chunks called with empty list so nothing to do")
        return

    import faiss

    recover_pending_transaction()
    faiss_idx = FAISSIndex.load()
    bm25_idx = BM25Index.load()
    if faiss_idx is None:
        dim = _get_embedder().get_embedding_dimension()
        faiss_idx = FAISSIndex(index=faiss.IndexFlatIP(dim))
        logger.info("Created new FAISS index (dim = %d)", dim)
    if bm25_idx is None:
        bm25_idx = BM25Index(bm25=None, tokens=[], chunks=[])

    existing_keys = {
        (c["paper_id"], c["chunk_idx"])
        for c in faiss_idx.meta
        if "paper_id" in c and "chunk_idx" in c
    }
    unique_new: list[dict] = []
    batch_keys: set[tuple] = set()
    for chunk in chunks:
        if "paper_id" in chunk and "chunk_idx" in chunk:
            key = (chunk["paper_id"], chunk["chunk_idx"])
            if key in existing_keys or key in batch_keys:
                continue
            batch_keys.add(key)
        unique_new.append(chunk.copy())

    if not unique_new:
        logger.info("All %d chunk(s) already indexed; skipping duplicate insert", len(chunks))
        return

    texts = [c["text"] for c in unique_new]
    logger.info("Embedding %d unique chunk(s)", len(texts))
    vecs = _embed(texts)
    faiss_idx.add(vecs, unique_new)

    all_tokens = bm25_idx.tokens + [_tokenize(c["text"]) for c in unique_new]
    all_chunks = bm25_idx.chunks + unique_new
    new_bm25 = BM25Index.build(all_tokens, all_chunks)

    commit_indexes(faiss_idx, new_bm25)

    logger.info(
        "Indexes committed FAISS: %d vectors | BM25: %d documents",
        faiss_idx.index.ntotal,
        len(new_bm25.chunks),
    )



def load_indexes() -> tuple:
    """
    Load both indexes from CACHE_DIR after reconciling any interrupted commit.
    Returns:
    (FAISSIndex, BM25Index)  if both exist
    (None, None)             if neither has been built yet
    """
    recover_pending_transaction()
    faiss_idx = FAISSIndex.load()
    bm25_idx  = BM25Index.load()

    if faiss_idx is None or bm25_idx is None:
        logger.warning("One or both indexes not found run the ingester first")
        return None, None

    return faiss_idx, bm25_idx




# Test
if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO, format = "%(levelname)s  %(message)s")
    from research_assistant.infrastructure.arxiv import fetch_new_papers
    from research_assistant.infrastructure.pdf import parse_paper


    papers = fetch_new_papers()
    if not papers:
        print("No new papers so nothing to index")
    else:
        test_paper = papers[0]
        print(f"\nIndexing: [{test_paper['paper_id']}] {test_paper['title'][:60]}")
        chunks = parse_paper(test_paper)
        add_chunks(chunks)

        print("\n── Search smoke-test ──────────────────────────────────────")
        faiss_idx, bm25_idx = load_indexes()

        query = "attention mechanism transformer"
        print(f"Query: '{query}'\n")
        q_vec = _embed([query])


        faiss_results = faiss_idx.search(q_vec, k = 3)
        print(f"FAISS top-{len(faiss_results)}:")
        for r in faiss_results:
            print(f"  [{r['paper_id']} p{r['page']}] score = {r['_faiss_score']:.3f}  {r['text'][:80].strip()!r}")


        bm25_results = bm25_idx.search(query, k = 3)
        print(f"\nBM25  top-{len(bm25_results)}:")
        for r in bm25_results:
            print(f"  [{r['paper_id']} p{r['page']}] score = {r['_bm25_score']:.3f}  {r['text'][:80].strip()!r}")
