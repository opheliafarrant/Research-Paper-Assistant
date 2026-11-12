"""Reproduce an interrupted-ingestion retry in an isolated temporary library.

Run from the project root: python -B -m evaluation.reproduce_ingestion_retry
No existing indexes, PDFs, or ingestion history are read or modified.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from research_assistant import config
from research_assistant.application.ingestion import run
from research_assistant.domain.contracts import IngestionDependencies
from research_assistant.infrastructure import indexes, local_store


def main():
    with TemporaryDirectory(prefix="rpa-ingest-repro-") as temporary:
        root = Path(temporary)
        cache, pdf, history = root / "cache", root / "pdfs", root / "ingested.txt"
        paper = {"paper_id": "synthetic-1", "title": "Synthetic paper"}
        chunk = {"paper_id": "synthetic-1", "chunk_idx": 0, "text": "evidence synthetic test",
                 "title": "Synthetic paper", "page": 1}

        def fetch():
            seen = set(history.read_text().splitlines()) if history.exists() else set()
            return [] if paper["paper_id"] in seen else [paper]

        def interrupt_record(_ids):
            raise RuntimeError("simulated interruption before history write")

        with (patch.object(config, "CACHE_DIR", cache),
              patch.object(config, "PDF_DIR", pdf),
              patch.object(config, "INGESTED_FILE", history),
              patch.object(indexes, "_FAISS_INDEX_PATH", cache / "faiss.index"),
              patch.object(indexes, "_FAISS_META_PATH", cache / "faiss_meta.pkl"),
              patch.object(indexes, "_BM25_PATH", cache / "bm25.pkl"),
              patch.object(indexes, "_get_embedder", return_value=SimpleNamespace(get_embedding_dimension=lambda: 2)),
              patch.object(indexes, "_embed", side_effect=lambda texts: np.array([[1.0, 0.0] for _ in texts], dtype="float32"))):
            common = dict(prepare=local_store.prepare_storage, fetch=fetch,
                          parse=lambda _paper: [chunk], add_chunks=indexes.add_chunks)
            try:
                run(dependencies=IngestionDependencies(**common, record=interrupt_record))
            except RuntimeError as exc:
                if str(exc) != "simulated interruption before history write":
                    raise
                print("INTERRUPTED", str(exc))
            faiss_index, bm25_index = indexes.load_indexes()
            print("AFTER_FAILURE", faiss_index.index.ntotal, len(bm25_index.chunks), "HISTORY", history.exists())
            run(dependencies=IngestionDependencies(**common, record=local_store.record_ingested))
            faiss_index, bm25_index = indexes.load_indexes()
            keys = {(item["paper_id"], item["chunk_idx"]) for item in bm25_index.chunks}
            print("AFTER_RETRY", faiss_index.index.ntotal, len(bm25_index.chunks),
                  "UNIQUE", len(keys), "HISTORY", history.read_text().strip())
            assert faiss_index.index.ntotal == len(bm25_index.chunks) == 2 and len(keys) == 1


if __name__ == "__main__":
    main()
