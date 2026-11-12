"""Regression tests for cross-page citation spans and crash-safe index persistence."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import faiss
import numpy as np

from research_assistant import config
from research_assistant.application import ingestion
from research_assistant.domain.citations import (
    chunk_pages,
    format_chunk_text_with_pages,
    format_page_range,
    resolve_chunk_page,
)
from research_assistant.domain.contracts import IngestionDependencies
from research_assistant.infrastructure import indexes, local_store, pdf


class CrossPageCitationTests(unittest.TestCase):
    def test_chunk_text_with_spans_tracks_multi_page_offsets(self):
        pages = [
            (3, "3.1 Outlier Behavior\nFirst decoding sentences stay on page three. "),
            (4, "Over the first 30 decoding steps activation std is 20% higher on page four."),
        ]
        chunks = pdf._chunk_text_with_spans(pages, chunk_size=200, overlap=20)
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk["page"], 3)
        self.assertEqual(chunk["page_end"], 4)
        self.assertEqual(chunk_pages(chunk), (3, 4))
        self.assertEqual(format_page_range(chunk), "3-4")

        spans = chunk["page_spans"]
        self.assertEqual([s["page"] for s in spans], [3, 4])
        p3_seg = chunk["text"][spans[0]["start"] : spans[0]["end"]]
        p4_seg = chunk["text"][spans[1]["start"] : spans[1]["end"]]
        self.assertIn("First decoding sentences", p3_seg)
        self.assertIn("20% higher on page four", p4_seg)

        annotated = format_chunk_text_with_pages(chunk)
        self.assertIn("[Page 3]", annotated)
        self.assertIn("[Page 4]", annotated)
        self.assertEqual(resolve_chunk_page(chunk, "20% higher decoding steps"), 4)
        self.assertEqual(resolve_chunk_page(chunk, "First decoding sentences"), 3)

    def test_legacy_chunk_without_page_spans_remains_compatible(self):
        legacy = {"paper_id": "p", "chunk_idx": 0, "page": 7, "text": "Legacy single page."}
        self.assertEqual(chunk_pages(legacy), (7, 7))
        self.assertEqual(format_page_range(legacy), "7")
        self.assertEqual(format_chunk_text_with_pages(legacy), "Legacy single page.")
        self.assertEqual(resolve_chunk_page(legacy, "single page"), 7)


class IndexGovernanceTests(unittest.TestCase):
    def _mock_embedder(self):
        return (
            patch.object(indexes, "_get_embedder", return_value=SimpleNamespace(get_embedding_dimension=lambda: 2)),
            patch.object(indexes, "_embed", side_effect=lambda texts: np.array([[1.0, 0.0] for _ in texts], dtype="float32")),
        )

    def test_add_chunks_is_idempotent_across_interrupted_ingestion_retries(self):
        with tempfile.TemporaryDirectory(prefix="rpa-gov-retry-") as temp_dir:
            root = Path(temp_dir)
            cache = root / "cache"
            history = root / "ingested.txt"
            paper = {"paper_id": "2611.99999", "title": "Retry Test"}
            chunks = [
                {"paper_id": "2611.99999", "chunk_idx": 0, "text": "alpha chunk zero", "title": "Retry Test", "page": 1},
                {"paper_id": "2611.99999", "chunk_idx": 1, "text": "beta chunk one", "title": "Retry Test", "page": 2},
            ]

            def fetch():
                seen = set(history.read_text(encoding="utf-8").splitlines()) if history.exists() else set()
                return [] if paper["paper_id"] in seen else [paper]

            def crash_on_record(_ids):
                raise RuntimeError("simulated interruption before history update")

            p1, p2 = self._mock_embedder()
            with (
                patch.object(config, "CACHE_DIR", cache),
                patch.object(config, "PDF_DIR", root / "pdfs"),
                patch.object(config, "INGESTED_FILE", history),
                p1,
                p2,
            ):
                common = dict(
                    prepare=local_store.prepare_storage,
                    fetch=fetch,
                    parse=lambda _p: list(chunks),
                    add_chunks=indexes.add_chunks,
                )
                ingestion.run(dependencies=IngestionDependencies(**common, record=crash_on_record))
                fi, bi = indexes.load_indexes()
                self.assertEqual(fi.index.ntotal, 2)
                self.assertEqual(len(bi.chunks), 2)
                self.assertFalse(history.exists())

                # Re-running ingestion after failed record must NOT duplicate chunks
                ingestion.run(dependencies=IngestionDependencies(**common, record=local_store.record_ingested))
                # Explicit duplicate add_chunks call must also skip existing keys
                indexes.add_chunks(chunks)

                fi2, bi2 = indexes.load_indexes()
                self.assertEqual(fi2.index.ntotal, 2)
                self.assertEqual(len(bi2.chunks), 2)
                self.assertEqual(history.read_text(encoding="utf-8").splitlines(), ["2611.99999"])

    def test_mid_commit_crash_rolls_forward_from_verified_staging(self):
        with tempfile.TemporaryDirectory(prefix="rpa-gov-swap-") as temp_dir:
            cache = Path(temp_dir) / "cache"
            c1 = {"paper_id": "p1", "chunk_idx": 0, "text": "initial stable", "title": "P1", "page": 1}
            c2 = {"paper_id": "p2", "chunk_idx": 0, "text": "second committed", "title": "P2", "page": 1}
            p1, p2 = self._mock_embedder()
            with patch.object(config, "CACHE_DIR", cache), p1, p2:
                indexes.add_chunks([c1])

                real_install = indexes._install_staged_file
                calls = 0

                def fail_before_bm25(src, dst, tx_id):
                    nonlocal calls
                    calls += 1
                    if dst.name == "bm25.pkl":
                        raise RuntimeError("simulated power loss mid-commit before bm25 swap")
                    return real_install(src, dst, tx_id)

                with patch.object(indexes, "_install_staged_file", side_effect=fail_before_bm25):
                    with self.assertRaises(RuntimeError):
                        indexes.add_chunks([c2])

                # Before recovery runs explicitly, FAISS and BM25 differed on disk; load_indexes reconciles both
                fi, bi = indexes.load_indexes()
                self.assertEqual(fi.index.ntotal, 2)
                self.assertEqual(len(bi.chunks), 2)
                self.assertFalse((cache / ".commit_tx.json").exists())

    def test_mid_commit_crash_with_corrupted_staging_rolls_back_to_prior(self):
        with tempfile.TemporaryDirectory(prefix="rpa-gov-rb-") as temp_dir:
            cache = Path(temp_dir) / "cache"
            c1 = {"paper_id": "p1", "chunk_idx": 0, "text": "initial stable", "title": "P1", "page": 1}
            c2 = {"paper_id": "p2", "chunk_idx": 0, "text": "corrupting attempt", "title": "P2", "page": 1}
            p1, p2 = self._mock_embedder()
            with patch.object(config, "CACHE_DIR", cache), p1, p2:
                indexes.add_chunks([c1])

                real_install = indexes._install_staged_file

                def corrupt_stage_and_fail(src, dst, tx_id):
                    if dst.name == "bm25.pkl":
                        # Corrupt staged bm25.pkl and simulate interruption mid-swap
                        src.write_bytes(b"corrupted")
                        raise RuntimeError("simulated corruption + crash before bm25 swap")
                    return real_install(src, dst, tx_id)

                with patch.object(indexes, "_install_staged_file", side_effect=corrupt_stage_and_fail):
                    with self.assertRaises(RuntimeError):
                        indexes.add_chunks([c2])

                fi, bi = indexes.load_indexes()
                self.assertEqual(fi.index.ntotal, 1)
                self.assertEqual(len(bi.chunks), 1)
                self.assertEqual(bi.chunks[0]["paper_id"], "p1")
                self.assertFalse((cache / ".commit_tx.json").exists())

    def test_rebuild_deduplicated_indexes_removes_duplicates_and_enriches_spans(self):
        with tempfile.TemporaryDirectory(prefix="rpa-gov-dedup-") as temp_dir:
            cache = Path(temp_dir) / "cache"
            cache.mkdir(parents=True)
            with patch.object(config, "CACHE_DIR", cache):
                dup_chunks = [
                    {"paper_id": "p1", "chunk_idx": 0, "text": "unique zero", "title": "P1", "page": 1},
                    {"paper_id": "p1", "chunk_idx": 0, "text": "unique zero", "title": "P1", "page": 1},
                    {"paper_id": "p1", "chunk_idx": 1, "text": "unique one", "title": "P1", "page": 2},
                ]
                fi = indexes.FAISSIndex(index=faiss.IndexFlatIP(2))
                fi.add(np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype="float32"), dup_chunks)
                fi.save()
                bm = indexes.BM25Index.build([indexes._tokenize(c["text"]) for c in dup_chunks], dup_chunks)
                bm.save()

                overrides = {
                    ("p1", 1): {
                        "text": "unique one",
                        "page": 2,
                        "page_end": 3,
                        "page_spans": [{"page": 2, "start": 0, "end": 6}, {"page": 3, "start": 6, "end": 10}],
                    }
                }
                manifest = indexes.rebuild_deduplicated_indexes(chunk_overrides=overrides)
                self.assertEqual(manifest["vector_count"], 2)
                self.assertEqual(manifest["unique_keys"], 2)

                clean_fi, clean_bm = indexes.load_indexes()
                self.assertEqual(clean_fi.index.ntotal, 2)
                self.assertEqual(len(clean_bm.chunks), 2)
                self.assertEqual(clean_fi.meta[1]["page_end"], 3)
                self.assertTrue((cache / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
