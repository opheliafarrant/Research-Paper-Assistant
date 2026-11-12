"""Exercise real PDF parsing and both indexes in an isolated temporary library."""

from pathlib import Path
import shutil
import tempfile

from research_assistant import config

source_pdf = config.PDF_DIR / "2611.09892.pdf"
if not source_pdf.is_file():
    raise SystemExit(f"Expected existing test paper: {source_pdf}")

with tempfile.TemporaryDirectory(prefix="rpa-ingestion-test-") as temp:
    root = Path(temp)
    config.PDF_DIR = root / "pdfs"
    config.CACHE_DIR = root / "cache"
    config.INGESTED_FILE = root / "ingested.txt"
    config.PDF_DIR.mkdir()
    shutil.copyfile(source_pdf, config.PDF_DIR / source_pdf.name)

    # Imports follow the isolated path setup: index paths are bound at import time.
    from research_assistant.application import ingestion
    from research_assistant.domain.contracts import IngestionDependencies
    from research_assistant.infrastructure import indexes, local_store, pdf

    paper = {"paper_id": "2611.09892", "title": "On Agentic Manipulation Policies",
             "authors": [], "date": "", "abstract": "", "pdf_url": ""}
    deps = IngestionDependencies(
        prepare=local_store.prepare_storage, fetch=lambda: [paper],
        parse=pdf.parse_paper, add_chunks=indexes.add_chunks, record=local_store.record_ingested,
    )
    ingestion.run(dependencies=deps)
    dense, sparse = indexes.load_indexes()
    assert dense is not None and sparse is not None
    assert dense.index.ntotal == len(sparse.chunks) > 0
    assert config.INGESTED_FILE.read_text().splitlines() == [paper["paper_id"]]
    assert all(chunk["paper_id"] == paper["paper_id"] for chunk in dense.meta)
    print(f"ISOLATED_PAPER={paper['paper_id']} FAISS={dense.index.ntotal} BM25={len(sparse.chunks)} HISTORY=1")
