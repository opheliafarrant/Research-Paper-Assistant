"""Local directories and ingestion history."""

import os
from tempfile import NamedTemporaryFile
from research_assistant import config


def prepare_storage() -> None:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    config.PDF_DIR.mkdir(parents=True, exist_ok=True)


def record_ingested(paper_ids: list[str]) -> None:
    """Atomically append paper IDs to INGESTED_FILE without duplicating existing entries."""
    if not paper_ids:
        return
    path = config.INGESTED_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[str] = []
    seen: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            pid = line.strip()
            if pid and pid not in seen:
                seen.add(pid)
                existing.append(pid)
    changed = False
    for pid in paper_ids:
        clean = pid.strip()
        if clean and clean not in seen:
            seen.add(clean)
            existing.append(clean)
            changed = True
    if not changed and path.exists():
        return
    content = "\n".join(existing) + ("\n" if existing else "")
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=".ingested-", delete=False) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, path)
