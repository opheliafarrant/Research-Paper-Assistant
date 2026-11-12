"""
Orchestrates the full ingestion pipeline:
    fetch  ->  parse  ->  index  ->  record

run() -> None
    Fetch new papers from arXiv parse each into chunks
    add them to the indexes and record their IDs in INGESTED_FILE
    so they are skipped on the next run
"""

import logging
from research_assistant.domain.contracts import IngestionDependencies
logger = logging.getLogger(__name__)



def run(*, dependencies: IngestionDependencies) -> dict[str, object]:
    dependencies.prepare()

    logger.info("Fetching new papers")
    papers = dependencies.fetch()


    if not papers:
        logger.info("No new papers found nothing to ingest")
        return {"fetched": 0, "ingested": [], "failed": []}

    logger.info("%d new paper(s) to ingest", len(papers))



    ingested_ids: list[str] = []
    failed_ids:   list[str] = []

    for paper in papers:
        pid = paper["paper_id"]
        logger.info("Processing [%s] %s", pid, paper["title"][:60])

        try:
            chunks = dependencies.parse(paper)
            if not chunks:
                logger.warning("No chunks produced for %s skipping", pid)
                failed_ids.append(pid)
                continue

            dependencies.add_chunks(chunks)
            dependencies.record([pid])
            ingested_ids.append(pid)
            logger.info("Indexed %d chunks for %s", len(chunks), pid)

        except Exception as exc:
            logger.error("Failed to ingest %s: %s", pid, exc, exc_info=True)
            failed_ids.append(pid)

    logger.info(
        "Ingestion complete  success: %d  failed: %d",
        len(ingested_ids),
        len(failed_ids),
    )

    if failed_ids:
        logger.warning("Failed paper IDs: %s", failed_ids)

    return {
        "fetched": len(papers),
        "ingested": ingested_ids,
        "failed": failed_ids,
    }


if __name__ == "__main__":
    from research_assistant.bootstrap import build_application
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt = "%H:%M:%S",
    )
    build_application().ingest()
