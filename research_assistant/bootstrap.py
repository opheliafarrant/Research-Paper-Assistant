"""Composition root: connect use cases to local adapters."""

from dataclasses import dataclass
from functools import partial
from typing import Callable

from research_assistant.application import ingestion, library, research
from research_assistant.domain.contracts import IngestionDependencies, ResearchDependencies
from research_assistant.infrastructure import arxiv, embedding, indexes, local_store, ollama, pdf, pdf_preview


@dataclass(frozen=True)
class Application:
    answer: Callable
    summarise: Callable
    ingest: Callable
    list_papers: Callable
    load_indexes: Callable
    resolve_paper_pdf: Callable
    pdf_page_count: Callable
    render_pdf_page: Callable
    summarise_with_sources: Callable | None = None


def build_application() -> Application:
    research_deps = ResearchDependencies(
        load_indexes=indexes.load_indexes, embed=embedding.embed, generate=ollama.generate,
    )
    ingestion_deps = IngestionDependencies(
        prepare=local_store.prepare_storage, fetch=arxiv.fetch_new_papers,
        parse=pdf.parse_paper, add_chunks=indexes.add_chunks,
        record=local_store.record_ingested,
    )
    return Application(
        answer=partial(research.answer, dependencies=research_deps),
        summarise=partial(research.summarise, dependencies=research_deps),
        summarise_with_sources=partial(research.summarise_with_sources, dependencies=research_deps),
        ingest=partial(ingestion.run, dependencies=ingestion_deps),
        list_papers=partial(library.list_papers, load_indexes=indexes.load_indexes),
        load_indexes=indexes.load_indexes,
        resolve_paper_pdf=pdf_preview.resolve_paper_pdf,
        pdf_page_count=pdf_preview.page_count,
        render_pdf_page=pdf_preview.render_page,
    )
