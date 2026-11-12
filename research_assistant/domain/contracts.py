"""Contracts shared by application services and their adapters."""

from dataclasses import dataclass
from typing import Any, Callable, TypedDict


class Paper(TypedDict):
    paper_id: str
    title: str
    authors: list[str]
    date: str
    abstract: str
    pdf_url: str


class Chunk(TypedDict):
    paper_id: str
    title: str
    chunk_idx: int
    text: str
    page: int


class Answer(TypedDict):
    answer: str
    sources: list[dict]
    query: str


class Summary(TypedDict):
    text: str
    sources: list[dict]


@dataclass(frozen=True)
class ResearchDependencies:
    load_indexes: Callable[[], tuple]
    embed: Callable[[list[str]], Any]
    generate: Callable[..., str]


@dataclass(frozen=True)
class IngestionDependencies:
    prepare: Callable[[], None]
    fetch: Callable[[], list[dict]]
    parse: Callable[[dict], list[dict]]
    add_chunks: Callable[[list[dict]], None]
    record: Callable[[list[str]], None]
