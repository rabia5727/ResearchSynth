"""Frozen data contracts between agents.

These are the shapes every agent reads and writes - see the implementation
plan, section 3. Change something here only after telling the whole team;
everyone else is building against these exact field names right now.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class PaperRecord(BaseModel):
    id: str  # doi | arxiv:<id> | s2:<id>
    source: Literal["arxiv", "semantic_scholar", "pubmed"]  # pubmed: stretch
    title: str
    authors: list[str] = []
    year: int | None = None
    url: str
    pdf_accessible: bool = True
    subtopic_tags: list[str] = []


class ExtractedFinding(BaseModel):
    id: str
    paper_id: str
    claim: str
    method: str | None = None
    dataset: str | None = None
    metric: str | None = None
    value: str | None = None
    limitations: str | None = None
    section_source: str  # "results" | "methodology" | ...


class ContradictionPair(BaseModel):
    finding_a_id: str
    finding_b_id: str
    relation: Literal["contradicts", "partial_consensus", "methodological_divergence"]
    score: float  # 0-1 similarity/contradiction confidence
    explanation: str
    resolved: bool = False  # set true once a later cycle explains the split


class StrategyMemoryEntry(BaseModel):
    cycle_n: int
    query_terms: list[str]
    sources_queried: list[str]
    papers_yielded: int
    new_unique_findings: int
    refiner_decision: Literal["continue", "stop"]
    rationale: str


class CycleState(BaseModel):
    query: str
    subtopics: list[str] = []  # decomposed once, up front
    cycle_n: int = 0
    max_cycles: int = 2  # demo default; 3 is the configurable ceiling
    papers: list[PaperRecord] = []
    findings: list[ExtractedFinding] = []
    tensions: list[ContradictionPair] = []
    strategy_log: list[StrategyMemoryEntry] = []
    coverage_score: float = 0.0
    stop_reason: str | None = None
    latest_report: str | None = None  # Synthesis Writer's output, regenerated each cycle
