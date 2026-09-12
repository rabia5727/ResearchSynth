from pydantic import BaseModel
from typing import Literal

class PaperRecord(BaseModel):
    id: str
    source: Literal["arxiv", "semantic_scholar", "pubmed"]
    title: str
    authors: list[str]
    year: int | None
    url: str
    pdf_accessible: bool
    subtopic_tags: list[str] = []

class ExtractedFinding(BaseModel):
    id: str
    paper_id: str
    claim: str
    method: str | None
    dataset: str | None
    metric: str | None
    value: str | None
    limitations: str | None
    section_source: str

class ContradictionPair(BaseModel):
    finding_a_id: str
    finding_b_id: str
    relation: Literal[
        "contradicts",
        "partial_consensus",
        "methodological_divergence"
    ]
    score: float
    explanation: str
    resolved: bool = False

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
    subtopics: list[str]
    cycle_n: int = 0
    max_cycles: int = 2
    papers: list[PaperRecord] = []
    findings: list[ExtractedFinding] = []
    tensions: list[ContradictionPair] = []
    strategy_log: list[StrategyMemoryEntry] = []
    coverage_score: float = 0.0
    stop_reason: str | None = None
