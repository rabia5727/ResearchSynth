# ResearchSynth

An autonomous multi-agent system that runs literature reviews as an adaptive
loop: search -> read -> compare -> synthesize -> check coverage -> (if thin)
search again with a better strategy. See the team's implementation plan doc
for the full design and the 2-day build schedule.

## Setup

```
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
copy .env.example .env        # then fill in GEMINI_API_KEY or GROQ_API_KEY
```

`LLM_PROVIDER=mock` in `.env` needs no key at all - use it to build and test
your agent before real keys are wired up for everyone.

## Layout

```
state/    - the frozen data contracts every agent reads/writes (schemas.py)
tools/    - shared building blocks (llm.py, pdf_parser.py, ...)
agents/   - one file per agent (searcher, pdf_reader, ...)
tests/    - fixtures + smoke tests
scripts/  - stand-alone demo/smoke-test entry points
```

## Branch convention

One branch per agent: `agent-1-searcher`, `agent-2-pdf-reader`, etc. Land on
`main` once your agent runs against the frozen schemas in `state/schemas.py`.

**Do not change those schemas without telling the team first** - every other
agent is building against them right now.
