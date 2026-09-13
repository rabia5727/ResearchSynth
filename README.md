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

## Deploying the UI (Streamlit Community Cloud)

The app is ready to deploy as-is - `ui/app.py` is the entry point, `requirements.txt`
has every dependency, and `.env` (with real secrets) is correctly gitignored so
nothing sensitive ever gets pushed. Secrets get set separately in Streamlit
Cloud's own dashboard instead.

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with
   the GitHub account that owns (or was added as a collaborator to) this repo.
2. Click **New app** -> pick this repo, branch `main`, main file path `ui/app.py`.
3. Under **Advanced settings**, set the Python version to **3.11** (the code
   uses `X | None` union syntax and built-in generic types, which need 3.10+).
4. In the same Advanced settings screen, paste this into the **Secrets** box,
   filling in your real values (same ones as your local `.env`):

   ```toml
   LLM_PROVIDER = "gemini"

   GEMINI_API_KEY = "your-real-key-here"
   GEMINI_MODEL = "gemini-3.6-flash"

   GROQ_API_KEY = ""
   GROQ_MODEL = "llama-3.3-70b-versatile"

   PINECONE_API_KEY = "your-real-key-here"
   PINECONE_INDEX_NAME = "researchsynth-tensions"
   ```

   Streamlit Cloud exposes everything in this box as both `st.secrets` and as
   regular environment variables - the app already reads `os.environ`
   directly (see `tools/llm.py`), so no code changes are needed either way.
5. Click **Deploy**. First build takes a couple of minutes (installing
   PyMuPDF, LangGraph, etc.).

**Notes:**
- A real research run can take several minutes if arXiv/Semantic Scholar are
  rate-limiting (the UI now shows live per-term progress so this reads as
  "working," not "frozen" - see the "Cycle N" status box).
- The Contradiction Detector needs a **real Gemini + Pinecone index already
  created** (dimension 3072, matching `gemini-embedding-001`'s output) to run
  for real; without both secrets set, the app degrades gracefully and skips
  tension detection for that run instead of failing.
- Gemini's free tier caps at **20 requests/day** per project/model - heavy
  testing will exhaust it; the app now stops cleanly with a clear reason
  when that happens instead of crashing.
