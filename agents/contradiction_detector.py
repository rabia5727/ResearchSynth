import os
from google import genai
from google.genai import types
from state.schemas import CycleState, ContradictionPair
from tools import vector_store
from tools.llm import LLMError, generate_json

def detect_contradictions(state: CycleState) -> dict:
    """
    Agent 3: Evaluates new findings against the existing corpus to detect contradictions, 
    methodological divergences, or partial consensus strictly within specific subtopics.
    """
    current_findings = state.findings
    if not current_findings:
        return {"tensions": []}

    # 1. Use a static namespace for local runs
    current_namespace = "active-research-run"

    # 2. If this is the start of a new session, wipe the old database records clean
    if state.cycle_n == 1:
        vector_store.clear_namespace(current_namespace)

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # Map paper IDs to their assigned subtopic tags for metadata filtering
    paper_subtopics = {paper.id: paper.subtopic_tags for paper in state.papers}

    # 3. Batch embed all new claims using Google GenAI (Defaults to 3072 dimensions)
    claims = [finding.claim for finding in current_findings]
    
    embedding_response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=claims,
        config=types.EmbedContentConfig(
            task_type="SEMANTIC_SIMILARITY"
        )
    )
    
    vectors = [emb.values for emb in embedding_response.embeddings]

    # 4. Upsert to Pinecone via the vector_store wrapper
    records_to_store = []
    for finding, vector in zip(current_findings, vectors):
        subtopics = paper_subtopics.get(finding.paper_id, ["general"])
        records_to_store.append({
            "id": finding.id,
            "values": vector,
            "metadata": {
                "subtopics": subtopics,
                "paper_id": finding.paper_id,
                "claim": finding.claim,
                "method": finding.method or "Not specified",
                "dataset": finding.dataset or "Not specified",
                "metric": finding.metric or "Not specified"
            }
        })
        
    vector_store.upsert_findings(records_to_store, namespace=current_namespace)

    new_tensions = []
    similarity_threshold = 0.75  
    processed_pairs = set()

    # 5. Search for candidate pairs and classify tensions
    for finding, vector in zip(current_findings, vectors):
        subtopics = paper_subtopics.get(finding.paper_id, ["general"])
        
        for subtopic in subtopics:
            matches = vector_store.search_within_subtopic(
                query_vector=vector, 
                subtopic=subtopic,
                namespace=current_namespace,
                top_k=5
            )
            
            for match in matches.get("matches", []):
                match_id = match["id"]
                score = match["score"]
                
                # Enforce the O(N^2) guardrail: skip self-matches, duplicates, and low scores
                pair_key = tuple(sorted([finding.id, match_id]))
                if match_id == finding.id or pair_key in processed_pairs or score < similarity_threshold:
                    continue
                    
                processed_pairs.add(pair_key)
                
                # LLM Pairwise Classification Prompt
                prompt = (
                    f"Analyze these two scientific findings within the subtopic '{subtopic}':\n\n"
                    f"Finding A ({finding.id}):\n"
                    f"- Claim: {finding.claim}\n"
                    f"- Method: {finding.method}\n"
                    f"- Metric: {finding.metric}\n\n"
                    f"Finding B ({match_id}):\n"
                    f"- Claim: {match['metadata']['claim']}\n"
                    f"- Method: {match['metadata']['method']}\n"
                    f"- Metric: {match['metadata']['metric']}\n\n"
                    "Determine the relationship: 'contradicts', 'partial_consensus', or 'methodological_divergence'. "
                    "Provide a confidence score (0.0 to 1.0) and a concise explanation."
                )

                # Structured Output Classification - via the shared abstraction
                # (tools/llm.py) so this respects LLM_PROVIDER=mock/gemini/groq
                # like every other agent, instead of a direct Gemini call.
                try:
                    tension_result = generate_json(prompt, ContradictionPair)
                    # Ensure metadata is perfectly intact for Agent 4 and Agent 5
                    tension_result.finding_a_id = finding.id
                    tension_result.finding_b_id = match_id
                    tension_result.resolved = False

                    new_tensions.append(tension_result)
                except LLMError as e:
                    print(f"Failed to classify tension between {finding.id} and {match_id}: {e}")

    # 6. Append only the tensions list update to LangGraph state
    return {"tensions": new_tensions}