import os
from rag import retrieve as retrieve_bm25
from rag_semantic import retrieve_semantic

try:
    import google.generativeai as genai
except ImportError:
    genai = None

API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY and genai:
    genai.configure(api_key=API_KEY)

_answer_cache = {}

def compare_retrievers(question: str, k: int = 5):
    print("\n" + "="*80)
    print(f"COMPARING BM25 vs SEMANTIC RETRIEVAL")
    print("="*80)
    print(f"Question: {question}\n")

    print("BM25 (Keyword-based):")
    bm25_results = retrieve_bm25(question, k)
    for i, (chunk, score) in enumerate(bm25_results, 1):
        head = chunk.text.replace("\n", " ")[:70]
        print(f"  {i}. Score: {score:6.2f} | {chunk.doc:25} | {head}...")

    print("\nSEMANTIC (Embedding-based):")
    semantic_results = retrieve_semantic(question, k)
    for i, (chunk, score) in enumerate(semantic_results, 1):
        head = chunk.text.replace("\n", " ")[:70]
        print(f"  {i}. Score: {score:6.4f} | {chunk.doc:25} | {head}...")

    bm25_docs = {chunk.doc for chunk, _ in bm25_results}
    semantic_docs = {chunk.doc for chunk, _ in semantic_results}
    common = bm25_docs & semantic_docs
    only_semantic = semantic_docs - bm25_docs

    print(f"\nCommon docs: {common if common else 'None'}")
    print(f"Only semantic found: {only_semantic if only_semantic else 'None'}")


def compare_all_questions():
    import json

    with open("questions.json") as f:
        questions_data = json.load(f)

    print("\n" + "="*80)
    print("BM25 vs SEMANTIC ON ALL TEST QUESTIONS")
    print("="*80)

    bm25_correct = 0
    semantic_correct = 0

    for q_data in questions_data:
        q_id = q_data["id"]
        question = q_data["question"]
        is_answerable = q_data["answerable"]

        bm25_results = retrieve_bm25(question, k=5)
        bm25_found = len(bm25_results) > 0 and bm25_results[0][1] > 2.0

        semantic_results = retrieve_semantic(question, k=5)
        semantic_found = len(semantic_results) > 0 and semantic_results[0][1] > 0.5

        bm25_ok = bm25_found == is_answerable
        semantic_ok = semantic_found == is_answerable

        if bm25_ok:
            bm25_correct += 1
        if semantic_ok:
            semantic_correct += 1

        status_bm25 = "[OK]" if bm25_ok else "[FAIL]"
        status_sem = "[OK]" if semantic_ok else "[FAIL]"

        print(f"{q_id}: BM25 {status_bm25} | Semantic {status_sem}")

    total = len(questions_data)
    print(f"\nBM25:     {bm25_correct}/{total} ({bm25_correct*100//total}%)")
    print(f"Semantic: {semantic_correct}/{total} ({semantic_correct*100//total}%)")
    print(f"Improvement: {semantic_correct - bm25_correct:+d} questions")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("BONUS: SEMANTIC EMBEDDINGS vs BM25 COMPARISON")
    print("="*80)

    test_qs = [
        "What is the DIM divisor for international shipments?",
        "What are the dock hours at Newark?",
        "What's Meridian's employee vacation policy?",
    ]

    for q in test_qs:
        compare_retrievers(q)

    compare_all_questions()

    print("\n" + "="*80)
    print("NOTE: Semantic embeddings handle paraphrases and synonyms better than")
    print("keyword-based BM25, especially for out-of-domain question detection.")
    print("="*80)
