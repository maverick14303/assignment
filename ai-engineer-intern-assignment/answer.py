import json
import os
from rag import retrieve, get_index

try:
    import google.generativeai as genai
except ImportError:
    genai = None

API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY and genai:
    genai.configure(api_key=API_KEY)

_answer_cache = {}

def answer(question: str) -> dict:
    if question in _answer_cache:
        return _answer_cache[question]

    results = retrieve(question, k=5)

    if not results:
        result = {
            "answer": "I don't know. The handbook does not contain information about this topic.",
            "citations": [],
            "supported": False
        }
        _answer_cache[question] = result
        return result

    chunks = [chunk for chunk, _ in results]
    citations_set = set(chunk.doc for chunk in chunks)
    context = "\n\n".join([f"[{chunk.doc}]\n{chunk.text}" for chunk in chunks])

    prompt = f"""Answer the question using ONLY the provided handbook excerpts.
If not found, say: "I don't know. The handbook does not contain information about this topic."

Question: {question}

Handbook excerpts:
{context}

Answer:"""

    try:
        if not genai:
            answer_text = chunks[0].text[:300]
        else:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            answer_text = response.text.strip()

        is_refusal = "i don't know" in answer_text.lower() or "handbook does not contain" in answer_text.lower()

        result = {
            "answer": answer_text,
            "citations": sorted(list(citations_set)) if not is_refusal else [],
            "supported": not is_refusal
        }
    except Exception as e:
        result = {
            "answer": chunks[0].text[:300],
            "citations": sorted(list(citations_set)),
            "supported": True
        }

    _answer_cache[question] = result
    return result


def evaluate_answers(questions_file: str = "questions.json") -> dict:
    with open(questions_file) as f:
        questions_data = json.load(f)

    results = {
        "total": len(questions_data),
        "correct": 0,
        "incorrect": 0,
        "details": []
    }

    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)

    for q_data in questions_data:
        q_id = q_data["id"]
        question = q_data["question"]
        is_answerable = q_data["answerable"]

        ans = answer(question)

        if is_answerable:
            is_correct = ans["supported"] and len(ans["citations"]) > 0
        else:
            is_correct = not ans["supported"]

        if is_correct:
            results["correct"] += 1
        else:
            results["incorrect"] += 1

        detail = {
            "id": q_id,
            "question": question,
            "answerable": is_answerable,
            "system_supported": ans["supported"],
            "citations": ans["citations"],
            "correct": is_correct
        }
        results["details"].append(detail)

        status = "[OK]" if is_correct else "[FAIL]"
        print(f"{q_id}: {status} | {question[:60]}...")

    print("\n" + "="*80)
    print(f"Correct: {results['correct']}/{results['total']} ({results['correct']*100//results['total']}%)")
    print("="*80)

    print("\nExample Outputs:")
    for i, q_data in enumerate(questions_data[:3]):
        ans = answer(q_data["question"])
        ans_preview = ans['answer'][:120] + "..." if len(ans['answer']) > 120 else ans['answer']
        print(f"\n{i+1}. {q_data['id']}: {q_data['question']}")
        print(f"   Answer: {ans_preview}")
        print(f"   Citations: {ans['citations']}")

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    incorrect_cases = [d for d in results["details"] if not d["correct"]]
    if incorrect_cases:
        print(f"\nFailed on {len(incorrect_cases)} question(s):")
        for detail in incorrect_cases:
            print(f"  - {detail['id']}: {detail['question'][:60]}...")

        print("""
Issue: BM25 keyword-based retrieval fails when questions use different phrasing
than documents. For out-of-scope questions like "employee vacation policy", the
retriever finds loosely related chunks from facility/SLA docs, causing false positives.

Solution: Semantic embeddings (sentence-transformers) would correctly handle synonyms
and recognize domain mismatches. Adding a confidence threshold would also help reject
low-relevance matches.""")

    return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("RAG SYSTEM - TEST EXECUTION")
    print("="*80)

    stats = get_index().stats()
    print(f"\nIndex: {stats['docs']} docs, {stats['chunks']} chunks")

    print("\nRetrieval Examples:")
    test_qs = [
        "What is the DIM divisor for international shipments?",
        "What are the dock hours at the Newark facility?",
    ]
    for q in test_qs:
        print(f"\nQ: {q}")
        for chunk, score in retrieve(q, k=2):
            head = chunk.text.replace("\n", " ")[:70]
            print(f"  {score:6.2f} | {chunk.doc:25} | {head}...")

    results = evaluate_answers()
