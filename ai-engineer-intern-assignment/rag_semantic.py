import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from rag import Chunk, chunk_document, CORPUS_DIR

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "-q", "sentence-transformers"])
    from sentence_transformers import SentenceTransformer

@dataclass
class SemanticIndex:
    def __init__(self, corpus_dir: Path = CORPUS_DIR, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading sentence-transformer: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.corpus_dir = corpus_dir
        self.chunks = []
        self.doc_names = set()
        self.embeddings = None
        self._load()

    def _load(self):
        paths = sorted(self.corpus_dir.glob("*.md"))
        if not paths:
            raise FileNotFoundError(f"no .md files in {self.corpus_dir}")

        for p in paths:
            name = p.stem
            self.doc_names.add(name)
            self.chunks.extend(chunk_document(name, p.read_text(encoding="utf-8")))

        print(f"Computing embeddings for {len(self.chunks)} chunks...")
        chunk_texts = [c.text for c in self.chunks]
        self.embeddings = self.model.encode(chunk_texts, show_progress_bar=True)

    def search(self, question: str, k: int = 5) -> list[tuple[Chunk, float]]:
        question_embedding = self.model.encode(question)
        similarities = np.dot(self.embeddings, question_embedding)
        top_indices = np.argsort(similarities)[::-1][:k]
        return [(self.chunks[i], float(similarities[i])) for i in top_indices]

    def stats(self) -> dict:
        return {
            "docs": len(self.doc_names),
            "chunks": len(self.chunks),
            "model": self.model_name,
            "embedding_dim": self.embeddings.shape[1],
            "avg_chunk_chars": round(sum(len(c.text) for c in self.chunks) / len(self.chunks)),
        }

_SEMANTIC_INDEX = None

def get_semantic_index() -> SemanticIndex:
    global _SEMANTIC_INDEX
    if _SEMANTIC_INDEX is None:
        _SEMANTIC_INDEX = SemanticIndex()
    return _SEMANTIC_INDEX

def retrieve_semantic(question: str, k: int = 5) -> list[tuple[Chunk, float]]:
    return get_semantic_index().search(question, k)
