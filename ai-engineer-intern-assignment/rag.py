import json 
import re

from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

CORPUS_DIR = Path(__file__).parent/"corpus"

CHUNK_TARGET_CHARS = 700
CHUNK_MAX_CHARS = 1100 
TOP_K = 5

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

def tokenize(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+" , text.lower())
    return [t for t in toks if t not in STOPWORDS]

@dataclass
class Chunk:
    doc: str 
    idx: int
    text: str
    tokens: list[str]

    @property
    def cid(self) -> str:
        return f"{self.doc} # {self.idx}"

def _split_paragraphs(body:str) -> list[str]:
    raw = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    merged:list[str] = []
    for p in raw:
        if re.match(r"^\s*(?:[-*]| \d + \.)\s", p) and merged:
            merged[-1] += "\n\n" +p
        else:
            merged.append(p)
    return merged

def chunk_document(doc_name: str, text: str) -> list[Chunk]:
    lines = text.strip().split("\n")
    has_title = bool(lines) and lines[0].startswith("#")
    title = lines[0].lstrip("#").strip() if has_title else doc_name
    body = "\n".join(lines[1:]) if has_title else text
    chunks: list[Chunk] = []
    buf: list[str] = []
    size = 0
    def flush ():
        nonlocal buf, size
        if not buf:
            return
        body_text = "\n\n".join(buf)
        chunks.append(Chunk(
            doc=doc_name,
            idx=len(chunks),
            text=f"#{title}\n\n{body_text}",
            tokens=tokenize(f"{title} {body_text}"),
        ))
        buf, size = [],0
    for p in _split_paragraphs(body):
        if size and size + len(p) > CHUNK_TARGET_CHARS:
            flush()
        buf.append(p)
        size += len(p)
        if size > CHUNK_MAX_CHARS:
            flush()
    flush()
    return chunks or [Chunk(doc_name, 0, text.strip(), tokenize(text))]


class Index:
    def __init__(self, corpus_dir: Path = CORPUS_DIR):
        self.corpus_dir = corpus_dir
        self.chunks: list[Chunk] = []
        self.doc_names: set[str] = set()
        self._load()
        self.bm25 = BM25Okapi([c.tokens for c in self.chunks])

    def _load(self):
        paths = sorted(self.corpus_dir.glob("*.md"))
        if not paths:
            raise FileNotFoundError(f"no .md file in {self.corpus_dir}")
        for p in paths:
            name = p.stem
            self.doc_names.add(name)
            self.chunks.extend(chunk_document(name, p.read_text(encoding="utf-8")))

    def search (self, question: str, k: int = TOP_K) -> list[tuple[Chunk, float]]:

        scores = self.bm25.get_scores(tokenize(question))
        order = sorted(range(len(scores)), key= lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]                                                                       
    def stats(self) -> dict:
        sizes = [len(c.text) for c in self.chunks]
        return {
            "docs": len(self.doc_names),
            "chunks": len(self.chunks),
            "avg_chunk_chars": round(sum(sizes)/ len(sizes)),
            "max_chunk_chars": max(sizes),
        }

_INDEX: Index| None = None

def get_index() -> Index:
    global _INDEX
    if _INDEX is None:
        _INDEX = Index()
    return _INDEX

def retrieve(question:str, k: int = TOP_K)-> list[tuple[Chunk,float]]:
    return get_index().search(question,k)

if __name__ == "__main__":
    ix = get_index()
    print(json.dumps(ix.stats(), indent=2)) 
    for q in [
        "What is the DIM divisor for international shipments?",
        "What are the dock hours at the Newark facility?",
        "What is Meridian's employee vacation policy?",
    ]:
        print ("=" *78)
        print ("Q:", q)
        for c, s in retrieve(q):
            head = c.text.replace("\n", " ")[:95]
            print(f" {s:6.2f} {c.cid:<32} {head}")

