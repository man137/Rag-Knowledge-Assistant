"""rag — hand-written retrieval core for the enterprise RAG project.

WHAT: package marker for the retrieval core (loaders, cleaning, chunking,
embedders, vector store, BM25, hybrid fusion, reranker, MMR, prompts,
generator, pipeline, security, evaluation).
WHY: the whole point of this project is that the retrieval maths is written
by hand with NumPy and the standard library — no RAG frameworks.
MATHS: none here.
FAILURE MODE: none at package level; each module documents its own.
INTERVIEW ONE-LINER: "I built the retrieval pipeline itself — chunking,
BM25, RRF fusion, IVF, MMR — in plain NumPy so I can explain every step."
"""

__version__ = "1.0.0"
