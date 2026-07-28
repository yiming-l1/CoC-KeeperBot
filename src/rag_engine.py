"""
rag_engine.py

A small but production-friendly RAG engine for:
- One-time initialization (embeddings + Chroma DB handle cached)
- PDF ingestion with robust chunking (separators prioritize headings/paragraphs)
- Metadata for observability (source/page/category/chunk_id)
- Category filtering for "spoiler-safe" retrieval
- Optional rerank (vector recall -> cross-encoder rerank)
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma


# =========================
# Default Paths / Settings
# =========================

# Default PDF path used by build_vector_database() (you can override via arguments)
PDF_PATH = "./data/raw/Call_of_Cthulhu_7th_Edition_Quick_Start_Rules.pdf"

# Chroma persistence path
DB_PATH = "./data/chroma_db"

# Page ranges are 0-indexed (PyPDFLoader's "page" metadata is usually 0-indexed as well)
RULES_START_PAGE = 4  # 5
RULES_END_PAGE = 14  # 15

# Scenario "The Haunting"
SCENARIO_PDF_PATH = "./data/raw/Call_of_Cthulhu_7th_Edition_Quick_Start_Rules.pdf"
SCENARIO_START_PAGE = 16  # 17
SCENARIO_END_PAGE = 31  # 32

# =========================
# Embeddings wrapper
# =========================


class SentenceTransformersEmbeddings(Embeddings):
    """LangChain Embeddings wrapper around sentence-transformers.

    Why:
    - SentenceTransformers downloads/loads models without relying on some newer Hugging Face Hub endpoints
      that may raise 404 for non-chat models (e.g. additional_chat_templates).
    - Keeps behavior stable for embedding-only models like sentence-transformers/all-MiniLM-L6-v2.
    """

    def __init__(
        self, model_name: str, *, normalize: bool = True, device: Optional[str] = None
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(
            texts, normalize_embeddings=self.normalize, show_progress_bar=False
        )
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> List[float]:
        v = self.model.encode(
            [text], normalize_embeddings=self.normalize, show_progress_bar=False
        )[0]
        return v.tolist()


# =========================
# Config / Engine
# =========================


@dataclass
class RAGConfig:
    """Configuration for the RAG engine."""

    db_path: str = DB_PATH
    embedding_model: str = "all-MiniLM-L6-v2"

    # Chunking: for rulebooks, prioritize paragraph boundaries; overlap helps keep rules intact
    chunk_size: int = 800
    chunk_overlap: int = 200
    separators: Tuple[str, ...] = (
        "\n\n\n",
        "\n\n",  # paragraphs / headings
        "\n",  # lines
        "\n• ",
        "\n- ",
        "\n1. ",
        "\n2. ",
        "\n3. ",
        "。",
        "！",
        "？",  # Chinese sentence endings (harmless if not present)
        ". ",
        "? ",
        "! ",  # English sentence endings
        "; ",
        ": ",
        ", ",
        " ",
        "",
    )

    # Retrieval defaults
    default_k: int = 15

    # MMR tuning (if you use search_type="mmr")
    mmr_lambda_mult: float = 0.5
    mmr_fetch_k_factor: int = 10  # fetch_k = max(30, k * factor)

    # Threshold retrieval
    score_threshold: float = 0.25  # 0.3

    # Optional rerank (improves relevance noticeably for QA/RAG)
    use_rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 50  # rerank top N from initial recall


class RAGEngine:
    """
    RAGEngine caches:
      - embeddings model
      - Chroma database handle
      - optional reranker

    It provides:
      - build/reset DB
      - ingest PDFs
      - retrieve with category filtering and optional rerank
      - results formatted with metadata for debugging and citations
    """

    def __init__(self, cfg: Optional[RAGConfig] = None):
        self.cfg = cfg or RAGConfig()
        self._embeddings: Optional[Embeddings] = None
        self._db: Optional[Chroma] = None
        self._reranker = None  # lazy init (CrossEncoder) only if enabled

    # ---------- Lazy singletons ----------
    @property
    def embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = SentenceTransformersEmbeddings(self.cfg.embedding_model)
        return self._embeddings

    @property
    def db(self) -> Chroma:
        if self._db is None:
            os.makedirs(self.cfg.db_path, exist_ok=True)
            # client = chromadb.Client(Settings(persist_directory=self.cfg.db_path, anonymized_telemetry=False))
            client = chromadb.PersistentClient(path=self.cfg.db_path)
            self._db = Chroma(
                client=client,
                collection_name="coc_rag_v1",
                embedding_function=self.embeddings,
            )
        return self._db

    def _get_reranker(self):
        if self._reranker is None:
            # Optional dependency. Install: pip install sentence-transformers
            from sentence_transformers import CrossEncoder  # type: ignore

            self._reranker = CrossEncoder(self.cfg.rerank_model)
        return self._reranker

    # ---------- Chunking ----------
    def make_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
            separators=list(self.cfg.separators),
        )

    # ---------- Ingestion ----------
    def reset_db(self) -> None:
        if os.path.exists(self.cfg.db_path):
            shutil.rmtree(self.cfg.db_path)
        self._db = None

    def load_pdf_pages(
        self,
        pdf_path: str,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        *,
        category: str,
        source_name: Optional[str] = None,
    ):
        """
        Loads a PDF and returns page Documents in [start_page, end_page] inclusive.
        Adds metadata: category, source.
        """
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # slice pages (inclusive)
        if start_page is not None or end_page is not None:
            s = start_page or 0
            e = end_page if end_page is not None else len(docs) - 1
            docs = docs[s : e + 1]

        src = source_name or os.path.basename(pdf_path)
        for d in docs:
            md = d.metadata or {}
            md["category"] = category
            md["source"] = src
            d.metadata = md
        return docs

    def add_documents(self, docs, *, persist: bool = True) -> int:
        """Split and add documents into Chroma; returns number of chunks added."""
        splitter = self.make_splitter()
        chunks = splitter.split_documents(docs)

        # add stable chunk_id for observability/debug
        for i, c in enumerate(chunks):
            md = c.metadata or {}
            md.setdefault("chunk_id", i)
            c.metadata = md

        self.db.add_documents(chunks)
        if persist:
            if hasattr(self.db, "persist"):
                self.db.persist()
        return len(chunks)

    def build_from_pdfs(
        self,
        *,
        rules_pdf: str = PDF_PATH,
        rules_pages: tuple[int, int] = (RULES_START_PAGE, RULES_END_PAGE),
        scenario_pdf: str | None = SCENARIO_PDF_PATH,
        scenario_pages: tuple[int, int] | None = None,
        reset: bool = False,
    ):
        """
        Build (or load) the DB from configured PDFs.

        Semantics:
        - reset=True  -> delete & rebuild from PDFs
        - reset=False -> if DB exists AND non-empty, just load; else ingest
        """

        # --- 0) If reset -> delete DB and rebuild ---
        if reset:
            self.reset_db()

        # --- 1) If not reset: try to load existing DB and verify it's usable (non-empty) ---
        if not reset:
            # Open DB (persistent client should point to cfg.db_path)
            db = self.db

            # Determine if collection already has data
            count = 0
            try:
                # langchain-chroma typically exposes underlying collection here
                count = db._collection.count()
            except Exception:
                # If API changed, treat as empty and fall back to ingestion
                count = 0

            if count > 0:
                print(f"✅ Loaded existing DB ({count} chunks). Skip ingestion.")
                return db
            else:
                print("⚠️ DB exists but is empty/unusable. Re-ingesting...")

        print(f"📦 Building RAG DB at {self.cfg.db_path}...")

        # --- 2) Ingest Rules ---
        rules_docs = self.load_pdf_pages(
            rules_pdf,
            start_page=rules_pages[0],
            end_page=rules_pages[1],
            category="rule_system",
            source_name="CoC Quick Start Rules",
        )
        self.add_documents(rules_docs, persist=False)

        # --- 3) Ingest Scenario (optional) ---
        if scenario_pdf:
            # scenario_pages fallback if not provided
            if (
                scenario_pages is None
                and SCENARIO_START_PAGE is not None
                and SCENARIO_END_PAGE is not None
            ):
                scenario_pages = (SCENARIO_START_PAGE, SCENARIO_END_PAGE)

            if scenario_pages is not None:
                scen_docs = self.load_pdf_pages(
                    scenario_pdf,
                    start_page=scenario_pages[0],
                    end_page=scenario_pages[1],
                    category="scenario_data",
                    source_name=os.path.basename(scenario_pdf),
                )
            else:
                scen_docs = self.load_pdf_pages(
                    scenario_pdf,
                    start_page=None,
                    end_page=None,
                    category="scenario_data",
                    source_name=os.path.basename(scenario_pdf),
                )

            self.add_documents(scen_docs, persist=False)

        # --- 4) Persist once at the end (if supported) ---
        if hasattr(self.db, "persist") and callable(getattr(self.db, "persist")):
            try:
                self.db.persist()
            except Exception:
                pass

        # --- 5) Sanity print ---
        try:
            final_count = self.db._collection.count()
            print(f"✅ Ingestion done. DB now has {final_count} chunks.")
        except Exception:
            print("✅ Ingestion done.")

        return self.db

    # ---------- Retrieval ----------
    def _make_search_kwargs(
        self,
        search_type: str,
        k: int,
        category: Optional[str],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"k": k}

        if search_type == "mmr":
            fetch_k = max(30, k * self.cfg.mmr_fetch_k_factor)
            kwargs.update({"fetch_k": fetch_k, "lambda_mult": self.cfg.mmr_lambda_mult})

        if search_type in ("threshold", "similarity_score_threshold"):
            # langchain's canonical name
            kwargs = {"k": max(k, 8), "score_threshold": self.cfg.score_threshold}

        if category:
            # Chroma metadata filter
            kwargs["filter"] = {"category": category}

        return kwargs

    def retrieve(
        self,
        query: str,
        *,
        search_type: str = "mmr",
        k: Optional[int] = None,
        category: Optional[str] = None,
        use_rerank: Optional[bool] = None,
        return_raw_docs: bool = False,
    ):
        """
        Retrieve relevant chunks.

        category usage:
          - Players (no spoilers): category="rule_system"
          - Keeper (allow scenario): category=None or category="scenario_data" (or both via separate calls)

        If use_rerank=True:
          - vector recall happens first
          - top N are reranked by cross-encoder
        """
        k = k or self.cfg.default_k
        use_rerank = self.cfg.use_rerank if use_rerank is None else use_rerank

        # For rerank, we usually want a wider initial recall
        recall_k = k
        if use_rerank:
            recall_k = max(k, min(self.cfg.rerank_top_n, max(20, k * 6)))

        search_kwargs = self._make_search_kwargs(search_type, recall_k, category)

        # Normalize search_type
        if search_type == "threshold":
            search_type = "similarity_score_threshold"

        retriever = self.db.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )
        # docs = retriever.get_relevant_documents(query)
        docs = retriever.invoke(query)

        if use_rerank and docs:
            docs = self._rerank(query, docs, k=k)

        if return_raw_docs:
            return docs

        return self.format_results(docs)

    def _rerank(self, query: str, docs, *, k: int):
        reranker = self._get_reranker()
        pairs = [(query, d.page_content) for d in docs]
        scores = reranker.predict(pairs)

        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)

        top_docs = []
        for d, s in scored[:k]:
            md = d.metadata or {}
            md["rerank_score"] = float(s)
            d.metadata = md
            top_docs.append(d)
        return top_docs

    # ---------- Observability / formatting ----------
    def format_results(self, docs) -> List[Dict[str, Any]]:
        """
        Return a compact structure you can feed into prompts and show as citations.
        """
        results: List[Dict[str, Any]] = []
        for d in docs:
            md = dict(d.metadata or {})
            results.append(
                {
                    "text": d.page_content,
                    "source": md.get("source"),
                    "page": md.get("page", md.get("page_number")),
                    "category": md.get("category"),
                    "chunk_id": md.get("chunk_id"),
                    "rerank_score": md.get("rerank_score"),
                }
            )
        return results


# =========================
# Backward-compatible API in main_keeper.ipynb for testing
# =========================

_DEFAULT_ENGINE: Optional[RAGEngine] = None


def get_engine() -> RAGEngine:
    """Module-level singleton engine."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = RAGEngine(RAGConfig(db_path=DB_PATH))
    return _DEFAULT_ENGINE


def build_vector_database(
    pdf_path: str = PDF_PATH,
    *,
    reset: bool = False,
    rules_pages: tuple[int, int] = (RULES_START_PAGE, RULES_END_PAGE),
    scenario_pdf: str | None = SCENARIO_PDF_PATH,
    scenario_pages: tuple[int, int] | None = None,
):
    """
    Backward compatible builder.
      - reset=True: rebuild from scratch
      - reset=False: load existing DB if non-empty, otherwise ingest
    """
    engine = get_engine()
    db = engine.build_from_pdfs(
        rules_pdf=pdf_path,
        rules_pages=rules_pages,
        scenario_pdf=scenario_pdf,
        scenario_pages=scenario_pages,
        reset=reset,
    )

    # Print a more accurate status
    try:
        cnt = db._collection.count()
        if reset:
            print(f"✅ Database rebuilt successfully! chunks={cnt}")
        else:
            print(f"✅ Database ready. chunks={cnt} (loaded or ingested as needed)")
    except Exception:
        print("✅ Database ready.")

    return db


def get_embeddings() -> Embeddings:
    """
    Backward compatible embeddings getter.
    Returns a cached embeddings object (not re-initialized each call).
    """
    return get_engine().embeddings


def get_retriever(
    search_type: str = "mmr",
    k: int = 5,
    *,
    category: Optional[str] = None,
    use_rerank: Optional[bool] = None,
):
    """
    Backward compatible retriever factory.
    - category="rule_system" enables spoiler-safe retrieval
    - use_rerank=True enables cross-encoder rerank (needs sentence-transformers)
    """
    engine = get_engine()

    # Build a retriever that mirrors engine.retrieve() search kwargs
    # Note: rerank is applied in engine.retrieve(), not in retriever itself.
    if search_type == "threshold":
        search_type_norm = "similarity_score_threshold"
    else:
        search_type_norm = search_type

    # If rerank is enabled, we recall more docs and rerank later in engine.retrieve().
    recall_k = k
    if engine.cfg.use_rerank if use_rerank is None else use_rerank:
        recall_k = max(k, min(engine.cfg.rerank_top_n, max(20, k * 6)))

    search_kwargs = engine._make_search_kwargs(search_type, recall_k, category)
    return engine.db.as_retriever(
        search_type=search_type_norm, search_kwargs=search_kwargs
    )
