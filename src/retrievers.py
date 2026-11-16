"""Ba phương pháp retrieval Day 7 trên cùng corpus (index E của Day 6).

- dense : query embedding (cùng model Day 6) -> pgvector cosine, exact search. Bắt buộc theo đề.
- bm25  : BM25 Okapi tự cài (không phụ thuộc thư viện) trên data/day6/chunks.json.
- hybrid: Reciprocal Rank Fusion của dense và bm25 (không cộng score thô vì thang đo khác nhau).
- rerank: lấy `candidates` kết quả hybrid rồi xếp lại bằng cross-encoder (đọc query + chunk cùng lúc).

Mọi retriever có cùng giao diện `search(query, filters=None, top_k=3) -> list[dict]`; mỗi kết quả có
id, text, source_id, source, page, metadata, score, rank, retriever. Filter chỉ cho phép `section`
(thông tin người dùng cung cấp kèm câu hỏi), giống Day 6. Lỗi DB/model được raise, không fallback giả.
"""
from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAY6 = Path(os.environ.get("DAY6_DIR", r"C:\VinAI\K5-L3B-Day06-2A202604102-DataIngestionEngineering"))
ALLOWED_FILTERS = {"section"}


def check_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    filters = filters or {}
    bad = set(filters) - ALLOWED_FILTERS
    if bad:
        raise ValueError(f"Filter không hỗ trợ: {sorted(bad)}; chỉ cho phép {sorted(ALLOWED_FILTERS)}")
    return filters


def passes(chunk: dict, filters: dict[str, Any]) -> bool:
    return all(str(chunk["metadata"].get(k)) == str(v) for k, v in filters.items())


def load_chunks(path: Path = ROOT / "data" / "day6" / "chunks.json") -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_env(path: Path) -> dict[str, str]:
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


class DenseRetriever:
    """pgvector Day 6. Query encode giống hệt Day 6: normalize_embeddings=True, không prefix."""
    name = "dense"

    def __init__(self, day6_dir: Path = DEFAULT_DAY6, strategy: str = "E",
                 database_url: str | None = None, model_path: Path | None = None) -> None:
        from sentence_transformers import SentenceTransformer
        import psycopg  # noqa: F401  (báo lỗi sớm nếu thiếu)

        env = read_env(day6_dir / "starter" / ".env")
        self.database_url = database_url or os.environ.get("DATABASE_URL") or env.get("DATABASE_URL")
        if not self.database_url:
            raise RuntimeError("Thiếu DATABASE_URL")
        model_path = model_path or day6_dir / "starter" / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
        if not model_path.is_dir():
            raise FileNotFoundError(f"Không thấy model local {model_path}")
        self.strategy = strategy
        self.model = SentenceTransformer(str(model_path), device="cpu")
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dimensions = int(get_dim())
        if self.dimensions != 384:
            raise RuntimeError(f"Model {self.dimensions} chiều, index Day 6 là 384 chiều")
        self.info = {"model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                     "model_path": str(model_path), "dimensions": self.dimensions, "normalize": True,
                     "index": f"pgvector table chunks, strategy={strategy}, cosine <=>, exact"}

    def embed(self, text: str) -> list[float]:
        v = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        return [float(x) for x in v]

    def search(self, query: str, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict]:
        import psycopg
        filters = check_filters(filters)
        qv = "[" + ",".join(repr(x) for x in self.embed(query)) + "]"
        where, params = ["strategy = %s"], [self.strategy]
        if "section" in filters:
            where.append("section = %s")
            params.append(str(filters["section"]))
        sql = (f"SELECT chunk_id, document_id, source, page, text, metadata, embedding <=> %s::vector AS d "
               f"FROM chunks WHERE {' AND '.join(where)} ORDER BY d LIMIT %s")
        with psycopg.connect(self.database_url, connect_timeout=5) as conn, conn.cursor() as cur:
            cur.execute(sql, [qv, *params, top_k])
            rows = cur.fetchall()
        return [{"id": cid, "text": text, "source_id": doc, "source": src, "page": page,
                 "metadata": meta, "score": round(1 - float(d), 6), "rank": r, "retriever": self.name}
                for r, (cid, doc, src, page, text, meta, d) in enumerate(rows, 1)]


_TOKEN = re.compile(r"\w+(?:[.,-]\w+)*")


def tokenize(text: str) -> list[str]:
    """Chữ thường + NFC; giữ số kiểu 16,977 / 21.8 / GPT-5.4 thành một token."""
    return _TOKEN.findall(unicodedata.normalize("NFC", text).casefold())


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tfs = [Counter(tokenize(c["text"])) for c in chunks]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lens) / len(self.lens)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.info = {"k1": k1, "b": b, "tokenizer": "casefold NFC regex \\w+ (giữ số thập phân/dấu nối)",
                     "stopwords": None, "n_chunks": n}

    def search(self, query: str, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict]:
        filters = check_filters(filters)
        q = [t for t in tokenize(query) if t in self.idf]
        scored = []
        for c, tf, dl in zip(self.chunks, self.tfs, self.lens):
            if not passes(c, filters):
                continue
            s = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                    for t in q if t in tf)
            if s > 0:
                scored.append((s, c))
        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        return [{**c, "score": round(s, 6), "rank": r, "retriever": self.name}
                for r, (s, c) in enumerate(scored[:top_k], 1)]


class HybridRRF:
    """score(d) = Σ 1/(k + rank_i(d)) trên các retriever; lấy `candidates` kết quả từ mỗi bên."""
    name = "hybrid"

    def __init__(self, retrievers: list, k: int = 60, candidates: int = 20) -> None:
        self.retrievers, self.k, self.candidates = retrievers, k, candidates
        self.info = {"fusion": "RRF", "k": k, "candidates_per_retriever": candidates,
                     "members": [r.name for r in retrievers]}

    def search(self, query: str, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict]:
        fused: dict[str, dict] = {}
        for r in self.retrievers:
            for hit in r.search(query, filters=filters, top_k=self.candidates):
                e = fused.setdefault(hit["id"], {**hit, "score": 0.0, "member_ranks": {}})
                e["score"] += 1 / (self.k + hit["rank"])
                e["member_ranks"][r.name] = hit["rank"]
        ranked = sorted(fused.values(), key=lambda h: (-h["score"], h["id"]))[:top_k]
        return [{**h, "score": round(h["score"], 6), "rank": i, "retriever": self.name}
                for i, h in enumerate(ranked, 1)]


class RerankRetriever:
    name = "rerank"

    def __init__(self, base, model_name: str = "BAAI/bge-reranker-v2-m3", candidates: int = 20,
                 max_length: int = 512) -> None:
        from sentence_transformers import CrossEncoder
        self.base, self.candidates = base, candidates
        self.model = CrossEncoder(model_name, device="cpu", max_length=max_length)
        self.info = {"base": getattr(base, "info", {}), "base_name": base.name, "reranker": model_name,
                     "candidates": candidates, "max_length": max_length}

    def search(self, query: str, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict]:
        cands = self.base.search(query, filters=filters, top_k=self.candidates)
        if not cands:
            return []
        scores = self.model.predict([(query, c["text"]) for c in cands], convert_to_numpy=True)
        ranked = sorted(zip(scores, cands), key=lambda x: (-float(x[0]), x[1]["id"]))[:top_k]
        return [{**c, "base_rank": c["rank"], "base_score": c["score"], "score": round(float(s), 6),
                 "rank": i, "retriever": self.name} for i, (s, c) in enumerate(ranked, 1)]


def build(method: str, chunks: list[dict] | None = None, **kw):
    chunks = chunks if chunks is not None else load_chunks()
    if method == "dense":
        return DenseRetriever(**kw)
    if method == "bm25":
        return BM25Retriever(chunks)
    if method == "hybrid":
        return HybridRRF([DenseRetriever(**kw), BM25Retriever(chunks)])
    if method == "rerank":
        reranker = kw.pop("reranker", "BAAI/bge-reranker-v2-m3")
        return RerankRetriever(HybridRRF([DenseRetriever(**kw), BM25Retriever(chunks)]), model_name=reranker)
    raise ValueError(f"Phương pháp không hỗ trợ: {method}")
