"""Bước 1 Day 7: xuất chunk từ vector store Day 6 (pgvector) và chuyển benchmark Day 6 sang schema Day 7.

Chạy từ thư mục gốc repo Day 7, khi Docker Day 6 đang chạy:
    python src/export_day6.py --day6 C:\\VinAI\\K5-L3B-Day06-2A202604102-DataIngestionEngineering --strategy E

Kết quả (data/day6/):
    chunks.json          chunk theo schema Day 7: id, text, source_id, source, page, metadata
    questions_day6.json  20 câu Day 6 đổi sang schema Day 7 (gold: source_id, page, evidence)
    export_report.json   thống kê, cấu hình embedding/vector store, evidence không khớp

Nguyên tắc:
- Đọc chunk trực tiếp từ bảng `chunks` của Day 6 (cùng index mà dense retrieval Day 7 sẽ truy vấn),
  không tạo lại embedding, không đổi text. Cột embedding không xuất ra JSON.
- `--source rebuild` chỉ để KIỂM TRA khi không có DB: dựng lại chunk từ load/load_result.jsonl
  bằng code chunking Day 6. Không dùng chế độ này để báo cáo kết nối vector store.
- Evidence Day 6 được so khớp như Day 6 (bỏ khác biệt khoảng trắng/hoa thường), rồi ghi lại đúng
  chuỗi con nguyên văn trong chunk để `starter/baseline.py` khớp được bằng phép `in`.
- Câu thiếu bằng chứng và test mới KHÔNG được tạo ở đây: phải tự viết ở bước benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = {  # khớp starter/src/run.py của Day 6
    "A": ("fixed", {"size": 500, "overlap": 50}),
    "B": ("paragraph", {"max_size": 600}),
    "D": ("fixed", {"size": 350, "overlap": 50}),
    "E": ("paragraph", {"max_size": 350}),
}
EMBEDDING = {"model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "dimensions": 384,
             "normalize": True, "max_seq_length_tokens": 128}


def read_env(path: Path) -> dict[str, str]:
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def to_day7(row: dict, strategy: str, export_source: str) -> dict:
    meta = dict(row.get("metadata") or {})
    meta.update({"section": row.get("section", meta.get("section")), "strategy": strategy,
                 "export_source": export_source})
    return {"id": row["chunk_id"], "text": row["text"], "source_id": row["document_id"],
            "source": row["source"], "page": int(row["page"]), "metadata": meta}


def chunks_from_db(database_url: str, strategy: str) -> list[dict]:
    import psycopg  # cài trong venv Day 6: psycopg[binary]
    with psycopg.connect(database_url, connect_timeout=5) as conn, conn.cursor() as cur:
        cur.execute("SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                    "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'")
        col = cur.fetchone()[0]
        if col != f"vector({EMBEDDING['dimensions']})":
            raise RuntimeError(f"Cột embedding là {col}, cần vector({EMBEDDING['dimensions']})")
        cur.execute("SELECT chunk_id, document_id, source, page, section, text, metadata "
                    "FROM chunks WHERE strategy = %s "
                    "ORDER BY page, (metadata->>'char_start')::int, chunk_id", (strategy,))
        names = [d.name for d in cur.description]
        rows = [dict(zip(names, r)) for r in cur.fetchall()]
    if not rows:
        raise RuntimeError(f"Không có chunk nào với strategy={strategy}. Đã chạy ingest Day 6 chưa?")
    return [to_day7(r, strategy, "pgvector") for r in rows]


def chunks_rebuilt(day6: Path, strategy: str) -> list[dict]:
    sys.path.insert(0, str(day6 / "starter"))
    from src.chunking import chunk_pages  # type: ignore
    from src.models import DocumentPage  # type: ignore
    pages = []
    for line in (day6 / "starter" / "load" / "load_result.jsonl").read_text(encoding="utf-8").splitlines():
        p = json.loads(line)
        pages.append(DocumentPage(p["document_id"], p["source"], p["page"], p["text"], p["metadata"]))
    method, params = STRATEGIES[strategy]
    out = []
    for c in chunk_pages(pages, method, **params):
        row = {"chunk_id": c.chunk_id, "document_id": c.document_id, "source": c.source, "page": c.page,
               "section": c.metadata.get("section"), "text": c.text, "metadata": c.metadata}
        out.append(to_day7(row, strategy, "rebuilt_from_load_result (chỉ để kiểm tra)"))
    return out


def norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def raw_span(chunk_text: str, quote: str) -> str | None:
    """Tìm lại chuỗi nguyên văn trong chunk, cho phép khác khoảng trắng và hoa thường."""
    pattern = r"\s+".join(re.escape(t) for t in quote.split())
    m = re.search(pattern, chunk_text, flags=re.IGNORECASE)
    return m.group(0) if m else None


def evidence_sets(q: dict) -> list[dict]:
    """Giống evaluation.evidence_sets của Day 6: mỗi bộ cần TẤT CẢ quote trong cùng một chunk."""
    pages = q["gold_page"] if isinstance(q["gold_page"], list) else [q["gold_page"]]
    quotes = q["evidence_quote"] if isinstance(q["evidence_quote"], list) else [q["evidence_quote"]]
    sets = [{"pages": pages, "quotes": quotes}]
    sets += [{"pages": [a["page"]], "quotes": [a["quote"]]} for a in q.get("alternative_evidence", [])]
    return sets


def convert_question(q: dict, chunks: list[dict], split: str) -> tuple[dict, list[str]]:
    gold, problems = [], []
    for gi, ev in enumerate(evidence_sets(q)):
        found = False
        for c in chunks:
            if c["source_id"] != q["gold_source_id"] or c["page"] not in ev["pages"]:
                continue
            if not all(norm(x) in norm(c["text"]) for x in ev["quotes"]):
                continue
            spans = [raw_span(c["text"], x) for x in ev["quotes"]]
            if None in spans:
                continue
            found = True
            for s in spans:
                g = {"source_id": c["source_id"], "page": c["page"], "evidence": s, "group": gi}
                if g not in gold:
                    gold.append(g)
        if not found:
            problems.append(f"{q['id']}: bộ evidence {gi} (trang {ev['pages']}) không nằm trọn trong một chunk")
    out = {"id": q["id"], "split": split, "question": q["question"], "expected_answer": q["expected_answer"],
           "gold": gold, "answerable": True, "type": q.get("type"), "day6_split": q["split"]}
    if q.get("filter"):
        out["filter"] = q["filter"]
    if any(len(s["quotes"]) > 1 for s in evidence_sets(q)):
        out["gold_mode"] = "cần đủ mọi evidence cùng group trong một chunk"
    return out, problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day6", type=Path, required=True, help="Thư mục gốc repo Day 6")
    ap.add_argument("--strategy", default="E", choices=sorted(STRATEGIES), help="Index Day 6 dùng cho Day 7")
    ap.add_argument("--source", default="db", choices=["db", "rebuild"])
    ap.add_argument("--database-url", default=None, help="Mặc định lấy DATABASE_URL từ env hoặc Day6/starter/.env")
    ap.add_argument("--keep-day6-split", action="store_true",
                    help="Giữ split Day 6. Mặc định đưa cả 20 câu vào dev vì test Day 6 đã bị xem khi chỉnh D/E/F")
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "day6")
    args = ap.parse_args()

    if args.source == "db":
        url = args.database_url or os.environ.get("DATABASE_URL") or read_env(args.day6 / "starter" / ".env").get("DATABASE_URL")
        if not url:
            sys.exit("Thiếu DATABASE_URL")
        chunks = chunks_from_db(url, args.strategy)
    else:
        chunks = chunks_rebuilt(args.day6, args.strategy)

    ids = [c["id"] for c in chunks]
    if len(ids) != len(set(ids)) or any(not c["text"].strip() for c in chunks):
        raise RuntimeError("Chunk ID trùng hoặc text rỗng")

    questions, problems = [], []
    for line in (args.day6 / "benchmark" / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            q = json.loads(line)
            split = q["split"] if args.keep_day6_split else "dev"
            cq, p = convert_question(q, chunks, split)
            questions.append(cq)
            problems += p
    no_gold = [q["id"] for q in questions if not q["gold"]]

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    usable = [q for q in questions if q["gold"]]
    (args.output / "questions_day6.json").write_text(json.dumps(usable, ensure_ascii=False, indent=1), encoding="utf-8")
    lengths = sorted(len(c["text"]) for c in chunks)
    report = {
        "day6_repo": str(args.day6), "strategy": args.strategy, "chunking": STRATEGIES[args.strategy],
        "export_source": args.source, "embedding": EMBEDDING,
        "vector_store": "PostgreSQL 16 + pgvector 0.8.7, bảng chunks, cosine <=>, exact search",
        "n_chunks": len(chunks), "n_pages": len({(c["source_id"], c["page"]) for c in chunks}),
        "chars_min_median_max": [lengths[0], lengths[len(lengths) // 2], lengths[-1]],
        "chunks_per_section": dict(Counter(str(c["metadata"].get("section")) for c in chunks)),
        "n_questions": len(questions), "split_counts": dict(Counter(q["split"] for q in usable)),
        "questions_without_gold_dropped": no_gold, "evidence_problems": problems,
    }
    (args.output / "export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("strategy", "export_source", "n_chunks", "n_pages", "split_counts",
                                              "questions_without_gold_dropped", "evidence_problems")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
