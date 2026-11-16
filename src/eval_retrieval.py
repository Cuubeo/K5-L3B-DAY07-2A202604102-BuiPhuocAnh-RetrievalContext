"""Benchmark retrieval + context trên một split, cùng corpus/nhãn cho mọi phương pháp.

    python src/eval_retrieval.py --methods dense bm25 hybrid --split dev
    python src/eval_retrieval.py --methods bm25 hybrid --split dev --budget 1600 --neighbors 1 --tag nb1-b1600
    python src/eval_retrieval.py --methods rerank --split dev

Metric (chỉ trên câu có đáp án):
- Hit@1, Evidence Hit@3: tồn tại một group evidence mà MỌI đoạn trích của group nằm trong top-1 / top-3
  (mỗi đoạn trích nằm trọn trong một chunk đúng nguồn + trang). Group 1 đoạn = định nghĩa của baseline.
- Context Evidence Rate: như trên nhưng xét các chunk thật sự vào context (src/context.pack: budget,
  dedup, tùy chọn thêm chunk liền kề). Context có thể có chunk ngoài top-k nhờ neighbors.
Câu thiếu evidence vẫn được chạy và lưu trace (score top-1) nhưng không vào mẫu số.
Filter: dùng trường `filter` người dùng cung cấp kèm câu hỏi (nếu có); gold không vào retriever.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT / "src"))
from context import pack  # noqa: E402
from retrievers import build, load_chunks  # noqa: E402


def evidence_in(chunk: dict, g: dict) -> bool:
    return chunk["source_id"] == g["source_id"] and chunk["page"] == g["page"] and g["evidence"] in chunk["text"]


def group_hit(chunks: list[dict], gold: list[dict]) -> bool:
    groups: dict[int, list[dict]] = {}
    for g in gold:
        groups.setdefault(g.get("group", id(g)), []).append(g)
    return any(all(any(evidence_in(c, g) for c in chunks) for g in gs) for gs in groups.values())


def run(method: str, retriever, questions: list[dict], top_k: int, budget: int,
        chunks: list[dict] | None = None, neighbors: int = 0) -> tuple[list, dict]:
    traces = []
    for q in questions:
        t0 = time.perf_counter()
        err, results = None, []
        try:
            results = retriever.search(q["question"], filters=q.get("filter"), top_k=max(top_k, 3))
        except Exception as e:  # ghi lỗi, không thay bằng kết quả giả
            err = f"{type(e).__name__}: {e}"
        ms = (time.perf_counter() - t0) * 1000
        used = results[:top_k]
        context, ids, ctx_chunks = pack(used, chunks, budget=budget, neighbors=neighbors)
        t = {"query_id": q["id"], "split": q["split"], "answerable": q["answerable"], "question": q["question"],
             "filter": q.get("filter"), "method": method, "top_k": top_k, "budget_chars": budget,
             "neighbors": neighbors, "results": [{k: c.get(k) for k in ("rank", "id", "page", "score", "member_ranks", "base_rank")} |
                         {"section": c["metadata"].get("section"), "text": c["text"]} for c in results],
             "context_ids": ids, "context_chars": len(context), "elapsed_ms": round(ms, 2), "error": err}
        if q["answerable"]:
            t |= {"hit1": group_hit(results[:1], q["gold"]), "hit3": group_hit(results[:3], q["gold"]),
                  "context_hit": group_hit(ctx_chunks, q["gold"])}
        traces.append(t)
    ans = [t for t in traces if t["answerable"]]
    n = len(ans)
    rate = lambda k: f"{sum(t[k] for t in ans)}/{n} ({sum(t[k] for t in ans) / n:.0%})" if n else "0/0"
    summary = {"method": method, "split": questions[0]["split"] if questions else None,
               "config": getattr(retriever, "info", {}), "top_k": top_k, "budget_chars": budget,
               "neighbors": neighbors,
               "n_answerable": n, "n_abstain": len(traces) - n,
               "hit1": rate("hit1"), "evidence_hit3": rate("hit3"), "context_evidence_rate": rate("context_hit"),
               "misses_hit3": [t["query_id"] for t in ans if not t["hit3"]],
               "errors": [t["query_id"] for t in traces if t["error"]],
               "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1),
               "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 2), "cost_usd": 0}
    return traces, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["dense", "bm25", "hybrid"])
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--budget", type=int, default=800, help="Context budget (ký tự, gồm header nguồn)")
    ap.add_argument("--neighbors", type=int, default=0, help="Thêm n chunk liền kề cùng trang vào context")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--tag", default=None, help="Hậu tố thư mục kết quả: <method>-<tag>")
    ap.add_argument("--output", type=Path, default=ROOT / "results" / "retrieval")
    args = ap.parse_args()

    bench = json.loads((ROOT / "data" / "benchmark" / "benchmark.json").read_text(encoding="utf-8"))
    questions = [q for q in bench if q["split"] == args.split]
    chunks = load_chunks()
    rows = []
    for m in args.methods:
        retriever = build(m, chunks, **({"reranker": args.reranker} if m == "rerank" else {}))
        traces, summary = run(m, retriever, questions, args.top_k, args.budget, chunks, args.neighbors)
        summary["run"] = f"{m}-{args.tag}" if args.tag else m
        out = args.output / args.split / summary["run"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        rows.append(summary)
        print(json.dumps({k: summary[k] for k in ("method", "hit1", "evidence_hit3", "context_evidence_rate",
                                                  "misses_hit3", "errors", "median_ms")}, ensure_ascii=False))
    print("\n| Run | top-k | budget | neighbors | Hit@1 | Evidence Hit@3 | Context Evidence Rate | median ms |")
    print("|---|---|---|---|---|---|---|---|")
    for s in rows:
        print(f"| {s['run']} | {s['top_k']} | {s['budget_chars']} | {s['neighbors']} | {s['hit1']} | {s['evidence_hit3']} | "
              f"{s['context_evidence_rate']} | {s['median_ms']} |")


if __name__ == "__main__":
    main()
