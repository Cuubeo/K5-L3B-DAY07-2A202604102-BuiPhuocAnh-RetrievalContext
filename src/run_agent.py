"""Chạy agent trên một split và lưu trace + mẫu nhãn để tự chấm.

    python src/run_agent.py --split dev --method bm25 --budget 1600 --neighbors 1 --tag bm25-nb1
    python src/run_agent.py --split dev --method hybrid --limit 3          # thử nhanh 3 câu

Đầu ra results/agent/<split>/<tag>/:
    traces.json         mỗi câu: query_id, status, answer, citations, events, token, thời gian, lỗi
    summary.json        tool use, số lượt search, tỷ lệ trả lời/từ chối, lỗi, token, chi phí,
                        Context Evidence Rate trên context cuối (đối chiếu gold SAU khi chạy)
    reviews_template.json  khung nhãn để bạn tự đọc và chấm (answer_grade, citation_support, ...)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agent import LLMClient, run_agent  # noqa: E402
from eval_retrieval import group_hit  # noqa: E402
from retrievers import build, load_chunks  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["dev", "test"])
    ap.add_argument("--method", default="bm25", choices=["dense", "bm25", "hybrid", "rerank"])
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--budget", type=int, default=1600, help="Ký tự context mỗi lượt search")
    ap.add_argument("--neighbors", type=int, default=0)
    ap.add_argument("--max-searches", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="Chỉ chạy n câu đầu (thử nhanh)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    bench = [q for q in json.loads((ROOT / "data/benchmark/benchmark.json").read_text(encoding="utf-8"))
             if q["split"] == args.split][: args.limit]
    chunks = load_chunks()
    by_id = {c["id"]: c for c in chunks}
    retriever = build(args.method, chunks)
    llm = LLMClient()
    tag = args.tag or f"{args.method}-k{args.top_k}-b{args.budget}-nb{args.neighbors}"
    out = ROOT / "results" / "agent" / args.split / tag
    out.mkdir(parents=True, exist_ok=True)

    traces = []
    for q in bench:
        r = run_agent(q["question"], llm, retriever, chunks, filters=q.get("filter"), top_k=args.top_k,
                      budget=args.budget, neighbors=args.neighbors, max_searches=args.max_searches)
        r = {"query_id": q["id"], "split": q["split"], "answerable": q["answerable"], "question": q["question"], **r}
        if q["answerable"]:  # chấm context SAU khi agent chạy xong; gold không vào vòng lặp
            r["context_hit"] = group_hit([by_id[i] for i in r["final_context_ids"]], q["gold"])
        traces.append(r)
        print(f"{q['id']}: {r['status']:8} searches={r['n_searches']} {r['latency_s']}s  {r['answer'][:90]!r}")
        (out / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding="utf-8")

    ok = [t for t in traces if t["status"] != "error"]
    ans = [t for t in traces if t["answerable"] and t["status"] != "error"]
    una = [t for t in traces if not t["answerable"] and t["status"] != "error"]
    frac = lambda a, b: f"{a}/{b}" + (f" ({a / b:.0%})" if b else " (không áp dụng)")
    summary = {
        "run": tag, "split": args.split, "model": llm.model, "base_url": llm.base_url,
        "config": {"method": args.method, "retriever": getattr(retriever, "info", {}), "top_k": args.top_k,
                   "budget_chars_per_search": args.budget, "neighbors": args.neighbors,
                   "max_searches": args.max_searches, "temperature": llm.temperature},
        "n": len(traces), "errors": [t["query_id"] for t in traces if t["status"] == "error"],
        "tool_use": frac(sum(t["n_searches"] > 0 for t in ok), len(ok)),
        "searches_distribution": dict(sorted(Counter(t["n_searches"] for t in ok).items())),
        "context_evidence_rate": frac(sum(t["context_hit"] for t in ans), len(ans)),
        "answered_answerable": frac(sum(t["status"] == "answered" for t in ans), len(ans)),
        "abstained_unanswerable": frac(sum(t["status"] == "abstain" for t in una), len(una)),
        "answers_with_citation": frac(sum(bool(t["citations"]) for t in ok if t["status"] == "answered"),
                                      sum(t["status"] == "answered" for t in ok)),
        "invalid_citations_total": sum(len(t.get("invalid_citations", [])) for t in ok),
        "median_latency_s": statistics.median(t["latency_s"] for t in ok) if ok else None,
        "tokens": {k: sum(t["usage"][k] for t in traces) for k in ("prompt_tokens", "completion_tokens")},
        "cost_usd": round(sum(t["cost_usd"] for t in traces), 4),
        "note": "answer_grade, citation_support, abstention_correct phải tự chấm trong reviews.json",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    template = [{"query_id": t["query_id"], "answerable": t["answerable"], "status": t["status"],
                 "answer": t["answer"], "citations": t["citations"],
                 "citation_support": [None] * len(t["citations"]),
                 **({"answer_grade": None} if t["answerable"] else {"abstention_correct": None}),
                 "reason": ""} for t in traces if t["status"] != "error"]
    (out / "reviews_template.json").write_text(json.dumps(template, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
