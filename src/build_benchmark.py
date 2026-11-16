"""Ghép benchmark Day 7 từ câu Day 6 (đã chuyển schema) và câu mới đã duyệt.

    python src/build_benchmark.py

Đầu vào:  data/day6/questions_day6.json, data/benchmark/new_questions.json, data/day6/chunks.json
Đầu ra (data/benchmark/):
    benchmark.json            toàn bộ câu (có đáp án + thiếu evidence)
    questions_answerable.json chỉ câu có đáp án, dùng cho retrieval metrics và starter/baseline.py
    cases.json                id, split, answerable cho starter/agent_metrics.py
Không file nào trong số này được đưa vào agent hay retriever (chỉ dùng để chấm).
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "benchmark"


def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def main():
    chunks = load("data/day6/chunks.json")
    qs = load("data/day6/questions_day6.json") + load("data/benchmark/new_questions.json")
    ids = [q["id"] for q in qs]
    assert len(ids) == len(set(ids)), "trùng ID câu hỏi"
    for q in qs:
        assert q["split"] in ("dev", "test") and q["question"].strip(), q["id"]
        if q["answerable"]:
            assert q["expected_answer"].strip() and q["gold"], q["id"]
            for g in q["gold"]:
                assert any(c["source_id"] == g["source_id"] and c["page"] == g["page"] and g["evidence"] in c["text"]
                           for c in chunks), f"{q['id']}: evidence không nằm trọn trong chunk"
        else:
            assert q["expected_behavior"] == "search_and_abstain" and q["absence_reason"].strip(), q["id"]
    OUT.mkdir(parents=True, exist_ok=True)
    dump = lambda name, obj: (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    dump("benchmark.json", qs)
    dump("questions_answerable.json", [q for q in qs if q["answerable"]])
    dump("cases.json", [{"id": q["id"], "split": q["split"], "answerable": q["answerable"]} for q in qs])
    print(Counter((q["split"], "answerable" if q["answerable"] else "abstain") for q in qs))


if __name__ == "__main__":
    main()
