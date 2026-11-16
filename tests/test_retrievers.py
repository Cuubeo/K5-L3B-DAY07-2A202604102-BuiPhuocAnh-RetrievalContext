"""Test code retrieval Day 7. Stub chỉ để kiểm tra logic, KHÔNG dùng để báo benchmark."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "starter"))
from retrievers import BM25Retriever, HybridRRF, tokenize  # noqa: E402
from eval_retrieval import group_hit  # noqa: E402
from context import pack  # noqa: E402

CHUNKS = [
    {"id": "d:p1:0", "text": "ProgramSelectBench has 16,977 programs.", "source_id": "d", "source": "x", "page": 1, "metadata": {"section": "1"}},
    {"id": "d:p2:0", "text": "Greedy search uses 5 drafts.", "source_id": "d", "source": "x", "page": 2, "metadata": {"section": "A"}},
    {"id": "d:p3:0", "text": "The judge model is GPT-5.4.", "source_id": "d", "source": "x", "page": 3, "metadata": {"section": "D"}},
]


class Stub:
    def __init__(self, name, order):
        self.name, self.order = name, order

    def search(self, query, filters=None, top_k=3):
        return [{**next(c for c in CHUNKS if c["id"] == i), "rank": r, "score": 1 / r}
                for r, i in enumerate(self.order[:top_k], 1)]


class TestRetrievers(unittest.TestCase):
    def test_tokenize_keeps_numbers(self):
        self.assertIn("16,977", tokenize("có 16,977 programs"))
        self.assertIn("gpt-5.4", tokenize("GPT-5.4 judge"))

    def test_bm25_rank_and_filter(self):
        r = BM25Retriever(CHUNKS)
        self.assertEqual(r.search("how many programs")[0]["id"], "d:p1:0")
        self.assertEqual([h["id"] for h in r.search("programs", filters={"section": "A"})], [])
        with self.assertRaises(ValueError):
            r.search("x", filters={"page": 1})

    def test_rrf_fuses_ranks(self):
        h = HybridRRF([Stub("a", ["d:p1:0", "d:p2:0"]), Stub("b", ["d:p2:0", "d:p3:0"])], k=60)
        out = h.search("q", top_k=3)
        self.assertEqual(out[0]["id"], "d:p2:0")  # rank 2 + rank 1 thắng rank 1 đơn lẻ
        self.assertEqual([o["rank"] for o in out], [1, 2, 3])
        self.assertEqual(out[0]["member_ranks"], {"a": 2, "b": 1})

    def test_group_hit_requires_all_quotes_in_group(self):
        gold = [{"source_id": "d", "page": 1, "evidence": "16,977", "group": 0},
                {"source_id": "d", "page": 2, "evidence": "5 drafts", "group": 0}]
        self.assertFalse(group_hit(CHUNKS[:1], gold))
        self.assertTrue(group_hit(CHUNKS[:2], gold))
        self.assertFalse(group_hit([CHUNKS[1]], [{"source_id": "d", "page": 1, "evidence": "5 drafts"}]))


    def test_pack_neighbors_same_page_in_order(self):
        page = [{"id": f"d:p9:{i}", "text": f"t{i}", "source_id": "d", "source": "x", "page": 9,
                 "metadata": {"char_start": i * 10}} for i in range(4)]
        other = {"id": "d:p8:0", "text": "o", "source_id": "d", "source": "x", "page": 8, "metadata": {"char_start": 0}}
        _, ids, _ = pack([page[2]], page + [other], budget=10_000, neighbors=1)
        self.assertEqual(ids, ["d:p9:1", "d:p9:2", "d:p9:3"])
        _, ids, _ = pack([page[2], page[3]], page, budget=10_000, neighbors=0)
        self.assertEqual(ids, ["d:p9:2", "d:p9:3"])
        _, ids, _ = pack([page[0]], page, budget=30, neighbors=1)  # budget nhỏ: chỉ vừa 1 block
        self.assertEqual(len(ids), 1)


if __name__ == "__main__":
    unittest.main()
