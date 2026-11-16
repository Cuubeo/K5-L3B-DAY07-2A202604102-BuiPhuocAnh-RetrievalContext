"""Test vòng lặp agent bằng LLM giả lập. Chỉ kiểm tra code, KHÔNG phải benchmark."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent import ABSTAIN, run_agent  # noqa: E402
from retrievers import BM25Retriever  # noqa: E402

CHUNKS = [{"id": "d:p1:para350:0", "text": "The judge model is GPT-5.4.", "source_id": "d", "source": "x",
           "page": 1, "metadata": {"section": "D", "char_start": 0}}]


class ScriptedLLM:
    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def chat(self, messages, tools=None, tool_choice="auto"):
        self.seen.append((json.loads(json.dumps(messages)), tool_choice))
        return {"choices": [{"message": self.replies.pop(0)}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def tool(q, i="c1"):
    return {"content": "", "tool_calls": [{"id": i, "type": "function",
                                           "function": {"name": "search", "arguments": json.dumps({"query": q})}}]}


class TestAgentLoop(unittest.TestCase):
    def test_search_then_cited_answer(self):
        llm = ScriptedLLM([tool("judge model"), {"content": "GPT-5.4 [d:p1:para350:0]."}])
        r = run_agent("Which judge?", llm, BM25Retriever(CHUNKS), CHUNKS, filters={"section": "D"})
        self.assertEqual(r["status"], "answered")
        self.assertEqual(r["citations"], ["d:p1:para350:0"])
        self.assertEqual(r["n_searches"], 1)
        self.assertIn("[d:p1:para350:0 |", llm.seen[1][0][-1]["content"])  # context đưa cho model
        self.assertNotIn("expected_answer", json.dumps(llm.seen))

    def test_citation_not_in_context_abstains(self):
        llm = ScriptedLLM([tool("judge"), {"content": "GPT-5.4 [d:p9:para350:3]."}])
        r = run_agent("Which judge?", llm, BM25Retriever(CHUNKS), CHUNKS)
        self.assertEqual(r["status"], "abstain")
        self.assertEqual(r["invalid_citations"], ["d:p9:para350:3"])

    def test_explicit_abstain_and_search_limit(self):
        llm = ScriptedLLM([tool("a", "1"), tool("b", "2"), {"content": f"{ABSTAIN}: not in paper"}])
        r = run_agent("Cost in USD?", llm, BM25Retriever(CHUNKS), CHUNKS, max_searches=1)
        self.assertEqual(r["status"], "abstain")
        self.assertEqual(r["n_searches"], 1)
        self.assertEqual(llm.seen[-1][1], "none")  # hết lượt -> buộc trả lời

    def test_provider_error_is_reported(self):
        class Boom:
            def chat(self, *a, **k):
                raise RuntimeError("HTTP 401")
        r = run_agent("q", Boom(), BM25Retriever(CHUNKS), CHUNKS)
        self.assertEqual(r["status"], "error")
        self.assertIn("401", r["error"])


if __name__ == "__main__":
    unittest.main()
