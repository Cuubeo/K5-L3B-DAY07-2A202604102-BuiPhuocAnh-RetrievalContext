# Tiêu chí chọn cấu hình (ghi TRƯỚC khi chạy agent đầy đủ trên dev — 2026-11-16 11:30)

Các cấu hình agent so sánh trên dev (22 câu: 20 có đáp án + 2 thiếu evidence), cùng LLM `openai/gpt-4o-mini`
qua OpenRouter, temperature 0, max_searches 4, top_k 3:

| Tag | Retrieval | Budget/lượt | Neighbors | Vai trò |
|---|---|---|---|---|
| `dense-b800` | dense pgvector (index E Day 6) | 800 | 0 | Baseline |
| `rerank-nb1-b1600` | hybrid RRF top-20 → bge-reranker-v2-m3 | 1600 | ±1 | Ứng viên (tốt nhất ở retrieval dev: Hit@1 16/20, CER 17/20) |

Thứ tự tiêu chí (áp dụng lần lượt, chỉ sang tiêu chí sau khi hòa):
1. Grounded answer trung bình /2 trên câu có đáp án (tự chấm theo BENCHMARK.md).
2. Abstention đúng trên câu thiếu evidence, và ít từ chối nhầm câu có đáp án.
3. Citation hợp lệ (có trong context VÀ hỗ trợ câu trả lời) / tổng citation.
4. Thời gian trung vị mỗi câu và chi phí; nếu chất lượng chênh ≤ 1 câu, ưu tiên cấu hình rẻ/nhanh hơn.

Sau khi chọn: chốt cấu hình, rồi mới chạy baseline + cấu hình đã chọn trên test (7 câu) đúng một lần.

## Quyết định (sau dev, TRƯỚC khi chạy test — 2026-11-16 ~11:45)
Theo tiêu chí 1 (grounded answer dev): `rerank-nb1-b1600` 1.90/2 > `dense-b800` 1.65/2 (chênh 3 câu > ngưỡng 1 câu),
nên chọn `rerank-nb1-b1600` dù chậm gấp ~2 lần (11.8 s vs 6.1 s/câu) và tốn thêm ~40% token.
Nhãn dev: results/agent/dev/<run>/reviews.json. Test chạy đúng một lần sau quyết định này; không chỉnh gì sau khi xem test.
