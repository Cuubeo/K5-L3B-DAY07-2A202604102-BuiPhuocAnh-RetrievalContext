# Báo cáo Day 7 — Agentic RAG

**Học viên:** Bùi Phước Anh · **MSSV:** 2A202604102 · **Lớp:** L3B
**Repo:** https://github.com/Cuubeo/K5-L3B-DAY07-2A202604102-BuiPhuocAnh-RetrievalContext

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** bài báo arXiv 2611.17323v1 *"AgentSSL: Can MLE Agents Leverage Unlabeled Data?"* (CC BY 4.0), dùng lại corpus Day 6: 32/39 trang nội dung (bỏ tài liệu tham khảo, trang chỉ có hình). Index **E** của Day 6 (chunk theo đoạn, tối đa 350 ký tự): **436 chunk / 32 trang**, độ dài min/median/max = 22/280/350 ký tự. Xuất từ pgvector bằng `src/export_day6.py` → `data/day6/chunks.json` (đã đối chiếu: cùng ID và text với bản dựng lại từ `load_result.jsonl` của Day 6).
- **Vector store, embedding model, phiên bản và số chiều:** PostgreSQL 16 + pgvector 0.8.7 (Docker Day 6, bảng `chunks`, `strategy='E'`), cosine `<=>`, exact search (không ANN). Embedding `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, **384 chiều**, normalize, query/document cùng `encode()`, chạy local CPU, chi phí 0. Code kiểm tra cột `vector(384)` và số chiều model trước khi truy vấn.
- **Agent model, cách chạy và phân chia dev/test:** `openai/gpt-4o-mini` qua OpenRouter (API tương thích OpenAI, tool calling), temperature 0. Lý do chọn index E: tốt nhất trên dev Day 6. **Split:** 20 câu Day 6 đều đưa vào **dev**, vì test Day 6 đã bị xem khi chỉnh D/E/F ở Day 6; viết **test mới** 5 câu có đáp án ở các trang chưa dùng (p3, p4, p19, p20, p28). Tổng 29 câu: dev 20 có đáp án + 2 thiếu evidence; test 5 + 2.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:** câu có đáp án ghi `gold = [{source_id, page, evidence, group}]`; evidence là chuỗi nguyên văn nằm trọn trong một chunk đúng trang (kiểm tra tự động trong `src/build_benchmark.py`). Câu cần nhiều đoạn (q03, q13, t25) dùng `group`: trúng khi đủ mọi đoạn trích của một group. Câu thiếu evidence: tìm từ khóa trên **toàn bộ 39 trang PDF** (gồm trang không index) và ghi `absence_reason`; mỗi câu có bẫy gần chủ đề (vd n01 chỉ có GPU-giờ chứ không có USD; n04 chỉ có link Hugging Face/website, không có GitHub). **Sửa nhãn dev (có ghi lý do, trước khi chạy test):** q03 thêm evidence tương đương ở p4 (`data/benchmark/gold_fixes.json`).

## Retrieval và thử nghiệm

Cùng corpus (index E), cùng 20 câu dev có đáp án, top-k 3, filter `section` chỉ dùng khi câu hỏi có `filter` do người dùng cung cấp. Context budget tính bằng **ký tự** (gồm header `[id | source | page]`). Hit = có group evidence đủ trong top-1/top-3; Context Evidence Rate = đủ evidence trong context cuối. Lệnh: `python src/eval_retrieval.py ...`; kết quả `results/retrieval/<split>/<run>/`.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | **Dense** pgvector, k=3, filter section, budget 800 | Phương pháp bắt buộc trên vector store Day 6 | dev | 13/20 (65%) | 15/20 (75%) | 15/20 (75%) | 44 ms/câu, $0 |
| Phương pháp 2 | **BM25** (k1 1.5, b 0.75, tokenizer giữ số `16,977`, `GPT-5.4`), k=3, budget 800 | Câu có số/tên riêng hợp lexical | dev | 13/20 (65%) | 15/20 (75%) | 15/20 (75%) | 1.6 ms, $0 |
| Phương pháp 3 | **Hybrid RRF** (dense + BM25, mỗi bên top-20, k=60), k=3, budget 800 | Bù trượt của từng bên | dev | 14/20 (70%) | 14/20 (70%) | 14/20 (70%) | 47 ms, $0 |
| Đối chứng | Dense / BM25 / Hybrid, budget **1600**, không neighbor | Tách tác động của budget | dev | 13 / 13 / 14 | 15 / 15 / 14 | 15 / 15 / 14 | như trên |
| Cải thiện 1 | Như trên + **±1 chunk liền kề cùng trang**, budget 1600 | Chunk 350 ký tự cắt đáp án sang chunk bên cạnh (t18, q10) | dev | 13 / 13 / 14 | 15 / 15 / 14 | 15 / **16** / **15** | ~+0 ms, context ~1460 ký tự |
| Cải thiện 2 | **Hybrid top-20 → rerank** cross-encoder `BAAI/bge-reranker-v2-m3`, k=3, budget 800 | Cross-encoder phân biệt các đoạn prompt gần giống nhau; RRF đẩy gold khỏi top-3 (q08, q15) | dev | **16/20 (80%)** | **16/20 (80%)** | 16/20 (80%) | 6.9 s/câu (CPU), $0 |
| Cấu hình đã chọn | Rerank + ±1 neighbor, budget 1600 | Kết hợp cải thiện 1 + 2 | dev | 16/20 (80%) | 16/20 (80%) | **17/20 (85%)** | 7.0 s/câu, $0 |
| Cấu hình đã chọn | như trên | — | test | 4/5 (80%) | 5/5 (100%) | 5/5 (100%) | 7.7 s/câu, $0 |
| Baseline | Dense, budget 800 | — | test | 4/5 (80%) | 5/5 (100%) | 5/5 (100%) | 57 ms/câu, $0 |

Tham khảo: TF-IDF của starter trên cùng dev (định nghĩa hit một đoạn trích): 9/20 · 13/20 · 12/20.

**Hai lỗi dev, nguyên nhân, cách thử:**
1. **t18 — đáp án bị cắt sang chunk liền kề.** Dense lấy p9 c7, BM25 lấy p9 c9; câu "Evolutionary consistently produces the highest diversity scores" nằm ở **p9 c8** giữa hai chunk. Giả thuyết: chunk 350 ký tự tách đoạn so sánh khỏi kết luận. Thử: thêm ±1 chunk liền kề → context BM25 15→16/20; với rerank 16→17/20; agent rerank trả lời đúng "Evolutionary" trong khi agent dense trả lời sai "MCTS". Đối chứng tăng budget mà không neighbor: không đổi → tác động đến từ neighbor, không phải budget. q10 không sửa được vì đáp án cách 2 chunk.
2. **q08 / q15 — RRF làm rơi gold, prompt gần giống nhau.** BM25 xếp gold q15 hạng 2 nhưng hybrid đẩy ra khỏi top-3; q08 gold ở hạng 9 trong candidate hybrid. Giả thuyết: cross-encoder đọc query + chunk cùng lúc sẽ xếp lại đúng. Thử rerank: q08 hạng 9→1, q15 hạng 8→1; Hit@1 14→16/20. Không sửa được q14 (câu hỏi ngôn ngữ tự nhiên, đáp án nằm trong code `alpha: float = 0.85, iters: int = 25`; rerank top-1 chỉ 0.05) và t20 (gold thiếu evidence tương đương ở p26 c13, xem bên dưới).
3. (Lỗi nhãn) **q03:** dense trả p4 c9 có cùng nội dung "two core challenges…" nhưng gold chỉ có p2 → đã bổ sung gold dev. **t20** cũng thiếu evidence tương đương p26 c13 (nằm trong *Debug operator prompt*); chưa sửa gold, nên Context Evidence Rate của t20 bị tính trượt dù agent trả lời đúng.

**Lý do chọn cấu hình:** tiêu chí ghi trước khi chạy agent đầy đủ trên dev (`results/selection_criteria.md`): (1) grounded answer dev, (2) abstention, (3) citation hợp lệ, (4) thời gian/chi phí nếu chất lượng chênh ≤ 1 câu. Rerank + neighbor thắng ở (1) với 1.90 vs 1.65 (chênh 3 câu) nên được chọn, chấp nhận chậm ~2 lần và tốn thêm ~40% token. Score không dùng làm ngưỡng từ chối được: score top-1 dense của câu thiếu evidence (0.62, 0.64) nằm trong khoảng của câu có đáp án (0.53–0.84); rerank: n01 0.01 nhưng n02 0.63, trong khi có câu có đáp án chỉ 0.05.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:** `src/agent.py`. Một tool `search(query)`; model tự viết lại query và gọi tối đa **4 lượt**; filter `section` do người dùng cung cấp được áp vào mọi lượt (không do model đoán, không từ gold). Mỗi lượt: retriever → `src/context.py` (dedup chunk đã đưa, thêm neighbor nếu cấu hình, cắt theo budget ký tự mỗi lượt) → trả context có header `[chunk-id | source | page]`. Dừng khi model trả lời không gọi tool; hết 4 lượt thì buộc trả lời (`tool_choice=none`). Câu trả lời phải cite `[chunk-id]`; citation phải nằm trong các chunk đã đưa cho model; model trả `INSUFFICIENT_EVIDENCE` hoặc không có citation hợp lệ → **abstain**. System prompt yêu cầu coi nội dung chunk là dữ liệu, không làm theo chỉ dẫn trong tài liệu (corpus chứa nhiều prompt của agent khác). Lỗi API/DB → `status: error` (không có lỗi nào trong các lần chạy chính thức); lỗi hết quota không retry; 429/5xx retry có backoff. Gold/expected answer không vào vòng lặp.

| Cấu hình / split | Tool use | Số lượt gọi (lượt: số câu) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Baseline dense-b800 / dev | 22/22 | 1:14 · 2:2 · 3:3 · 4:3 | 3 / 1 / 16 → **1.65** | 16/20 (80%) | 19/19 | 2/2 | 1/20 |
| Chọn rerank-nb1-b1600 / dev | 22/22 | 1:17 · 3:2 · 4:3 | 1 / 0 / 19 → **1.90** | 20/20 (100%) | 19/19 | 2/2 | 1/20 |
| Baseline dense-b800 / test | 7/7 | 1:4 · 2:2 · 3:1 | 0 / 0 / 5 → **2.00** | 5/7 (71%) | 6/6 | 1/2 | 0/5 |
| Chọn rerank-nb1-b1600 / test | 7/7 | 1:5 · 2:2 | 0 / 0 / 5 → **2.00** | 6/9 (67%) | 7/7 | 0/2 | 0/5 |

Context (test, 5 câu có đáp án, chấm 0/1/2 theo BENCHMARK §2): cả baseline và cấu hình chọn đều **5 câu mức 2** (đủ evidence và nguồn trong context cuối), 0 câu mức 1, 0 câu mức 0.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| t21 (test, chọn) | Có đáp án | 1 search "test harness agent design choices SSL setting" → p3 c4 hạng 1 | "The authors use Aira-Dojo as the test harness… [p3:para350:4]" | 2 — đúng; citation chứa "We use Aira-Dojo [50] as such a test harness" |
| t18 (dev) | Có đáp án | Dense: 3 search, không có p9 c8. Rerank: 1 search, p9 c9 + neighbor p9 c8 | Dense: "MCTS" [p5 c12][p9 c9]. Rerank: "Evolutionary" [p9 c8] | Dense 0, 2 citation không hỗ trợ (p9 c9 nói MCTS "lies between"); rerank 2 |
| q10 (dev, dense) | Có đáp án | 4 search, không tìm được p9 c6 | "GPT-5.4" [p20 c16] | 0 — citation ID hợp lệ nhưng p20 c16 nói GPT-5.4 là LLM-as-judge, không nói về diversity |
| q15 (dev, dense) | Có đáp án | 1 search → p27 c9 | "torch.optim.AdamW" [p27 c9] | 1 — giá trị đúng nhưng p27 c9 thuộc *Improve* prompt, không chứng minh điều kiện "draft prompt" |
| q14 (dev, cả hai) | Có đáp án | 4 search tự viết lại ("Retinopathy program default alpha iterations"…), không có p34 c2 | `INSUFFICIENT_EVIDENCE` | 0 — từ chối nhầm; giới hạn với câu hỏi về code |
| n01 (dev, cả hai) | Thiếu evidence | 3–4 search về "cost of LLM API calls" | `INSUFFICIENT_EVIDENCE: No information on the total cost…` | Đúng — không dùng số 2,484 GPU-giờ thay cho USD |
| n03 (test, cả hai) | Thiếu evidence | 2 search "AgentSSL text classification…" | Khẳng định AgentSSL "strong performance in text classification", cite p1 c3 (8/12, +21.8%), p6 c14, p25 c8 | Sai — bịa; mọi citation nói về phân loại **ảnh**, đúng nguyên văn nhưng không hỗ trợ khẳng định |
| n04 (test) | Thiếu evidence | Dense 3 search; rerank 1 search → p1 c13 | Dense: `INSUFFICIENT_EVIDENCE`. Rerank: "GitHub URL… not explicitly mentioned… project website https://codiscovery.ai/AgentSSL [p1 c13]" | Dense đúng. Rerank tính **sai** theo định nghĩa (status answered); nội dung là từ chối "mềm", không bịa URL |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Trace agent: `results/agent/<split>/<run>/traces.json` (mọi tool call, query, kết quả, context ID, câu trả lời, citation, token, thời gian); tổng hợp `summary.json`; nhãn `reviews.json`; metric chính thức `agent_metrics.json` (`starter/agent_metrics.py`). Trace retrieval: `results/retrieval/<split>/<run>/`.
  - Thời gian trung vị/câu: dense-b800 6.1 s (dev) / 6.4 s (test); rerank-nb1-b1600 11.8 s / 13.0 s (rerank CPU ~7 s/lượt).
  - Token (prompt/completion): dense dev 29,060/1,973, test 8,371/647; rerank dev 41,519/1,891, test 9,464/655. Chi phí API LLM (ước tính từ token × giá `openai/gpt-4o-mini` trên OpenRouter quan sát được trong phản hồi API: $0.15/1M token vào, $0.60/1M token ra; lúc chạy `LLM_PRICE_*` chưa đặt nên `cost_usd` trong `summary.json` ghi 0): dense dev ≈ $0.0055, dense test ≈ $0.0016, rerank dev ≈ $0.0074, rerank test ≈ $0.0018; tổng ≈ $0.016. Embedding, BM25, reranker chạy local: $0. (Code đã sửa để lấy chi phí thật OpenRouter trả về trong các lần chạy sau.)
  - Lệnh: xem mục **Cách chạy** bên dưới.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - Trên dev, rerank + neighbor cải thiện retrieval (Hit@1 13→16/20, Context 15→17/20) và câu trả lời (1.65→1.90; citation hợp lệ 16/20→20/20). Đánh đổi: chậm ~2 lần mỗi câu, +43% prompt token, cần tải model reranker ~2.2 GB.
  - **Trên test, ưu thế không lặp lại:** cả hai đạt 2.00 trên 5 câu có đáp án (test mới dễ hơn: không có câu chunk-liền-kề/prompt gần giống), còn abstention của cấu hình chọn kém hơn (0/2 vs 1/2) và chậm gấp đôi. Test chỉ 7 câu nên chênh 1 câu là 14–50 điểm phần trăm; không đủ kết luận cấu hình nào tốt hơn trên test.
  - **Điểm yếu lớn nhất: từ chối với câu gần chủ đề.** n03 (text classification) cả hai cấu hình đều bịa bằng cách gán kết quả phân loại ảnh. Câu thiếu evidence ở dev (n01, n02) dễ hơn nên lỗi này không lộ ra khi chọn cấu hình — giới hạn của bộ dev. Không chỉnh prompt sau khi xem test; muốn sửa (vd yêu cầu model kiểm tra điều kiện/miền của câu hỏi trước khi trả lời, hoặc bước verify citation) cần thử trên dev và **viết test mới** để đánh giá độc lập.
  - Citation hợp lệ về ID chưa đủ: 3 citation của n03 và citation q10/q15 (dense) đúng ID, nằm trong context, nhưng không hỗ trợ câu trả lời — chỉ phát hiện được khi đọc đối chiếu.
  - Giới hạn khác: q14 (đáp án trong code) chưa giải quyết; gold t20 thiếu evidence tương đương; câu hỏi và nhãn do AI đề xuất, học viên duyệt → có nguy cơ thiên lệch; benchmark nhỏ (29 câu, 1 tài liệu); một model LLM duy nhất, temperature 0, mỗi cấu hình chạy 1 lần.

## Cách chạy

Python 3.12 (venv Day 6 có `sentence-transformers`, `psycopg`), Docker Day 6 đang chạy; `.env` theo `.env.example` (LLM_BASE_URL/LLM_API_KEY/LLM_MODEL; không commit `.env`).

```powershell
$py = "C:\VinAI\K5-L3B-Day06-2A202604102-DataIngestionEngineering\starter\.venv\Scripts\python.exe"
& $py src\export_day6.py --day6 C:\VinAI\K5-L3B-Day06-2A202604102-DataIngestionEngineering --strategy E   # chunk + câu Day 6
& $py src\build_benchmark.py                                                                             # benchmark 29 câu
& $py -m unittest discover -s tests -v                                                                   # test code (mock, không phải benchmark)
& $py src\eval_retrieval.py --methods dense bm25 hybrid --split dev
& $py src\eval_retrieval.py --methods dense bm25 hybrid --split dev --budget 1600 --tag b1600
& $py src\eval_retrieval.py --methods dense bm25 hybrid --split dev --budget 1600 --neighbors 1 --tag nb1-b1600
& $py src\eval_retrieval.py --methods rerank --split dev
& $py src\eval_retrieval.py --methods rerank --split dev --budget 1600 --neighbors 1 --tag nb1-b1600
& $py src\run_agent.py --split dev  --method dense  --budget 800  --neighbors 0 --tag dense-b800
& $py src\run_agent.py --split dev  --method rerank --budget 1600 --neighbors 1 --tag rerank-nb1-b1600
# sau khi chốt cấu hình (results/selection_criteria.md):
& $py src\run_agent.py --split test --method dense  --budget 800  --neighbors 0 --tag dense-b800
& $py src\run_agent.py --split test --method rerank --budget 1600 --neighbors 1 --tag rerank-nb1-b1600
& $py src\eval_retrieval.py --methods dense rerank --split test
& $py src\eval_retrieval.py --methods rerank --split test --budget 1600 --neighbors 1 --tag nb1-b1600
python starter\agent_metrics.py --cases data\benchmark\cases.json --traces results\agent\test\rerank-nb1-b1600\traces.json --reviews results\agent\test\rerank-nb1-b1600\reviews.json --split test --output results\agent\test\rerank-nb1-b1600\agent_metrics.json
```

Mã nguồn: `src/export_day6.py`, `src/build_benchmark.py`, `src/retrievers.py` (dense/BM25/hybrid/rerank), `src/context.py`, `src/eval_retrieval.py`, `src/agent.py`, `src/run_agent.py`; test `tests/test_retrievers.py`, `tests/test_agent_loop.py`.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện.

## Nguồn

- Lin et al., *AgentSSL: Can MLE Agents Leverage Unlabeled Data?*, arXiv:2611.17323v1, CC BY 4.0 — https://arxiv.org/abs/2611.17323v1
- Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; reranker: `BAAI/bge-reranker-v2-m3` (Hugging Face).
- Có dùng Claude hỗ trợ viết code, đề xuất câu hỏi/nhãn và soạn báo cáo; học viên chạy, kiểm tra và chịu trách nhiệm kết quả.
