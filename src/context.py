"""Chọn và đóng gói context gửi model.

pack(results, chunks, budget, neighbors):
- Dedup theo chunk ID, giữ thứ tự xếp hạng.
- neighbors=n: sau mỗi chunk được chọn, thêm tối đa n chunk liền trước/liền sau CÙNG TRANG
  (theo char_start trong trang). Lý do: chunk 350 ký tự hay cắt câu trả lời sang chunk bên cạnh
  (lỗi dev t18, q10). Chunk liền kề được ghi là `neighbor_of`.
- Mỗi block có header [id | source | page]; chunk không vừa budget (ký tự, gồm header) bị bỏ qua,
  giống starter pack_context.
"""
from __future__ import annotations


def neighbor_index(chunks: list[dict]) -> dict[str, list[dict]]:
    by_page: dict[tuple, list[dict]] = {}
    for c in chunks:
        by_page.setdefault((c["source_id"], c["page"]), []).append(c)
    out = {}
    for page_chunks in by_page.values():
        page_chunks.sort(key=lambda c: c["metadata"].get("char_start", 0))
        for i, c in enumerate(page_chunks):
            out[c["id"]] = page_chunks
    return out


def pack(results: list[dict], all_chunks: list[dict] | None = None, budget: int = 800,
         neighbors: int = 0) -> tuple[str, list[str], list[dict]]:
    index = neighbor_index(all_chunks) if neighbors and all_chunks else {}
    order: list[dict] = []
    seen: set[str] = set()
    for r in results:
        cands = [r]
        if neighbors and r["id"] in index:
            page = index[r["id"]]
            i = next(j for j, c in enumerate(page) if c["id"] == r["id"])
            near = page[max(0, i - neighbors): i] + page[i + 1: i + 1 + neighbors]
            # Giữ đúng thứ tự trong trang để model đọc liền mạch.
            cands = sorted([r] + [{**c, "neighbor_of": r["id"]} for c in near],
                           key=lambda c: c["metadata"].get("char_start", 0))
        for c in cands:
            if c["id"] not in seen:
                seen.add(c["id"])
                order.append(c)
    blocks, kept = [], []
    for c in order:
        block = f"[{c['id']} | {c['source']} | page {c['page']}]\n{c['text']}"
        if len("\n\n".join(blocks + [block])) <= budget:
            blocks.append(block)
            kept.append(c)
    return "\n\n".join(blocks), [c["id"] for c in kept], kept
