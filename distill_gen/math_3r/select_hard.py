"""Select the harder 1000 problems from each source -> 2000-problem hard subset.

Difficulty signals (per the survey; the two sources are NOT cross-comparable, so each is ranked
within its own source):
  - FineProofs : `fp_qwen_reward` (Qwen3-4B-Thinking pass@128, gpt-oss-20b graded). LOW = harder.
                 Take the 1000 lowest-reward problems (origin FineProofs or both; reward present).
  - Nemotron   : per-problem MEAN response length (reasoning_content + content) over its proofs.
                 All Nemotron proofs are one config (DeepSeek-V4-Pro, thinking, no tools, 1 turn,
                 AoPS, ultra_v3) so length is comparable within the source. LONGER = harder.
                 Take the 1000 longest, among origin == Nemotron-Math-Proofs-v2 (disjoint from the
                 FineProofs side, so 2000 distinct problems).

Output: hard2000.parquet = the selected rows of distill_gen/problems/problems.parquet, plus
difficulty annotation columns. Drop-in --input for run.py.

Run:
    uv run python distill_gen/math_3r/select_hard.py
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
CANON = REPO / "distill_gen" / "problems" / "problems.parquet"
NM_GLOB = str(REPO / "datasets" / "training_l2_v2" /
              "dataset=nemotron-math-proofs-v2" / "domain=proof" / "*.parquet")
OUT = Path(__file__).resolve().parent / "hard2000.parquet"
N_PER_SOURCE = 1000


def pid_of(p: str) -> str:
    return hashlib.blake2b((p or "").encode(), digest_size=8).hexdigest()


def J(x):
    return json.loads(x, strict=False) if isinstance(x, str) else x


def main() -> None:
    table = pq.read_table(CANON)
    cols = {c: table.column(c).to_pylist() for c in table.column_names}
    n = table.num_rows
    origin = cols["origin"]
    reward = cols["fp_qwen_reward"]

    # member text -> canonical row index (so Nemotron proofs map to canonical problems)
    member2canon: dict[str, int] = {}
    for i, (ct, mem) in enumerate(zip(cols["problem"], cols["members"])):
        member2canon[ct] = i
        for m in J(mem):
            member2canon[m["problem"]] = i

    # ---- Nemotron per-canonical mean response length ----
    # The host flips bytes under heavy reads (memory host-memory-instability): a row group's
    # to_pylist() can raise UnicodeDecodeError on a transient flip. Retry the read; if a row is
    # persistently corrupt, decode the rest row-by-row and skip (and count) only the bad ones.
    bad = 0

    def rows_of(pf, rg):
        nonlocal bad
        for _ in range(3):
            try:
                tb = pf.read_row_group(rg, columns=["meta", "messages"])
                return list(zip(tb.column("meta").to_pylist(), tb.column("messages").to_pylist()))
            except UnicodeDecodeError:
                continue
        tb = pf.read_row_group(rg, columns=["meta", "messages"])
        out = []
        for j in range(tb.num_rows):
            try:
                r = tb.slice(j, 1)
                out.append((r.column("meta").to_pylist()[0], r.column("messages").to_pylist()[0]))
            except UnicodeDecodeError:
                bad += 1
        return out

    agg: dict[int, list[int]] = {}  # canon idx -> [sum_len, count]
    for f in sorted(glob.glob(NM_GLOB)):
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            for meta, ms in rows_of(pf, rg):
                d = J(meta)
                idx = member2canon.get(d.get("problem"))
                if idx is None:
                    continue
                a = [m for m in J(ms) if m.get("role") == "assistant"]
                if not a:
                    continue
                rlen = len(a[-1].get("reasoning_content") or "") + len(a[-1].get("content") or "")
                s = agg.setdefault(idx, [0, 0])
                s[0] += rlen
                s[1] += 1
    if bad:
        print(f"[hard3000] WARNING: skipped {bad} undecodable Nemotron rows (bit-flip/corruption)")
    mean_len = {i: v[0] / v[1] for i, v in agg.items()}
    n_resp = {i: v[1] for i, v in agg.items()}

    # ---- FineProofs side: 1500 lowest reward ----
    fp_idx = [i for i in range(n) if reward[i] is not None]
    fp_idx.sort(key=lambda i: (reward[i], pid_of(cols["problem"][i])))  # stable tie-break
    fp_sel = fp_idx[:N_PER_SOURCE]

    # ---- Nemotron side: 1500 longest mean response, Nemotron-only origin ----
    nm_idx = [i for i in range(n) if origin[i] == "Nemotron-Math-Proofs-v2" and i in mean_len]
    nm_idx.sort(key=lambda i: (-mean_len[i], pid_of(cols["problem"][i])))
    nm_sel = nm_idx[:N_PER_SOURCE]

    assert not (set(fp_sel) & set(nm_sel)), "FineProofs and Nemotron selections overlap"
    sel = fp_sel + nm_sel

    # ---- build output: selected rows + difficulty annotation ----
    sub = table.take(sel)
    diff_source, diff_value, diff_rank, diff_nresp = [], [], [], []
    for rank, i in enumerate(fp_sel):
        diff_source.append("fp_qwen_reward"); diff_value.append(float(reward[i]))
        diff_rank.append(rank + 1); diff_nresp.append(None)
    for rank, i in enumerate(nm_sel):
        diff_source.append("nemotron_resp_len"); diff_value.append(float(mean_len[i]))
        diff_rank.append(rank + 1); diff_nresp.append(n_resp[i])
    sub = (sub.append_column("difficulty_source", pa.array(diff_source, pa.string()))
              .append_column("difficulty_value", pa.array(diff_value, pa.float64()))
              .append_column("difficulty_rank_in_source", pa.array(diff_rank, pa.int64()))
              .append_column("nemotron_n_responses", pa.array(diff_nresp, pa.int64())))

    pq.write_table(sub, OUT, compression="zstd")
    back = pq.read_table(OUT)
    assert back.num_rows == len(sel), "row count mismatch on readback"
    assert len({pid_of(p) for p in back.column("problem").to_pylist()}) == len(sel), "dup problems"

    fp_rewards = [reward[i] for i in fp_sel]
    nm_lens = [mean_len[i] for i in nm_sel]
    print(f"[hard2000] wrote {OUT} rows={back.num_rows} ({OUT.stat().st_size/1e6:.1f}MB) -- readback OK")
    print(f"  FineProofs {N_PER_SOURCE}: fp_qwen_reward in [{min(fp_rewards):.3f}, {max(fp_rewards):.3f}] "
          f"(cutoff <= {max(fp_rewards):.3f})")
    print(f"  Nemotron   {N_PER_SOURCE}: mean_resp_len in [{int(min(nm_lens))}, {int(max(nm_lens))}] "
          f"(cutoff >= {int(min(nm_lens))} chars)")


if __name__ == "__main__":
    main()
