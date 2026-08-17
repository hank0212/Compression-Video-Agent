"""Build the LVBench temporal-localization report.

  python -m longvt_compression.fast_agent.present_lvbench

One question: **can Qwen3-VL-8B find the right moment in a long video by itself?**

LVBench ships a ground-truth evidence span per question, which VideoMME does not,
so the agent's own crop windows can be scored against where the answer actually is.
Three arms share one harness, one prompt, one 64-frame skim -- only WHO PICKS THE
WINDOW differs (nobody / the model / the harness at the GT span), which isolates
localization from perception.

No case studies here -- this is the measurement only. Every number and both figures
are recomputed from the run artifacts at build time.
"""

import base64
import io
import json
import math
import os
import statistics as st

import nbformat as nbf

from . import config
from .case_notes import rescore

FEATURES = "/local1/cfyang/hanklin/outputs/fast_agent/analysis/features.json"

DIRECT, CROP, ORACLE = ("lvbench_baseline_full", "lvbench_crop_full",
                        "lvbench_oracle_crop_cot")

GREEN, RED, BLUE, PURPLE, GREY = "#228833", "#cc3311", "#4477aa", "#aa3377", "#888888"


def load_full(run: str) -> dict:
    """All trajectories of a run, following `traj_path` out of results.jsonl.

    NOT `case_notes.load_run`, which globs `<run>/traj/*.json` and so silently
    drops the ~30% of questions whose answers were REUSED from an earlier run
    (`reused_from`, e.g. the pilot) and whose trajectory lives in that run's
    directory. Globbing gives 140/200 here; this gives all 200.
    """
    out = {}
    for line in open(os.path.join(config.RUN_ROOT, run, "results.jsonl")):
        r = json.loads(line)
        p = r.get("traj_path")
        if p and os.path.exists(p):
            out[str(r["question_id"])] = json.load(open(p))
    return out


# ----------------------------------------------------------------- primitives
def union_len(spans) -> float:
    """Total seconds covered by `spans`, overlaps counted once."""
    total, cs, ce = 0.0, None, None
    for s, e in sorted(spans):
        if ce is None or s > ce:
            if ce is not None:
                total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return total + (ce - cs) if ce is not None else 0.0


def pearson(a, b) -> float:
    if len(a) < 2:
        return 0.0
    ma, mb = st.mean(a), st.mean(b)
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / den if den else 0.0


def compute(runs: dict, feats: dict) -> dict:
    d, c, o = runs[DIRECT], runs[CROP], runs[ORACLE]
    paired = sorted(set(d) & set(c), key=int)
    ev = [q for q in paired if q in feats]  # coverage needs a parsed evidence span

    def acc(run, qs):
        return 100 * sum(rescore(run[q]) == run[q]["gold"] for q in qs) / len(qs) if qs else 0.0

    cov = lambda q: feats[q].get("crop_cov") or 0.0
    ncall = lambda q: feats[q].get("crop_ncalls") or 0
    landed = [q for q in ev if ncall(q) > 0 and cov(q) > 0]
    missed = [q for q in ev if ncall(q) > 0 and cov(q) == 0]
    nocall = [q for q in ev if ncall(q) == 0]

    # Where the evidence is vs where it looked, both as % of video duration.
    pts = []
    for q in ev:
        f = feats[q]
        spans = f.get("crop_spans") or []
        if not spans or f.get("ev_start") is None:
            continue
        dur, evc = f["duration"], 0.5 * (f["ev_start"] + f["ev_end"])
        best = min(spans, key=lambda s: abs(0.5 * (s[0] + s[1]) - evc))
        pts.append((100 * evc / dur, 100 * 0.5 * (best[0] + best[1]) / dur, cov(q) > 0))

    widths = [e - s for q in ev for s, e in (feats[q].get("crop_spans") or [])]
    ev_w = [feats[q]["ev_end"] - feats[q]["ev_start"]
            for q in ev if feats[q].get("ev_start") is not None]
    seen = [100 * union_len(feats[q].get("crop_spans") or []) / feats[q]["duration"] for q in ev]
    ncalls = [ncall(q) for q in ev]
    pq = lambda v, p: sorted(v)[int(p * (len(v) - 1))]

    return {
        "n_paired": len(paired), "n_ev": len(ev),
        "acc": {k: acc(runs[k], ev) for k in (DIRECT, CROP, ORACLE)},
        "bands": [("落在證據上", landed, acc(c, landed)),
                  ("沒落在證據上", missed, acc(c, missed)),
                  ("從未呼叫工具", nocall, acc(c, nocall))],
        "acc_direct_missed": acc(d, missed),
        "called": landed + missed,
        "pts": pts,
        "r_all": pearson([p[0] for p in pts], [p[1] for p in pts]),
        "r_hit": pearson([p[0] for p in pts if p[2]], [p[1] for p in pts if p[2]]),
        "r_miss": pearson([p[0] for p in pts if not p[2]], [p[1] for p in pts if not p[2]]),
        "widths": widths, "ev_w": ev_w, "seen": seen,
        "w_med": st.median(widths), "w_p25": pq(widths, .25), "w_p75": pq(widths, .75),
        "ev_med": st.median(ev_w), "ev_p25": pq(ev_w, .25), "ev_p75": pq(ev_w, .75),
        "dur_med": st.median([feats[q]["duration"] for q in ev]),
        "seen_med": st.median(seen),
        "calls_med": st.median(ncalls),
        "calls_hist": {k: ncalls.count(k) for k in range(6)},
    }


# --------------------------------------------------------------------- figures
def _setup_cjk():
    """Pick a font that can draw the Chinese labels, else fall back to English."""
    import matplotlib
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Sans CJK TC",
                 "WenQuanYi Zen Hei", "Source Han Sans SC", "SimHei"):
        if name in have:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


def _embed(fig, width=1180) -> str:
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=82, optimize=True)
    return ('<img src="data:image/jpeg;base64,'
            + base64.b64encode(out.getvalue()).decode() + '" style="max-width:100%"/>')


def fig_localization(S: dict, cjk: bool) -> str:
    """The localization test: did it look where the answer is, and did that pay?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = (("證據在影片的位置 (%)", "它去看的位置 (%)", "有沒有看對地方",
          "落在證據上", "沒落在證據上", "accuracy (%)", "落點決定準確率",
          "direct 無工具", "oracle 強制對準", "從未呼叫工具") if cjk else
         ("where the evidence is (% of video)", "where it looked (% of video)",
          "Did it look in the right place", "landed on evidence", "missed",
          "accuracy (%)", "Landing decides accuracy", "direct (no tool)",
          "oracle (forced)", "never called"))
    band_lbl = [T[3], T[4], T[9]]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.2),
                                  gridspec_kw={"width_ratios": [1.15, 1]})
    hit = [p for p in S["pts"] if p[2]]
    mis = [p for p in S["pts"] if not p[2]]
    ax.plot([0, 100], [0, 100], "--", color=GREY, lw=1.2, zorder=1)
    ax.scatter([p[0] for p in mis], [p[1] for p in mis], s=26, alpha=.6, color=RED,
               label=f"{T[4]}  n={len(mis)}  r={S['r_miss']:.2f}", zorder=2)
    ax.scatter([p[0] for p in hit], [p[1] for p in hit], s=36, alpha=.9, color=GREEN,
               label=f"{T[3]}  n={len(hit)}  r={S['r_hit']:.2f}", zorder=3)
    ax.set_xlabel(T[0])
    ax.set_ylabel(T[1])
    ax.set_title(f"{T[2]}  (n={len(S['pts'])}, r={S['r_all']:.2f})", fontsize=12)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.grid(alpha=.25)
    ax.legend(fontsize=9, loc="upper left")

    vals = [b[2] for b in S["bands"]]
    ns = [len(b[1]) for b in S["bands"]]
    ax2.bar(range(3), vals, color=[GREEN, RED, GREY], width=.6, zorder=2)
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax2.text(i, v + 1.5, f"{v:.1f}%\nn={n}", ha="center", fontsize=10)
    ax2.axhline(S["acc"][DIRECT], ls="--", color=BLUE, lw=1.4,
                label=f"{T[7]} {S['acc'][DIRECT]:.1f}%")
    ax2.axhline(S["acc"][ORACLE], ls="--", color=PURPLE, lw=1.4,
                label=f"{T[8]} {S['acc'][ORACLE]:.1f}%")
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(band_lbl, fontsize=10)
    ax2.set_ylabel(T[5])
    ax2.set_ylim(0, 78)
    ax2.set_title(T[6], fontsize=12)
    ax2.grid(alpha=.25, axis="y")
    ax2.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    out = _embed(fig, 1280)
    plt.close(fig)
    return out


def fig_search(S: dict, cjk: bool) -> str:
    """How wide it looks, how often it calls, how much of the video it ever sees."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = (("長度 (秒, log)", "佔該群的 %", "窗口寬度 vs 證據寬度", "GT 證據", "crop 窗口",
          "每題呼叫次數", "題數", "呼叫次數兩極化", "整支影片被看過的比例 (%)",
          "搜尋觸及範圍", "中位數") if cjk else
         ("length (s, log)", "% of that population", "Window width vs evidence width", "GT evidence",
          "crop window", "calls per question", "questions", "Calls are bimodal",
          "% of video ever inspected", "Search reach", "median"))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    bins = [2 ** i for i in range(13)]
    ax = axes[0]
    # Percent of each population, not raw counts -- there are 344 crop windows
    # against 183 evidence spans, so counts would just show which list is longer.
    for vals, col, lbl, med in ((S["ev_w"], RED, T[3], S["ev_med"]),
                                (S["widths"], BLUE, T[4], S["w_med"])):
        ax.hist(vals, bins=bins, alpha=.65, color=col,
                weights=[100 / len(vals)] * len(vals),
                label=f"{lbl} ({T[10]} {med:.0f}s)")
    ax.set_xscale("log")
    ax.set_xlabel(T[0])
    ax.set_ylabel(T[1])
    ax.set_title(T[2], fontsize=12)
    ax.grid(alpha=.25)
    ax.legend(fontsize=9)

    ax = axes[1]
    ks = sorted(S["calls_hist"])
    ax.bar(ks, [S["calls_hist"][k] for k in ks], color=BLUE, width=.65)
    for k in ks:
        if S["calls_hist"][k]:
            ax.text(k, S["calls_hist"][k] + 1.5, S["calls_hist"][k], ha="center", fontsize=9)
    ax.set_xlabel(T[5])
    ax.set_ylabel(T[6])
    ax.set_title(f"{T[7]} ({T[10]} {S['calls_med']:.0f})", fontsize=12)
    ax.grid(alpha=.25, axis="y")

    ax = axes[2]
    ax.hist(S["seen"], bins=[0, 1, 2, 5, 10, 20, 50, 100], color=BLUE)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel(T[8])
    ax.set_ylabel(T[6])
    ax.set_title(f"{T[9]} ({T[10]} {S['seen_med']:.1f}%)", fontsize=12)
    ax.grid(alpha=.25)

    fig.tight_layout()
    out = _embed(fig, 1360)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------ text
def header_md(S: dict) -> str:
    return f"""# LVBench · 模型自己找得到那一刻嗎?

**測的是 temporal localization。** LVBench 每題附一段 ground-truth 證據區間
(VideoMME 沒有),所以 agent 自己挑的 crop 窗口可以拿去對答案真正在的地方。

Zero-shot Qwen3-VL-8B,一個工具(`crop_video`:≤128 幀 @1fps 全解析度 temporal crop),
每題最多 5 次呼叫。三個 arm 共用同一套 harness、prompt、64-frame skim 開場,
**只有「誰決定看哪裡」不同** —— 所以差異是定位,不是感知。

| arm | 誰選時間窗 | accuracy (n={S['n_ev']}) |
|---|---|---|
| `direct` | 沒有工具 | {S['acc'][DIRECT]:.1f}% |
| `crop` (autonomous) | **模型自己選** | {S['acc'][CROP]:.1f}% |
| `oracle_crop` | harness 強制對準 GT 證據 | **{S['acc'][ORACLE]:.1f}%** |

**自己選窗口一分沒賺;把窗口對準值 +{S['acc'][ORACLE] - S['acc'][CROP]:.0f}pp。**

> 影片中位數長 {S['dur_med'] / 60:.0f} 分鐘。全部用現行 parser 重新計分。
> 分析限在 {S['n_ev']}/{S['n_paired']} 題 —— 其餘 {S['n_paired'] - S['n_ev']} 題的
> `time_reference` 解不出合法區間。`oracle_crop` 用 `FA_ORACLE_COT=1`
> (舊版壓掉 CoT,57.0% → 61.5%),天花板應寫成 ~57–62%。
"""


def result_md(S: dict) -> str:
    landed, missed, _ = S["bands"]
    rows = "\n".join(
        f"| {lbl} | {len(qs)} | {100 * len(qs) / S['n_ev']:.0f}% | **{a:.1f}%** |"
        for lbl, qs, a in S["bands"])
    return f"""---
## 1 · 看中了多少 · 看中之後值多少

| crop 落點 | n | 佔比 | crop accuracy |
|---|---|---|---|
{rows}
| **全部** | {S['n_ev']} | 100% | {S['acc'][CROP]:.1f}% |

**{S['n_ev']} 題裡只有 {len(landed[1])} 題({100 * len(landed[1]) / S['n_ev']:.0f}%)落在證據上**
(只算有呼叫工具的 {len(S['called'])} 題也才 {100 * len(landed[1]) / len(S['called']):.0f}%)。

**落在證據上 {landed[2]:.1f}%,沒落在 {missed[2]:.1f}% —— 差 {landed[2] - missed[2]:.0f}pp。**
工具是有效的,失敗在「走不到」而不是「看不清」;總分打平是這兩群相消的結果。

> 沒落在證據上的那 {len(missed[1])} 題,crop 的 {missed[2]:.1f}% 比 direct 在**同一批題目**上的
> {S['acc_direct_missed']:.1f}% 還低 —— 搜錯地方不只是白費,它會把 skim 本來已經答對的題目洗掉。
"""


def search_md(S: dict) -> str:
    return f"""---
## 2 · 它是怎麼搜的

| | 中位數 | p25 – p75 |
|---|---|---|
| crop 窗口寬度 | {S['w_med']:.0f}s | {S['w_p25']:.0f} – {S['w_p75']:.0f}s |
| GT 證據寬度 | {S['ev_med']:.0f}s | {S['ev_p25']:.0f} – {S['ev_p75']:.0f}s |
| 每題呼叫次數 | {S['calls_med']:.0f} | — |
| 整支影片被看過的比例 | **{S['seen_med']:.1f}%** | — |

- **寬度是對的,位置是錯的.** 用 {S['w_med']:.0f}s 的窗口框 {S['ev_med']:.0f}s 的證據是合理邊際 ——
  它知道要多寬,不知道要去哪。
- **它只看過整支影片的 {S['seen_med']:.1f}%.** 中位數 {S['dur_med'] / 60:.0f} 分鐘的影片、
  中位數 {S['calls_med']:.0f} 次呼叫;5 次 × {S['w_med']:.0f}s 在結構上就搜不完一支長片。
- **呼叫次數兩極化.** {S['calls_hist'][0]} 題一次都不叫、{S['calls_hist'][5]} 題燒滿 5 次,
  中間幾乎是空的。不是「搜到滿意為止」,是「不搜」和「搜到死」。
"""


def verdict_md(S: dict) -> str:
    landed, missed, _ = S["bands"]
    return f"""---
## 結論 · 這個模型的 temporal localization 能力

1. **命中率 {100 * len(landed[1]) / S['n_ev']:.0f}%** —— {S['n_ev']} 題裡 {len(landed[1])} 題落在證據上。
2. **但落點不是隨機的.** 全體 r = {S['r_all']:.2f};拆開看,命中那群 r = {S['r_hit']:.2f}、
   沒命中那群 r = {S['r_miss']:.2f}。**要嘛它知道在哪(幾乎完美),要嘛它在猜 ——
   中間沒有「差不多對」。** 這是雙峰,不是一條有雜訊的迴歸線。
3. **命中就有回報.** {landed[2]:.1f}% vs {missed[2]:.1f}%;而強制對準的 oracle 是
   {S['acc'][ORACLE]:.1f}% —— 命中之後它已經取到大部分的值,剩下的差距屬於感知,不屬於定位。
4. **瓶頸是搜尋的觸及範圍.** 只看過影片的 {S['seen_med']:.1f}%,而寬度選得合理 ——
   缺的不是解析度、不是預算配置,是**知道要往哪裡去**。

**可訓練的量就是命中率。** Reward 應該直接打在 span 與證據的重疊上、獨立於答案對錯;
{S['calls_hist'][0]} 不叫 / {S['calls_hist'][5]} 燒滿的兩極分布則指向一條 STOP 規則
(round 0 已覆蓋證據就停)。
"""


# ------------------------------------------------------------------------ main
def main():
    runs = {r: load_full(r) for r in (DIRECT, CROP, ORACLE)}
    feats = {str(r["qid"]): r for r in json.load(open(FEATURES))}
    S = compute(runs, feats)
    cjk = _setup_cjk()
    if not cjk:
        print("note: no CJK font found, figure labels fall back to English")

    cells = [nbf.v4.new_markdown_cell(header_md(S)),
             nbf.v4.new_markdown_cell(result_md(S)),
             nbf.v4.new_markdown_cell(fig_localization(S, cjk)),
             nbf.v4.new_markdown_cell(search_md(S)),
             nbf.v4.new_markdown_cell(fig_search(S, cjk)),
             nbf.v4.new_markdown_cell(verdict_md(S))]

    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    path = os.path.join(os.path.dirname(__file__), "PRESENT_lvbench_localization.ipynb")
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB)")
    return path


if __name__ == "__main__":
    main()
