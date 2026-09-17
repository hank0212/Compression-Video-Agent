# VidCom² 實作等價性驗證計畫 — v2

狀態：**只是計畫**。沒有修改任何 code，沒有啟動任何 experiment。
v1 2026-08-22 → **v2 2026-08-22**，依 review 意見改寫。
對照 RESEARCH.md `# THE FIVE RUNS` 與 2026-08-14 的 MEASURED PIXEL BUDGET 三項更正。

---

## v2 相對 v1 改了什麼（先講差異）

| # | v1 的問題 | v2 的處理 |
| --- | --- | --- |
| 1 | Stage 0 把「公式等價」與「integer allocation 差異」混在一起，用 `k_t 95% 相同` 當等價證明 | 拆成 **0A scorer parity**（應近乎完全一致）／ **0B allocation deviation**（**刻意不同**，只做量化描述，不是 pass/fail） |
| 2 | 用 `McNemar p > 0.05` 當 equivalence criterion | 刪除。改為 `D = Δ_off − Δ_ours` 的 **paired bootstrap CI**，並附上 power 計算——結論是 **±2 pp 的等價帶在 LVBench 全量下都幾乎打不到**（見 §6.3） |
| 3 | B/C 只比各自的 Δ，仍有 prompt × decoder × compressor 的交互 confound | 新增 **Stage D：validation-only matched inference mode**（同 proxy、同 prompt、greedy、16 tokens、同 parser），此時 prediction agreement 才有意義；LongVT prompt / T=0.7 只留給 research run |
| 4 | timestamp 疑慮與 compressor correctness 混寫 | 明確分開：那是 **official Qwen3 preprocessing / temporal-metadata path** 的問題，不是 compressor 問題。Stage A-lite 只 print 四個量 |
| 5 | 斷言「180 tok/frame = VidCom² validated regime」 | **撤回**。已知的只是「我們自己的 256f regime 比我們自己的 32f hi-res regime 每個 temporal step 壓縮兇 ~8×」。官方實際落點待 Stage A-lite 實測 |
| 6 | 建議排除 `localization_trivial` 題目 | 改為 **stratify 不刪題**：all / non-trivial / trivial / GT-hit / GT-miss 四層都報 |
| 7 | 7 個 stage 的線性計畫 | 縮成 **0A → 0B → 0C → A-lite → DECISION POINT**，通過就直接跑 `arm f − arm e`，不預先排 Stage B/C 300×4 |

---

## 0. 這份計畫要回答的問題，與它**不**回答的問題

**要回答**：我們 vLLM plugin 裡的 VidCom² scorer，在收到同一份 `video_embeds` 時，
是否與官方 `token_compressor/vidcom2` 產生同樣的 u_t / r_t / token ranking。

**明確不回答**（且不該用 accuracy 去回答）：
- 我們的 integer budget allocation 是否與官方相同 → **它刻意不同**，只做量化描述。
- 我們的 LVBench 數字是否應該等於官方任何已發表數字 → 兩邊 benchmark、prompt、decoder、
  operating point 全不同，沒有應該相等的理由。
- VidCom² 在 LVBench 上「應該」表現多好 → 那是研究問題，不是驗證問題。

---

## 1. 官方 VidCom² + Qwen3-VL + lmms-eval 路徑（trace 結果，v1 內容保留）

### 1.1 Repo 佈局

| 項目 | 事實 |
| --- | --- |
| repo | `github.com/xuyang-liu16/vidcom2` |
| branches | `main`(LLaVA)、`qwen`、`omni`、`llava`、`homepage` |
| 本機既有 checkout | `/home/cfyang/hanklin/vidcom2` = **main**，commit `944a17d`。含 `token_compressor/vidcom2/models/qwen3_vl.py`，**不含** qwen branch 的 `lmms_eval/` |
| 本次 trace 用 | `qwen` branch clone 到 scratchpad（commit `a85d485`） |
| 核心演算法檔 | `token_compressor/vidcom2/vidcom2.py` — **main 與 qwen branch byte-identical**（`diff` 無輸出） |

### 1.2 掛載方式

`lmms_eval/models/simple/qwen3_vl.py`，`COMPRESSOR=vidcom2` 時整段換掉 `Qwen3VLModel.forward`：

```python
self._model.model.forward = types.MethodType(Qwen3VLModel_forward, self._model.model)
```

模型：`Qwen3VLForConditionalGeneration.from_pretrained(..., dtype=bfloat16, attn_implementation=flash_attention_2)`。
wrapper 預設 `min_pixels=256*28*28`、`max_pixels=1605632`、`max_num_frames=32`、
`system_prompt="You are a helpful assistant."`。

### 1.3 插入點

`Qwen3VLModel_forward`：

1. `video_embeds, deepstack = self.get_video_features(pixel_values_videos, video_grid_thw)`
   → **ViT 之後、merger 之後**，已在 LLM hidden space（8B = 4096）。
2. `masked_scatter` 填回序列。
3. **position_ids 在 pruning 前算完**（`get_rope_index`）。
4. `compression_on` = `COMPRESSOR=="vidcom2"` ∧ 有 video ∧ prefill ∧ **`batch_size == 1`**（bs>1 靜默不壓縮）。
5. 每支影片獨立算 keep indices（見 §1.4）。
6. **deepstack 用同一組 token index 裁切**。
7. 被裁 token 從序列**實體移除**，`position_ids[:, :, keep]` 一起切
   → **存活 token 保留原 MRoPE t/h/w，不重新編號**。

### 1.4 演算法（reference 值）

| 名稱 | 值／公式 |
| --- | --- |
| channel 子集 | variance **最低** 50%（`select_low_var_channels(ratio=0.5)`），variance 在整支影片全部 token 上算 |
| normalize | `F.normalize(frames, dim=-1)` **之後**才算 centroid |
| video centroid | `frames.mean(dim=(0,1))` |
| frame centroid | `frames.mean(dim=1)` |
| kernel | `Σ_k exp(-d²/(2·2^k))`, k = −3…1 |
| u_t | `-vid_score.mean(dim=-1)` |
| r_t | `R·(1 + p − mean p)`, `p = softmax((u − max u)/0.01)`, `clamp(max=1)` |
| k_t | `round(r_t·tpf).clamp(min=1)` — **各幀獨立 round，總和漂移** |
| token 選擇 | `topk(vid_score + frame_score, k_t, largest=False)` |
| dtype | 跟著 model dtype（bf16），不 upcast |

`R_RATIO` = **retention**（預設 0.25）。除了「動態 tpf、deepstack 傳遞、position 切片」外，
**Qwen3 沒有任何演算法特化**。

### 1.5 frame sampling（**兩段抽幀**）

```python
image_inputs, video_inputs = process_vision_info(batched_messages)   # qwen_vl_utils 自己的政策
total_frames = video_inputs[0].shape[0]
indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
indices = np.unique(indices)
if total_frames - 1 not in indices:        # 可能變成 33 幀
    indices = np.append(indices, total_frames - 1); indices = np.unique(indices)
video_inputs[0] = video_inputs[0][indices]
inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, ...)
```

`max_pixels` 只透過 message dict 與 `AutoProcessor` 傳入，而 2026-08-14 我們已量到
`Qwen3VLVideoProcessor` 不吃 `max_pixels`、只吃 `size.longest_edge`。
**所以官方那條路每幀實際解析度必須實測，不能從參數推。**

### 1.6 官方 preprocessing 的 temporal-metadata 疑慮（**與 compressor 無關**）

wrapper 把預抽好的 tensor 丟給 processor，**沒有帶 `video_metadata`**。
`transformers/models/qwen3_vl/processing_qwen3_vl.py:202-208`：

> "Qwen3VL requires frame timestamps to construct prompts, but the `fps` of the input video
> could not be inferred… Defaulting to `fps=24`."

若成立：一支 1 小時影片抽 32 幀 → 被當成 32 frames @ 24 fps → `<t …>` ≈ 0–1.3 秒。
另外 `Qwen3VLVideoProcessor.do_sample_frames` 預設 `True`，對已抽好的 tensor 可能再抽一次。

**這是 `official Qwen3 preprocessing / temporal metadata path may not preserve the original
video timeline`，不是「VidCom² compressor 有問題」。** 兩者不可混談。

影響範圍：
- 若屬實，官方 HF + LVBench **不能**當 temporal-reasoning reference；
- 但**仍可**當 compressor reference（compressor 只看 embeds，不看時間標記）。

---

## 2. lmms-eval 控制了什麼（v1 內容保留）

**標準化**：dataset 載入、video 路徑解析、prompt 模板
（`question + "\nAnswer the question with the option letter"`，無句點）、
`max_new_tokens: 16`、答案 parsing（`extract_characters_regex` 取**第一個** [ABCD]）、
scoring、batching、多卡 sharding、`--log_samples`。

**不標準化**（全在 model wrapper）：frame decoding／sampling、解析度、system prompt、
**timestamps（根本沒傳）**、decoding 預設值（wrapper `temperature=0.0` greedy）、
compression 本身、HF vs vLLM。

→ **它是 dataset+prompt+scoring 的共同地基，不是視覺輸入的共同地基。**
我們懷疑的東西全在視覺輸入端，所以光用他們的 fork 不足以構成 controlled common ground。

---

## 3. 我們的 pipeline 與對照表（v1 內容保留，僅修正第 5 點的措辭）

| 階段 | 實作 |
| --- | --- |
| serving | vLLM **0.19.0** OpenAI server，Qwen3-VL-8B-Instruct |
| compressor 掛載 | `vidcom2_vllm/plugin.py` 註冊在 `vllm.general_plugins`（每 process 都跑，含 spawn 的 EngineCore）→ rebind `compute_retention_mask` 進 `vllm.model_executor.models.qwen3_vl` namespace |
| 啟用 | `FA_PRUNE_METHOD=vidcom2` + `--video-pruning-rate 0.75` |
| 插入點 | `qwen3_vl.py::_postprocess_video_embeds_evs`（1891 行起）→ post-merger，與官方同層 |
| 收到的 tensor | `cat([merger_out] + deepstack, dim=1)`（qwen3_vl.py:655），寬 4096×4 = 16,384 |
| scoring 輸入 | `x[:, :hidden_size]` → merger block，與官方一致 |
| q 語意 | vLLM `q` = pruning rate，`base = 1−q`；0.75 → 保留 0.25 |
| 總 budget | 強制 = `evs.compute_retained_tokens_count = max(tpf, int(T·tpf·(1−q)))` |
| k_t | ideal `r_t·tpf` 經 **largest-remainder apportionment** 配到剛好等於 target |
| position | `_get_expanded_positions`：未 pruning layout 算 mrope → `[retention_mask]`，語意與官方一致 |
| frame source | proxy mp4，n ∈ {64,256,768}，**max_side 448**，`fps = n/duration` 保留真實 timeline |
| 幀數 | `--media-io-kwargs '{"video":{"num_frames":N}}'` |
| 每幀 token | 由 `size.longest_edge = 25,165,824` 推出的整支影片 12,288 token 上限決定；實測 64f≈112、256f≈91 tok/grid-step（video-dependent） |
| prompt | LongVT no-tool system prompt + `<think>/<answer>` |
| decoding | T=0.7, top_p .8, top_k 20, rep 1.0, max_tokens 1024 |
| scoring | `data.extract_answer`（最後一個 `<answer>`，refusal-aware） |

### 3.1 對照表

| Component | Official | Ours | 判定 |
| --- | --- | --- | --- |
| checkpoint | Qwen3-VL-8B-Instruct bf16 FA2 | 同權重，vLLM bf16 | Same |
| serving stack | HF `generate` | vLLM 0.19 | Different（已知） |
| frame sampling | qwen_vl_utils → linspace(max_num_frames)（可能 33 幀） | ffmpeg proxy → `num_frames` 恆等重抽 | Different |
| video preprocessing | 原始影片，解析度由 processor 決定 | proxy 已降到 max_side 448 | **Different（重大）** |
| tok / grid-step | **未知，待 Stage A-lite 實測** | 64f=112 / 256f=91（實測） | **Unknown** |
| ViT + merger | HF | vLLM（同權重不同 kernel） | Same weights, numerics 待 0C 量測 |
| 插入點 | post-merger | post-merger | Same |
| scoring 輸入寬度 | 4096 | `x[:, :4096]` | Same |
| VidCom² scorer | 見 §1.4 | 逐行對照 | **Same（待 0A 確認）** |
| dtype | bf16 | bf16 | Same |
| ideal r_t | `R·(1+p−mean p)`, temp .01 | 同式，`base = 1−q` | **Same（待 0A 確認）** |
| integer k_t | 各幀獨立 round | largest-remainder | **刻意不同（0B 量化）** |
| 總 retention | ≈ R·N（漂移） | = `max(tpf, int(N·R))` | **刻意不同（0B 量化）** |
| Top-K | `largest=False` | 同 | Same |
| deepstack | 同一組 index | mask per-token 自動套用 | Same |
| positional | 先算 rope 再切 | 先算 unpruned mrope 再 mask | Same |
| `<t s>` timestamps | **疑似 fps=24 假時間（待驗）** | proxy 保留真實秒數 | Different（待驗） |
| prompt | lmms-eval post_prompt | LongVT `<think>/<answer>` | Different（重大） |
| decoding | greedy, 16 tokens | T=0.7, 1024 tokens | Different（重大） |
| scoring | 第一個 [ABCD] | 最後一個 `<answer>` | Different |
| batch gate | bs≠1 靜默不壓縮 | 不適用 | Different（不影響其數字） |

### 3.2 v1 的一項過度斷言（撤回）

v1 寫「VidCom² 驗證過的操作點是 ~180 tok/frame」。**這句沒有根據，撤回。**
180 是**我們自己 arm f** 的量測值，不是官方的。
目前有根據的說法只有：

> our current dense-frame regime (256f) is ~8× more aggressively compressed **per temporal step**
> than our own 32-frame high-resolution regime (arm e/f).

官方 Qwen3-VL 在 `max_num_frames=32` 下實際的 `grid_thw` / pre-prune / post-prune / tok-per-grid-step
**必須等 Stage A-lite dump 出來才能說**，因為 Qwen3 的 dynamic resolution + whole-video pixel budget
不是能從參數推的。

---

## 4. 驗證設計（v2）

### Stage 0A — **scorer parity**（同一份 `video_embeds`，兩份 scorer）

目的：證明我們的 scorer 忠實重現 VidCom² 的**連續量**部分。這一段兩邊**應該幾乎完全一致**。

```
                同一份 video_embeds (HF, bf16, 4096 寬)
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
  official scorer                      our scorer
  select_low_var_channels              select_low_var_channels
  → channel index set                  → channel index set
  compute_gaussian_scores              compute_gaussian_scores
  → vid_score, frame_score             → vid_score, frame_score
  → u_t = -vid_score.mean(-1)          → frame_importance
  compute_scales(u, R=0.25)            base=1-q → scales
  → ideal r_t                          → ideal r_t
  token_score = vid+frame              token_score
  → per-frame ranking (argsort)        → per-frame ranking
```

**逐項 dump 並比較**（不預設閾值，先看原始差異）：

| 量 | 為什麼要單獨看 |
| --- | --- |
| `T`, `tpf` | 結構；不同即結構性錯誤 |
| **channel index set 的交集比例** | `topk(var, k=2048, largest=False)` 在 4096 個 channel 上，若有大量 near-tie，兩實作可能選到**不同 channel 集合**，後面所有量都會跟著偏。這是最可能被誤判成「公式寫錯」的假陽性來源，必須先看 |
| `vid_score`, `frame_score` | max abs diff |
| `u_t` | max abs diff、Spearman ρ、rank 是否完全相同 |
| ideal `r_t`（尚未取整） | max abs diff |
| `token_score` | max abs diff |
| **per-frame token ranking** | 對每幀取 argsort，比較前 k 名的集合（k 掃 0.1/0.25/0.5·tpf），看 ranking 本身是否一致——**這是與 k_t 無關的 scorer 品質指標** |

**這一段完全不需要改任何既有 code**：官方 `select_low_var_channels / compute_gaussian_scores /
compute_scales` 與我們的 `retention.vidcom2_scores / _apportion` 都可直接 import。
（餵 4096 寬 tensor 時記得設 `VIDCOM2_MAIN_WIDTH=4096`，否則 `main_embed_width` 會走 argv 掃描的
 fallback 並印 warning。）

### Stage 0B — **allocation deviation**（刻意的差異，做量化描述，不是 pass/fail）

```
                    同一組 ideal r_t
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
  official                              ours
  k_t = round(r_t·tpf).clamp(min=1)     k_t = largest-remainder(r_t·tpf, target)
  總和自由漂移                            總和 == evs.compute_retained_tokens_count
```

**要報的量**：

- `sum(k_t)` 兩邊、以及各自對 `R·N` 的相對誤差
- `#frames with identical k_t`、`max |Δk_t|`、`Δk_t` 的分佈
- 用兩組 k_t 從**同一份 token ranking** 取 top-k → **selected-token Jaccard**（per-frame 與 global）
- 在兩個 operating point 各做一次：**64f（tpf≈112）與 256f（tpf≈91，低 tpf）**——
  `clamp(min=1)` 與 `max(tpf, ·)` 下限在低 tpf 時才會咬到

**結論的形式**應該長這樣（而不是「95% 相同所以等價」）：

> Our scorer reproduces VidCom²'s continuous quantities to within `<X>`;
> our serving integration differs **only** in the final integer budget allocation, which is
> required by vLLM's pre-sized placeholder run. That difference moves `<Y>` tokens per video
> (`<Z>%` of the retained set) at R=0.25.

### Stage 0C — **representation noise floor**（HF ViT vs vLLM ViT）

同一支 proxy、同樣 `num_frames`，各自走自己的 preprocessing：

1. **先比 `video_grid_thw`**。若不相等，就停在這裡——差異來自 preprocessing，不是 ViT，
   而且 0C 的 embedding 比較也無從做起（形狀不同）。這本身就是一個發現。
2. 若 grid 相同，比 `video_embeds`：per-token cosine 的中位數/最小值、max abs diff。
3. **把同一份 scorer 分別跑在兩組 embeds 上**，比 retained-mask Jaccard
   → 這就是後續一切比較的 **noise floor**。

**可選 0C′（更乾淨的隔離）**：把 vLLM processor 產生的 `pixel_values_videos` 直接餵 HF 的 ViT，
這樣就把 preprocessing 差異剔除，純測 ViT/merger 的 kernel 數值差。若 0C 的 grid 就不同，
0C′ 是唯一能拿到 ViT-only 數字的方法。

**0A 的判準必須引用 0C 的結果**：若 0C 顯示 mask Jaccard 只有 0.90，那 0A 要求 0.98 就不合理。
**先量再訂。**

### Stage A-lite — 官方 preprocessing 的四個數字（~20 題）

**只印，不評分**：

```
actual video duration (ffprobe)
sampled original frame indices / timestamps (wrapper 的 `indices`)
processor <t ...> values (decode 後 prompt 的 regex)
video_grid_thw  +  pre-prune tokens  +  post-prune tokens  +  tok per grid-step
```

兩個用途：
1. §1.6 的 temporal-metadata 疑慮拍板。
2. **拿到官方真正的 tok-per-grid-step**，才有資格談「arm f 是不是和官方同一個 operating point」。

需要新 conda env（lmms-eval + qwen_vl_utils + decord）。**Stage 0A/0B/0C 不需要**——
已驗證 `vllm` env 的 transformers 4.57.6 有 `Qwen3VLModelOutputWithPast`，
官方 `token_compressor` 可直接 import。

### DECISION POINT

- **0A/0B/0C/A-lite 都正常** → **不跑 Stage B/C**。compressor implementation 已驗得夠強。
  直接跑 `arm f − arm e`（同一個 32f hi-res regime 下的 vanilla vs VidCom² 25%）。
- **0A 有異常** → 停下來修 scorer，不跑任何 accuracy 實驗。
- **0A 過但 0C 顯示 grid 或 embeds 差很多** → 問題在 preprocessing／serving，不在 compressor；
  往 Stage D 走。

### Stage D — validation-only matched inference（**只在 DECISION POINT 之後、且真的需要時才做**）

若真要做 cross-stack 的行為比較，必須把 prompt/decoder 也 match，否則測的是
`f(visual representation, prompt, decoder)` 的交互而不是 compressor：

```
same LVBench qids   ·   same proxy   ·   same 64 frames
same user prompt (lmms-eval lvbench post_prompt)
same system prompt ("You are a helpful assistant.")
greedy   ·   max_new_tokens = 16   ·   same MCQ parser (extract_characters_regex)

        HF official + token_compressor     vs     vLLM + our plugin
```

此時 **prediction agreement 才是有意義的讀數**，而且它的統計效率遠高於 Δ 的差
（§6.3）。LongVT prompt / T=0.7 / 1024 tokens **只留給 research run，不參與 validation**。

### Stage E — 真正推進研究的下一步（不屬於驗證）

若 `arm f − arm e` 小、而 256f 的 compression loss 大，那已經抓到一個乾淨的現象：

> **VidCom² 的失效強烈取決於 per-temporal-step 的 token regime，而不是 retention ratio 本身。**

接著做固定 total budget 的 scaling curve：

```
frames:  32 → 64 → 128 → 256 → 512     (total visual token budget 固定)
report:  GT hit rate ↑  ·  tok per temporal step ↓
         conditional accuracy | hit  ·  conditional accuracy | miss  ·  overall accuracy
```

---

## 5. 要 dump 的量與定位階梯

### 5.1 dump 清單

| # | 量 | 官方 dump 點 | 我方 dump 點 | 屬於 |
| --- | --- | --- | --- | --- |
| 1 | `T`, `tpf` | `grid_thw` | `video_size_thw` | 0A |
| 2 | channel index set | `select_low_var_channels` 的 `topk_idx` | `vidcom2_scores` 的 `chan` | 0A |
| 3 | `vid_score` / `frame_score` | `compute_gaussian_scores` 回傳 | 同函式 | 0A |
| 4 | `u_t` | `-vid_score.mean(-1)` | `frame_importance` | 0A |
| 5 | ideal `r_t` | `compute_scales(...)` | inline `scales` | 0A |
| 6 | `token_score` | `vid+frame` | `token_score` | 0A |
| 7 | per-frame ranking（argsort 前 k） | 由 6 導出 | 由 6 導出 | 0A |
| 8 | `k_t` | `select_outlier_indices` 的 `ks` | `_apportion` 回傳 | 0B |
| 9 | `sum(k_t)` | 同上 | `compute_retained_tokens_count` | 0B |
| 10 | selected-token index / mask | `_map_linear_offset` 的 `keep` | `mask` | 0B |
| 11 | `video_grid_thw` | `inputs["video_grid_thw"]` | vLLM processor | 0C / A-lite |
| 12 | `video_embeds`（merger, 4096 寬） | `get_video_features()[0]` | `emb[:, :4096]`（需新增 env-gated dump） | 0C |
| 13 | duration / frame indices / `<t …>` | wrapper `indices` + decode prompt | proxy 建構參數 | A-lite |
| 14 | pre/post token 數、tok per grid-step | 由 11 導出 | `preflight.measure_tokens`（已有） | A-lite |
| 15 | raw output / parsed answer | `answers[i]` / `extract_characters_regex` | traj json / 同 parser | D |

### 5.2 定位階梯

```
channel index set 交集低
    → topk over near-tied variances；不是公式錯，但會讓後面所有量偏掉。先確認 tie 結構
u_t / r_t / token_score 差異 >> 0C 的 noise floor
    → scorer 真的寫錯：normalize 順序 / alpha / centroid 維度 / R vs 1−q
u_t、r_t、ranking 都一致，只有 k_t 與 mask 不同
    → 這是 0B 的刻意差異，不是 bug。量化它，寫進結論
grid_thw 不同（0C）
    → preprocessing：size.longest_edge vs max_pixels vs proxy 448px vs 兩段抽幀
grid 同但 embeds cosine 低
    → ViT/merger kernel 數值差 → 這就是 noise floor 本身
以上全過但 Stage D 的 prediction 不一致
    → serving / prompt / decoding，與 compressor 無關
```

---

## 6. 判準（v2 重寫）

### 6.1 Stage 0A — 唯一有硬判準的地方，但**閾值先量再訂**

只有兩項是無條件的：

- `T`、`tpf` 必須完全相同。
- ideal `r_t` 的差異必須 ≪ `R` 本身（否則 `R` vs `1−q` 的語意接反了）。

其餘（channel 交集、u_t max diff、ranking Jaccard）**先 dump 原始數字，
再以 Stage 0C 的 noise floor 為基準訂閾值**。v1 寫死的 `Spearman ≥ 0.9999`、
`Jaccard ≥ 0.98` 在 0C 出來前都是猜的，v2 移除。

### 6.2 Stage 0B — **沒有 pass/fail**

0B 是 characterization。輸出是一句可以寫進論文的描述句（§4 Stage 0B 末），
不是一個布林值。唯一會讓 0B 變成紅旗的情形是：
`sum(k_t)` 兩邊差 >5%，或 selected-token Jaccard < 0.5 —— 那代表 apportionment 把
VidCom² 的 allocation **形狀**也改掉了，而不只是取整。

### 6.3 為什麼 accuracy 幾乎無法證明等價（power 計算）

配對 Δ = (b−c)/n，b/c 是 discordant 計數，`d = (b+c)/n` 是 discordance rate。
`SE(Δ) ≈ sqrt(d/n)`；兩個獨立 stack 相減，`SE(D) = sqrt(2d/n)`；95% CI 半寬 = `1.96·sqrt(2d/n)`。

**D = Δ_official − Δ_ours 的 95% CI 半寬（pp）**

| n \ d | 0.10 | 0.15 | 0.20 |
| --- | --- | --- | --- |
| 300 | 5.06 | 6.20 | 7.16 |
| 1,000 | 2.77 | 3.39 | 3.92 |
| **1,549（LVBench 全量）** | **2.23** | **2.73** | **3.15** |
| 3,000 | 1.60 | 1.96 | 2.26 |

要讓 CI 半寬 ≤ 2 pp（即 CI 整段落在 [−2, +2] 才叫等價），需要
`n = 2d·(1.96/0.02)² ≈ 1,921 / 2,881 / 3,842`（對應 d = .10/.15/.20）。

> **LVBench 只有 1,549 題。也就是說，「Δ 差異 <2 pp」這個等價帶，
> 就算把整個 LVBench 跑滿、跑兩個 stack 各兩臂，也打不到。**

結論：
- **accuracy 只能偵測明顯的 bug（>5 pp 等級），不能 certify equivalence。**
- 因此 v1 的「B≈C 即驗證通過」在統計上是做不到的，這也是 v2 把 Stage B/C 從必做降為條件性的理由。
- 若真要跨 stack 比行為，用 **Stage D 的 prediction agreement**：量 `a` 的
  `SE = sqrt(a(1−a)/n)`，`a=0.90, n=300` → **1.7 pp**，比 D 的 5.06 pp **效率高約 3×**，
  而且是絕對量而非差之差。

### 6.4 Stage A-lite

沒有 pass/fail，只有「四個數字有沒有印出來」。
額外必記：官方的 tok-per-grid-step 是多少，以及它和我們 arm e/f 的關係。

### 6.5 分層報告（取代 v1 的「排除 localization_trivial」）

任何 accuracy 讀數都必須同時報四層，**primary result 用完整固定 subset**：

```
all questions
non-localization-trivial
localization-trivial
GT-hit  /  GT-miss     （用 time_reference 的 GT evidence span 分層）
```

刪題會變成 benchmark selection concern；我們有 temporal GT，應該用它 stratify。

---

## 7. 執行順序（v2，比 v1 短）

```
1. Stage 0A   scorer parity            ~20 min, 1 GPU   (vllm env，零 code 改動)
2. Stage 0B   allocation deviation     ~5 min,  0 GPU   (吃 0A 的中間量)
3. Stage 0C   HF vs vLLM noise floor   ~20 min, 1 GPU   (需 1 個 env-gated dump)
4. Stage A-lite  官方 20 題 dump        ~30 min, 1 GPU   (需新 conda env)
                    │
              DECISION POINT
                    │
5. arm f − arm e    32f hi-res: vanilla vs VidCom² 25%   ← 真正推進研究的一步
                    │
6. (條件性) Stage D  matched inference   只在 0C 或 A-lite 顯示異常時
7. (條件性) Stage E  32→512 scaling curve, 固定 total budget
```

**在 1–4 完成之前不啟動 6、7。**
第 5 步可與 1–4 並行（它用既有配置，不依賴驗證結果）。

---

## 8. 需要新增／改動的檔案

| 檔案 | 動作 | 屬於 |
| --- | --- | --- |
| **新檔** `longvt_compression/tools/vidcom2_parity.py` | Stage 0A + 0B 的 harness。載 HF 模型一次 → 取 `video_embeds` → 分別跑兩份 scorer → 輸出 §9 的 report | 0A/0B |
| `longvt_compression/vidcom2_vllm/retention.py` | **只加一個 env-gated dump**：`VIDCOM2_DUMP_EMBEDS=<path>` 時 `torch.save(video_embeds[:, :main_width])` 一次後即停。放在既有 `_dump` 區塊旁，關閉時零成本。**不改任何演算法** | 0C |
| **新檔** `longvt_compression/tools/vidcom2_noise_floor.py` | Stage 0C：比 grid_thw → 比 embeds → 同 scorer 跑兩組 embeds 比 mask Jaccard | 0C |
| scratchpad `vidcom2_qwen/lmms_eval/models/simple/qwen3_vl.py` | 加 `VIDCOM2_OFFICIAL_DUMP` 旗標，print §5.1 第 11、13、14 項。**不 commit 進我們的 repo** | A-lite |
| `RESEARCH.md` | 記錄驗證設定與結果；撤回「180 = official validated regime」的說法 | — |

**Stage 0A/0B 完全不需要改既有 code**（官方與我們的函式都可直接 import）。
**不需要改**：`serve_arm.sh`、`run_agent.py`、`data.py`、`longvt_scoring.py`、`preflight.py`。

Stage D 若真的要做，才需要在 `run_agent.py` 加一個 `--prompt-style lmms_eval` 的
**video-modality** 單次推論路徑（現有 `run_plain.py` 的 lmms_eval style 走 image modality，
而 vLLM 不 prune image，所以不能直接用）。**現在不做。**

---

## 9. Stage 0 的輸出格式（report spec）

每支影片、每個 operating point（64f / 256f）各一塊：

```
VIDEO <id>   frames=<n>   R=0.25

STRUCTURE
  T                          official / ours
  tpf                        official / ours

SCORER  (Stage 0A)
  channel set overlap        <|A∩B| / |A|>
  vid_score max abs diff
  frame_score max abs diff
  u_t max abs diff
  u_t Spearman
  u_t rank identical?        yes/no
  ideal r_t max abs diff
  token_score max abs diff
  token-ranking Jaccard      @ top-10% / 25% / 50% of tpf

INTEGER ALLOCATION  (Stage 0B)   -- intentional difference, characterize only
  official total k           <n>   ( <pct>% of T·tpf )
  ours total k               <n>   ( <pct>% of T·tpf )
  frames with same k_t       <n> / <T>
  max |Δk_t|
  Δk_t distribution          <hist>
  selected-token Jaccard     per-frame median / global

REPRESENTATION  (Stage 0C)   -- noise floor
  grid_thw equal?            yes/no      <-- 若 no，以下不做
  embedding cosine           median / min
  same-scorer mask Jaccard   <value>     <-- 0A 的閾值以此為基準
```

---

## 10. 這份計畫**不**主張什麼

- 不主張我們的 LVBench 數字應該接近任何已發表數字。
- 不主張 `arm f` 與官方在同一個 operating point——那要等 Stage A-lite 的實測。
- 不主張 integer allocation 與官方相同——它刻意不同，0B 會把差異寫成一句可引用的描述。
- 不主張 accuracy 能證明等價——§6.3 的計算顯示在 LVBench 的規模下做不到。
- 通過全部 Stage 0 之後，可以主張的只有一句：

> **Our scorer faithfully reproduces VidCom²'s frame-uniqueness, per-frame ideal retention,
> and token ranking on identical inputs; our serving integration differs only in the final
> integer budget allocation imposed by vLLM's pre-sized placeholder run.**

這句話足以停止懷疑「VidCom² 公式寫錯」。剩下的都是研究問題，不是驗證問題。
