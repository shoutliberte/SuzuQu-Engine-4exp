# SuzuQu-Engine-4exp

**SuzuQu-Engine-4exp**（略称`q4`）—
RDNA3 GPU（RX 7900 XTX 24 GiB / gfx1100）＋大容量 DRAM で

V100×2(VRAM容量合計64GB)のために量子化された
**Qwen3.8-Flash-Next IQ3E** を

極限まで最適化して動かす推論エンジン。現在は実験版。

基本設計はrouted expert を DRAM に全常駐させ、PLE（n-gram 26.82 GiB）だけ SSD に置く形態。
設計の詳細と計測は [DESIGN.md](DESIGN.md)、gfx1100/ROCm 固有の知見と
最適化ログは [ROCM-NOTES.md](ROCM-NOTES.md)。

**重要:** .mdファイルの文書群は、人間のレビューがまだあまりされていません。**鵜呑みにしないでください。**

GPU と DRAM はデスクトップと共有する前提で、明示的な上限付きで運用する。
q4 は同時に 1 プロセス。DRAM の空きは 4 GiB より削らない
（`Q4_DRAM_RESERVE_GIB`）。VRAM のプランナは 2 GiB を重みに使わない
（`Q4_VRAM_RESERVE_GIB`、1 GiB 未満にはしない）。262144 枠の実測ピークは
22.4 GiB / 24.0 GiB。ここを超えるとデスクトップのセッションごと落ちる可能性がある。

## 起動

```sh
make
HIP_VISIBLE_DEVICES=0 ./serve.sh
```

`serve.sh` の既定:

- モデル `~/Projects/models/IQ3E-Q8D-MTP/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf`
- expert 全常駐（`Q4_EXPERT_RESIDENT=1`）。GPU スロットの DRAM ミラーはなし。
- KV は q8_0(もう少し量子化を緩めてもいいかも)。ctx **262144**（学習長）。q8 KV + GDN は 3.30 GiB。
  QSA は 2051 トークン幅の疎なAttention機構なので、伸ばしても prefill はそこまで大きく落ちない。
  実際に最後まで埋めた計測は 90,112 トークン（旧カーネルで 59.5 tok/s。
  現在のカーネルでは未再測）。実用上の枠は ハーネス側の compactionで操作してね
- MTP はオフ（実装中。この構成の KV は q8。MTP を足すなら `Q4_KV=f32` が必要で、
  同梱の Unsloth サイドカーは別量子化）
- CPU エキスパートプール 12 スレッド（`Q4_CPU_THREADS`、CCD1 を残す）
- 日本語 warmup のピン集合は `ja-experts-iq3e.warm`。2 回目以降は復元。
  UD-Q4 用の `ja-experts.warm` はバイト数が違うので読まない
- 待受 `127.0.0.1:8090`（OpenAI `/v1/chat/completions`、ブラウザ UI は実装中）
- `systemd-run --user --scope -p MemoryMax=52G -p MemorySwapMax=0` で起動
  （`Q4_MEM_MAX`）。cap を超えたとき q4 だけが落ち、デスクトップは残る

```sh
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"1+1は?"}],"max_tokens":64}'
```

## 速度（2026-09-28、現行カーネル）

ctx ~16–21K、decode グラフ経路、expert 全常駐・ウォーム L1。

| | |
|---|---:|
| prefill 8,192 tok（probe） | **474.7 tok/s** |
| prefill 21,627 tok（serve 実測） | **561.8 tok/s** |
| decode（定常、live serve） | **43.6–45.2 tok/s** |
| decode（IQ3E, 部分resident 82-92%） | 23-33 tok/s |
| decode（IQ3E, `Q4_RES_BUDGET_GIB=46` フルresident） | **39.6 tok/s** |

軽い最適化のみ行なっていた旧エンジンからの改善幅: prefill ~308 → 474 tok/s（probe 8K）、
decode ~12–15 → ~44.4 tok/s。カーネル別内訳と手法は ROCM-NOTES.md。

GPU busy は ~25 ms/token で decode 壁時間とほぼ一致しており、
残りのカーネルはほぼ帯域律速（`gemv_mw` ~780 GB/s、wgv_q8 450–600 GB/s）。
さらなる高速化の主戦場は量子化自体（[QUANT-ESTIMATE.md](QUANT-ESTIMATE.md)
に新規量子化の費用・効果試算あり）。

### 以前の長コンテキスト計測（2026-09-26、旧カーネル）

同じ段落の繰り返し。expert の SSD 読みは 0。ctx 90112、KV 1.21 GiB、
GPU L1 4982 スロット。

| | |
|---|---:|
| prefill 最初の 512 | 76 tok/s |
| prefill 16384 | 62.5 tok/s（262 s） |
| prefill 90112 全体 | 59.5 tok/s（1514 s、25 分） |
| 90112 地点のチャンク | 58.4 tok/s |
| 短いターンの decode | 15 tok/s |
| 16384 のあとの decode | 約 12 tok/s |

90K を全部埋めたときの MemAvailable 最低は **9.6 GiB**。16K を 90K 枠の
中で埋めたときは 10.9 GiB。デスクトップは落ちず、終了後に RAM は
戻っている。常駐の DRAM 床は 4 GiB。

## ノブ

| env | 既定 | 意味 |
|---|---|---|
| `Q4_EXPERT_RESIDENT` | 1 | routed expert を tight pack で DRAM 常駐。0 で部分 L2 |
| `Q4_DRAM_RESERVE_GIB` | 4 | 常駐後に残す空き。4 未満にはしない |
| `Q4_MEM_MAX` | 52G | systemd scope の MemoryMax。0 で cap なし |
| `Q4_L2_GIB` | 0 | 常駐が成功したときは無視。常駐できないときの DRAM L2 上限 |
| `Q4_MLOCK_L2` | 0 | 常駐イメージは mlock しない（rlimit 8 MB。成功してもマシンが固まる） |
| `Q4_KV` | q8 | `f16` / `f32` で広げる。セッションの幅が違うキャッシュは読めない |
| `Q4_CTX` | 262144 | 学習長。q8 KV + GDN 3.30 GiB |
| `Q4_VRAM_RESERVE_GIB` | 2 | 重みに使わない VRAM。1 未満は 1 になる。超えると画面が落ちる |
| `Q4_L1_SLOTS` | 8192 | 要求の上限。VRAM 残量でこれより少なくなる |
| `Q4_IO_THREADS` | 8 | expert staging の I/O スレッド |
| `Q4_CPU_THREADS` | 12 | CPU エキスパートプール。16 でも同等だったので 12 で CCD を残す |
| `Q4_DEVROUTE` | 0 | デバイス側ルーティング。現状はホスト hybrid の方が ~11% 速い |
| `Q4_MISS_STAGE` | 1 | decode のミスを次トークン用に非同期ステージ。常駐時は 0 でもよい |
| `Q4_REPIN_EVERY` | 64 | decode N トークンごとに高ヒット expert を pin へ昇格 |
| `Q4_PIN_ARENA` | 0 | 部分 L2 の staging arena を hipHostRegister。常駐時は何もしない |
| `Q4_PIN_CAP` | (自動) | ピン数の上書き。未設定時は decode 1 ステップの 1.5 倍かつ L1 の 1/4 以下で、ヒット数が首位の 1/8 以上だけ |
| `Q4_WARM_FILE` | `ja-experts-iq3e.warm` | ピン集合。モデルのバイト数・層数・expert 数が違うと捨てる |
| `Q4_ROUTE_PROF` | （未設定） | ルーティングヒットのヒストグラムを終了時に warm 形式で保存。そのまま `Q4_WARM_FILE` に指せば、部分 resident が層別カバレッジ均等化 quota で hot-set 充填される |
| `Q4_SESS_CACHE` | `session.cache` | ターンをまたぐ prefix。同じプロンプトの再送は `session.cache.prompt` から再開する |
| `Q4_THINK` | 1 | 0 で空の think ブロック（reasoning なし） |
| `Q4_SHOW_THINKING` | 1 | reasoning を `reasoning_content` に出す |
| `Q4_MAX_NEW` | 32768 | クライアントが `max_tokens` を省略したときの上限。残コンテキストでも切る |
| `Q4_MTP` | 0 | 空でないパスでドラフトヘッド。この IQ3E 構成ではオフ |
| `HIP_VISIBLE_DEVICES` | 0 | iGPU を隠す |

ビスク用フォールバック（各 `=0` で旧カーネルに戻る）: `Q4_WMMA`,
`Q4_QSA_PF`, `Q4_GDN_SCAN`, `Q4_PLE_BATCH`, `Q4_DEC_SPLIT`, `Q4_WARPGV`,
`Q4_PF`。計測は `Q4_PROFILE` / `Q4_STEP_PROF` / `Q4_GEMV_PROF`。
`Q4_PROFILE=1` は区間ごとに同期するので、長い prefill には使わない。

## モデル

| | 場所 |
|---|---|
| IQ3E（実質全常駐） | `~/Projects/models/IQ3E-Q8D-MTP/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP.gguf` |
| UD-Q4_K_XL（SSD オフロード前提。常駐できない） | `models/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/` |

IQ3E の routed は gate/up がほぼ IQ2_S、down がほぼ IQ4_NL。blk.2 の gate/up
だけ IQ3_S。dense と shared はほぼ Q8_0。PLE は IQ4_NL のまま SSD。
GGUF は各配布元から取得し `models/` 以下に置く（リポジトリ内には存在しない）。
IQ3Eの配布元(製作者はpentacoxian様): https://huggingface.co/pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF

## 以前の UD-Q4 パス

2026-08-28 時点の UD-Q4_K_XL はページキャッシュ L2 + SSD で、prefill 46–47
tok/s、温まった decode 7.2–8.1 tok/s だった。expert が 71.7 GiB あり、
この 60 GiB のマシンでは常駐できない。MTP サイドカー
（`models/Qwen3.8-Flash-Next-GGUF/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`）
はその構成用。IQ3E の `serve.sh` は読まない。

## ライセンス

MIT（[LICENSE](LICENSE)）。GGUF モデル本体は各配布元のライセンスに従う。
