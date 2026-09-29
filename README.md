# SuzuQu-Engine-4exp

**SuzuQu-Engine-4exp**（略称`q4`）—
RDNA3 GPU（弊環境はRX 7900 XTX 24GB）＋大容量 DRAM (弊環境は64GB)で

Qwen3.8-Flash-Next(デフォルト推奨はV100*2のために量子化されたやつ)を

極限まで最適化して動かす推論エンジンです。(現在は**ベータ版**)

他者が使うことを想定した整備は道半ばなので、「インストールして動かす」というよりは「コードを見て活かす」という使い方がいいと思います。

基本設計はrouted expert を DRAM に全常駐させ、PLE（n-gram 26.82 GiB）だけ SSD に置く形態です。
設計の詳細と計測は [DESIGN.md](DESIGN.md)、gfx1100/ROCm 固有の知見と
最適化ログは [ROCM-NOTES.md](ROCM-NOTES.md)をご覧ください。

**重要:** .mdファイルの文書群は、人間のレビューがまだあまりされていません。**鵜呑みにしないでください。**
なお、文中に出てくる一部の参照先（`MODEL-STRATEGY.md` / `QUANT-ESTIMATE.md` / `RDNA3-OPT.md`）は未公開の作業メモです。

同様の理由で、導入経路とかもぐちゃぐちゃになってる気がします。適宜直しておくので、それまではコーディングエージェントとか使ってください...

## 起動

Windows？なんすかWindowsって。(Linux専用です)

```sh
make
HIP_VISIBLE_DEVICES=0 ./serve.sh
```

`serve.sh` の既定:

限界まで最適化した都合上、適宜書き換えないと動かないかもです！

- モデルは `./download_q4.sh` で `models/IQ3E-Q8D-MTP/` に取得（リポジトリ隣の `../models/` も自動検出。`Q4_MODEL` で任意パス可）
- expert 全常駐（`Q4_EXPERT_RESIDENT=1`）。GPU スロットの DRAM ミラーはなし。
- KV は q8_0(もう少し量子化を緩めてもいいかも)。ctx **262144**（学習長）。q8 KV + GDN は 3.30 GiB。
  QSA は 2051 トークン幅の疎なAttention機構なので、伸ばしても prefill はそこまで大きく落ちない。
  実際に最後まで埋めた計測は 90,112 トークン（旧カーネルで 59.5 tok/s。
  現在のカーネルでは未再測）。実用上の枠は ハーネス側の compactionで操作してね
- MTP はオフ（DRAMオフロードだからか遅くなった）
- ピン集合はルーティングプロファイル。`Q4_ROUTE_PROF=<path>` で採取・終了時に保存され、
  モデルの隣に `route-prof-iq3e.warm` があれば自動で `Q4_WARM_FILE` に採用。
  モデルごとにwarmが必要(SSDオフロードするときは超重要)
- 待受 `127.0.0.1:8090`（OpenAI `/v1/chat/completions`、ブラウザ UI は実装中）
- `systemd-run --user --scope -p MemoryMax=52G -p MemorySwapMax=0` で起動
  （`Q4_MEM_MAX`）。cap を超えたとき q4 だけが落ち、デスクトップは残る...かもしれない

```sh
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"1+1は?"}],"max_tokens":64}'
```

## 速度（2026-09-28、現行カーネル）

ctx ~16–21K、decode グラフ経路、expert 全常駐・ウォーム L1。

| | |
| --- | ---: |
| prefill 8,192 tok（probe） | **474.7 tok/s** |
| prefill 21,627 tok（serve 実測） | **561.8 tok/s** |
| decode（定常、live serve） | **43.6–45.2 tok/s** |
| decode（IQ3E, 部分resident 82-92%） | 23-33 tok/s |
| decode（IQ3E, `Q4_RES_BUDGET_GIB=46` フルresident） | **39.6 tok/s** |

軽い最適化のみ行なっていた旧エンジンからの改善幅: prefill ~308 → 474 tok/s（probe 8K）、
decode ~12–15 → ~44.4 tok/s。カーネル別内訳と手法は ROCM-NOTES.md。

GPU busy は ~25 ms/token で decode 壁時間とほぼ一致しており、
残りのカーネルはほぼ帯域律速（`gemv_mw` ~780 GB/s、wgv_q8 450–600 GB/s）。
さらなる高速化の主戦場は量子化自体。

## ノブ

| env | 既定 | 意味 |
| --- | --- | --- |
| `Q4_EXPERT_RESIDENT` | 1 | routed expert を tight pack で DRAM 常駐。0 で部分 L2 |
| `Q4_DRAM_RESERVE_GIB` | 4 | 常駐後に残す空き。 |
| `Q4_MEM_MAX` | 52G | systemd scope の MemoryMax。0 で cap なし |
| `Q4_L2_GIB` | 0 | 常駐が成功したときは無視。常駐できないときの DRAM L2 上限 |
| `Q4_MLOCK_L2` | 0 | 常駐イメージは mlock しない（rlimit 8 MB。成功してもマシンが固まる） |
| `Q4_KV` | q8 | `f16` / `f32` で広げる。セッションの幅が違うキャッシュは読めない |
| `Q4_CTX` | 262144 | 学習長。q8 KV + GDN 3.30 GiB |
| `Q4_VRAM_RESERVE_GIB` | 2 | 重みに使わない VRAM。1 未満は 1 になる。超えると画面が落ちるかも! |
| `Q4_L1_SLOTS` | 8192 | 要求の上限。VRAM 残量でこれより少なくなる |
| `Q4_IO_THREADS` | 8 | expert staging の I/O スレッド |
| `Q4_CPU_THREADS` | 12 | CPU エキスパートプール。16 でも同等だったので 12 で CCD を残した |
| `Q4_DEVROUTE` | 0 | デバイス側ルーティング。現状はホスト hybrid の方が ~11% 速い |
| `Q4_MISS_STAGE` | 1 | decode のミスを次トークン用に非同期ステージ。常駐時は 0 でもよい |
| `Q4_REPIN_EVERY` | 64 | decode N トークンごとに高ヒット expert を pin へ昇格 |
| `Q4_PIN_ARENA` | 0 | 部分 L2 の staging arena を hipHostRegister。常駐時は何もしない |
| `Q4_PIN_CAP` | (自動) | ピン数の上書き。未設定時は decode 1 ステップの 1.5 倍かつ L1 の 1/4 以下で、ヒット数が首位の 1/8 以上だけ |
| `Q4_WARM_FILE` | モデル横の `route-prof-iq3e.warm` | ピン集合。モデルのバイト数・層数・expert 数が違うと捨てる |
| `Q4_ROUTE_PROF` | （未設定） | ルーティングヒットのヒストグラムを終了時に warm 形式で保存。そのまま `Q4_WARM_FILE` に指せば、部分 resident が層別カバレッジ均等化 quota で hot-set 充填される |
| `Q4_SESS_CACHE` | `session.cache` | ターンをまたぐ prefix。同じプロンプトの再送は `session.cache.prompt` から再開する |
| `Q4_THINK` | 1 | 0 で空の think ブロック（reasoning なし） |
| `Q4_SHOW_THINKING` | 1 | reasoning を `reasoning_content` に出す |
| `Q4_MAX_NEW` | 32768 | クライアントが `max_tokens` を省略したときの上限。残コンテキストでも切る |
| `Q4_MTP` | 0 | 空でないパスでドラフトヘッド。今はオフ |
| `HIP_VISIBLE_DEVICES` | 0 | iGPU を隠す |

## モデル

GGUF は各配布元から取得してください。
IQ3Eの配布元(製作者はpentacoxian様): <https://huggingface.co/pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF>

## ライセンス

MIT（[LICENSE](LICENSE)）
