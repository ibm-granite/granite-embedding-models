# What's New in Granite Embedding R2

This document summarizes the improvements from R1 to R2 across both English and Multilingual Granite Embedding models.

## Architecture Changes (R1 → R2)

| | R1 | R2 |
|---|---|---|
| **Architecture** | XLM-RoBERTa (Multilingual) / BERT (English) | ModernBERT |
| **Max context** | 512 tokens (Multilingual) / 512 tokens (English) | 32,768 tokens (Multilingual) / 8,192 tokens (English) |
| **Attention** | Full | Alternating (global every 3rd layer) |
| **Position encoding** | Absolute | Rotary (RoPE) |
| **Activations** | GELU | GeGLU (311M, 149M, 47M) / SiLU (97M) |
| **Vocab size** | 250K shared XLM-R (Multilingual) / 30K (English) | 262K / 180K (Multilingual), 50K (English) — purpose-trained |
| **Code languages** | None | Python, Go, Java, JS, PHP, Ruby, SQL, C, C++ |
| **Matryoshka support** | No | Yes (311M Multilingual: 768/512/384/256/128) |
| **Flash Attention** | No | Yes (2.0) |

## English: R1 vs R2 Comparison

Performance comparison of Granite English R1 and R2 models on key benchmarks:

| Model                              | Parameters (M) | Embedding Size | BEIR Retrieval (15) | MTEB-v2 (41)| CoIR (10) | MLDR (En) | MTRAG (4) |  Encoding Speed (docs/sec) |
|------------------------------------|:--------------:|:--------------:|:-------------------:|:-----------:|:---------:|:---------:|:---------:|:-------------------------------:|
| granite-embedding-125m-english     |      125       |      768       |        52.3         |     62.1   |   50.3    |   35.0    |   49.4   |               149             |
| **granite-embedding-english-r2**   |      149       |      768       |        **53.1**     |   **62.8** | **55.3**  | **40.7**  | **56.7** |               144              |
| granite-embedding-30m-english      |       30       |      384       |        49.1         |     60.2   |   47.0    |   32.6    |   48.6   |               198             |
| **granite-embedding-small-english-r2** |   47       |      384       |        **50.9**     |   **61.1** | **53.8**  | **39.8**  | **48.1** |               199             |

Key improvements (English):
- **BEIR Retrieval**: +0.8 (149M) and +1.8 (47M) over R1
- **CoIR (Code Retrieval)**: +5.0 (149M) and +6.8 (47M) — the largest gains, reflecting new code training data
- **MLDR (Long Document)**: +5.7 (149M) and +7.2 (47M) — benefiting from extended context
- **MTRAG (Conversational)**: +7.3 (149M) — significant improvement in multi-turn retrieval

## Multilingual: R1 vs R2 Comparison

| Model | Params | Embed Dim | MTEB Multilingual Retrieval (18) | Code (12) | English Retrieval (10) | LongEmbed (6) | RaR-b (17) |
|---|---|---|---|---|---|---|---|
| granite-embedding-107m-multilingual (R1) | 107M | 384 | 48.1 | 40.7 | 47.9 | 34.3 | 17.1 |
| **granite-embedding-97m-multilingual-r2** | **97M** | **384** | **59.6** | **60.5** | **50.1** | **65.6** | **24.9** |
| granite-embedding-278m-multilingual (R1) | 278M | 768 | 52.2 | 48.5 | 51.5 | 37.7 | 18.9 |
| **granite-embedding-311m-multilingual-r2** | **311M** | **768** | **64.0** | **63.9** | **52.6** | **71.7** | **28.0** |

Key improvements (Multilingual):
- **MTEB Multilingual Retrieval**: +11.5 (97M) and +11.8 (311M) over R1
- **Code Retrieval**: +19.8 (97M) and +15.4 (311M) — reflecting new code training
- **LongEmbed**: +31.3 (97M) and +34.0 (311M) — the biggest gain, direct payoff of the 32K context window (R1 was limited to 512 tokens)
- **RaR-b**: +7.8 (97M) and +9.1 (311M)

## Multilingual Speed: R1 vs R2

| Model | Latency (s/query) | Throughput (docs/s) | MTEB Multilingual Retrieval |
|---|---:|---:|---:|
| granite-embedding-107m-multilingual (R1) | 0.30 | 3,337 | 48.1 |
| **granite-embedding-97m-multilingual-r2** | 0.35 | 2,894 | **59.6** |
| granite-embedding-278m-multilingual (R1) | 0.46 | 2,185 | 52.2 |
| **granite-embedding-311m-multilingual-r2** | 0.52 | 1,944 | **64.0** |

The R2 models have slightly lower throughput than R1 (due to the larger architecture), but deliver substantially higher retrieval quality — a worthwhile tradeoff for most workloads.

## Cross-lingual Retrieval: R1 vs R2

Average performance on cross-lingual tasks within MTEB Retrieval:

| Model | Belebele Retrieval | MLQA Retrieval |
|---|---|---|
| granite-embedding-107m-multilingual (R1) | 55.1 | 60.5 |
| granite-embedding-97m-multilingual-r2 | 52.9 | 49.9 |
| granite-embedding-278m-multilingual (R1) | 62.2 | 63.0 |
| granite-embedding-311m-multilingual-r2 | **66.5** | 59.5 |

The 311M R2 model gains +4.3 on Belebele over its R1 predecessor, showing improved cross-lingual transfer at the larger scale. The 97M R2 model shows lower cross-lingual scores compared to R1 — a tradeoff from the pruning and vocabulary reduction process. The R2 97M model's training prioritized the broader 18-language MTEB Multilingual Retrieval set (where it gains +11.5 over R1) and long-document retrieval (+31.3). If cross-lingual transfer across many language pairs is your primary use case, the full-size 311M model is the better choice.