---
## Introduction to Granite Embedding Models

- **Huggingface Repository:** [ibm-granite/granite-embedding-models](https://huggingface.co/collections/ibm-granite/granite-embedding-models-6750b30c802c1926a35550bb)
- **Documentation**: [Granite Docs](https://www.ibm.com/granite/docs/models/embedding/)
- **Granite Community**: [ibm-granite-community](https://github.com/ibm-granite-community)
 
The Granite Embedding collection delivers innovative sentence-transformer models purpose-built for retrieval-based applications. Featuring a bi-encoder architecture, these models generate high-quality embeddings for textual inputs such as queries, passages, and documents, enabling seamless comparison through cosine similarity. Built using retrieval oriented pretraining, contrastive finetuning, knowledge distillation, and model merging, the Granite Embedding lineup is optimized to ensure strong alignment between query and passage embeddings. 

Built on a foundation of carefully curated, permissibly licensed public datasets, the Granite Embedding models set a high standard for performance, maintaining competitive scores not only on academic benchmarks such as BEIR, but also out-perfoming models of the same size on many enterprise use cases. Developed to meet enterprise-grade expectations, they are crafted transparently in accordance with IBM's AI Ethics principles and offered under the Apache 2.0 license for both research and commercial innovation. 

The R2 models show strong performance across standard and IBM-built information retrieval benchmarks (BEIR, ClapNQ),
code retrieval (COIR), long-document search benchmarks (MLDR, LongEmbed), conversational multi-turn (MTRAG),
table retrieval (NQTables, OTT-QA, AIT-QA, MultiHierTT, OpenWikiTables), visual document retrieval (ViDoRe v1–v3, Real-MM-RAG, JinaVDR, M3DR, VisRAG, VisDocOOD), and on many enterprise use cases. For details on improvements over R1, see [What's New in R2](whats_new_in_r2.md).

These models use a bi-encoder architecture to generate high-quality embeddings from text inputs such as queries, passages, and documents, enabling seamless comparison through cosine similarity. Built using retrieval oriented pretraining, contrastive finetuning, knowledge distillation, and model merging, the Granite Embedding R2 models are optimized to ensure strong alignment between query and passage embeddings.

The Granite Embedding R2 release introduces English, Multilingual and Vision models, all built on the ModernBERT architecture:

**English:**
- [granite-embedding-english-r2](https://huggingface.co/ibm-granite/granite-embedding-english-r2) (**149M** parameters): with an output embedding size of _768_, replacing _granite-embedding-125m-english_.
- [granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2) (**47M** parameters): A reduced-size model, with fewer layers and a smaller output embedding size (_384_), replacing _granite-embedding-30m-english_.

**Multilingual:**
- [granite-embedding-311m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2) (**311M** parameters): A flagship multilingual model with 768-dimensional embeddings, Matryoshka dimension support, and top-tier multilingual retrieval quality.
- [granite-embedding-97m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2) (**97M** parameters): A compact multilingual model with 384-dimensional embeddings supporting 200+ languages with a 32,768-token context window.

**Vision:**
- [granite-embedding-748m-vision-r2](https://huggingface.co/ibm-granite/granite-embedding-748m-vision-r2) (**748M** parameters): A multimodal model that retrieves document pages directly from their images, without OCR, layout analysis or chunking. Built on the granite-embedding-311m-multilingual-r2 text encoder and a SigLIP2 NaFlex vision tower, it produces a _768_-dimensional single vector (Matryoshka) and _128_-dimensional multi-vectors (one per token, scored with MaxSim) from one forward pass, and still handles text passages.

## Model Details

- **Developed by:** Granite Embedding Team, IBM
- **Repository:** [ibm-granite/granite-embedding-models](https://github.com/ibm-granite/granite-embedding-models)
- **Paper:** [Techincal Report](papers/GraniteEmbeddingR2.pdf)
- **Language(s) (NLP):** English (all models), 200+ languages with enhanced retrieval for 52 languages (multilingual models), queries in the same languages with page images aligned in 22 languages (vision model)
- **Modalities:** Text (English and multilingual models); text queries, page images and text documents (vision model)
- **Release Date**: Aug 15, 2025 (English) / April 29, 2026 (Multilingual) / November 30, 2026 (Vision)
- **Model Size:** 47M & 149M (English) / 97M & 311M (Multilingual) / 748M (Vision)
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Usage

**Intended Use:** The model is designed to produce fixed length vector representations for a given text, which can be used for text similarity, retrieval, and search applications.

For efficient decoding, these models use Flash Attention 2. Installing it is optional, but can lead to faster inference.

```shell
pip install flash_attn
```

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

**Usage with Sentence Transformers:** 

The model is compatible with SentenceTransformer library and is very easy to use:

First, install the sentence transformers library
```shell
pip install sentence_transformers
```

The model can then be used to encode pairs of text and find the similarity between their representations

```python
from sentence_transformers import SentenceTransformer, util

model_path = "ibm-granite/granite-embedding-english-r2"
# Load the Sentence Transformer model
model = SentenceTransformer(model_path)

input_queries = [
    ' Who made the song My achy breaky heart? ',
    'summit define'
    ]

input_passages = [
    "Achy Breaky Heart is a country song written by Don Von Tress. Originally titled Don't Tell My Heart and performed by The Marcy Brothers in 1991. ",
    "Definition of summit for English Language Learners. : 1 the highest point of a mountain : the top of a mountain. : 2 the highest level. : 3 a meeting or series of meetings between the leaders of two or more governments."
    ]

# encode queries and passages. The model produces unnormalized vectors. If your task requires normalized embeddings pass normalize_embeddings=True to encode as below.
query_embeddings = model.encode(input_queries)
passage_embeddings = model.encode(input_passages)

# calculate cosine similarity
print(util.cos_sim(query_embeddings, passage_embeddings))
```

**Usage with Huggingface Transformers:** 

This is a simple example of how to use the granite-embedding-english-r2 model with the Transformers library and PyTorch.

First, install the required libraries
```shell
pip install transformers torch
```

The model can then be used to encode pairs of text

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_path = "ibm-granite/granite-embedding-english-r2"

# Load the model and tokenizer
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

input_queries = [
    ' Who made the song My achy breaky heart? ',
    'summit define'
    ]

# tokenize inputs
tokenized_queries = tokenizer(input_queries, padding=True, truncation=True, return_tensors='pt')

# encode queries
with torch.no_grad():
    # Queries
    model_output = model(**tokenized_queries)
    # Perform pooling. granite-embedding-278m-multilingual uses CLS Pooling
    query_embeddings = model_output[0][:, 0]

# normalize the embeddings
query_embeddings = torch.nn.functional.normalize(query_embeddings, dim=1)

```

**Usage of the vision model:**

granite-embedding-748m-vision-r2 is a sentence-transformers [`MultiVectorEncoder`](https://www.sbert.net/docs/multi_vector_encoder/usage/usage.html) and needs sentence-transformers 6 or later. Load it with `MultiVectorEncoder`, not `SentenceTransformer`, which would drop the multi-vector head. For more examples (hybrid search, cross-language queries, query-token match visualization, mixed text/image indexes, index compression and encoding speed), see the [vision examples notebook](code/granite_embedding_748m_vision_r2_examples.ipynb).

```shell
pip install -U "sentence-transformers>=6" pillow
```

```python
import torch
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder(
    "ibm-granite/granite-embedding-748m-vision-r2",
    trust_remote_code=True,
    model_kwargs={"dtype": torch.bfloat16},
)

queries = [
    "How much did revenue grow in the third quarter?",  # English query
    "Quel est le plafond pour l'hôtel ?",                # French query
]
# Page images: file paths or PIL images
pages = ["quarterly_report_page_3.png", "travel_policy_page_1.png"]

# Multi-vector search (MaxSim): each query and page becomes a sequence of 128-dimensional vectors
query_embeddings = model.encode_query(queries)
page_embeddings = model.encode_document(pages)
scores = model.similarity(query_embeddings, page_embeddings)  # (num_queries, num_pages)

# Single-vector search (cosine): the L2-normalized 768-dimensional [CLS] embedding,
# returned by the same forward pass together with the multi-vectors
query_features = model.encode_query(queries, output_value=None)
page_features = model.encode_document(pages, output_value=None)
query_vectors = torch.stack([f["sentence_embedding"] for f in query_features]).float()
page_vectors = torch.stack([f["sentence_embedding"] for f in page_features]).float()
scores = query_vectors @ page_vectors.T
```

Text passages are encoded with `encode_document` as well; a single `encode_document` call must contain one modality, so encode page images and text passages in separate calls.

## Evaluation Results
Granite embedding r2 models show strong performance across diverse tasks. The r2 models demonstrate speed and efficiency while maintaining competitive performance. For R1 vs R2 comparisons, see [What's New in R2](whats_new_in_r2.md).

### English Evaluation Results

Performance of the Granite English R2 models compared to other models on MTEB-v2 Retrieval, code retrieval (CoIR), long-document search benchmarks (MLDR, LongEmbed), conversational multi-turn (MTRAG), and table retrieval (NQTables, OTT-QA, AIT-QA, MultiHierTT, OpenWikiTables). Encoding speed is measured on a single H100 GPU using 512 context length chunks.

|Model                              | Parameters (M) | Embedding Size |**AVERAGE**|MTEB-v2 Retrieval (10) | CoIR (10) | MLDR (En) | LongEmbed (6)| Table IR (5)| MTRAG(4) |  Throughput (spans/s) |
|-----------------------------------|:--------------:|:--------------:|:---------:|:---------------------:|:---------:|:---------:|:------------:|:-----------:|:--------:|-------------------------------:|
|e5-base-v2                         |109|768|47.5|49.7|50.3|32.5|41.1|74.09|37.0| 1,455 |
|bge-base-en-v1.5                   |109|768|46.9|54.8|46.6|33.5|33.9|73.98|38.8| 1,730 |
|snowflake-arctic-embed-m-v2.0      |305|768|51.4|58.4|52.2|32.4|55.4|80.75|29.2| 2,833 |
|gte-base-en-v1.5                   |137|768|52.8|55.5|42.4|42.7|59.4|80.52|36.0| 1,488 |
|gte-modernbert-base                |149|768|57.5|57.0|71.5|46.2|57.0|76.68|36.8| 2,951 |
|nomic-ai/modernbert-embed-base     |149|768|48.0|48.7|48.8|31.3|56.3|66.69|36.2| 2,929 |
|||||||||||
|**granite-embedding-english-r2**       |149|768|**59.5**|56.4|54.8|41.6|67.8|78.53|57.6| 2,935 |
|**granite-embedding-small-english-r2** | 47|384|55.6|53.9|53.4|40.1|61.9|75.51|48.9| 3,460 |

### Multilingual Evaluation Results

Performance across the main multilingual benchmark suite. Scores are averages across tasks within each benchmark (higher is better). Encoding speed is measured on a single NVIDIA H100 GPU using 512-token chunks.

| Model | Params | Embed Dim | MTEB Multilingual Retrieval (18) | Code (12) | English Retrieval (10) | LongEmbed (6) | RaR-b (17) | Throughput (spans/s) |
|---|---|---|---|---|---|---|---|---:|
| multilingual-e5-small | 96M | 384 | 50.9 | 51.3 | 46.5 | 38.8 | 20.3 | 2,290 |
| **granite-embedding-97m-multilingual-r2** | **97M** | **384** | **59.6** | **60.5** | **50.1** | **65.6** | **24.9** | **3,379** |
| F2LLM-v2-80M | 80M | 320 | 50.1 | 68.0 | 47.5 | - | - | 2,498 |
| jina-embeddings-v5-text-nano | 239M | 768 | 63.3 | 71.2 | 58.8 | 63.6 | 25.2 | 1,081 |
| harrier-oss-v1-270m | 270M | 640 | 66.4 | 62.4 | 52.1 | 65.0 | 32.9 | 2,055 |
| multilingual-e5-base | 278M | 768 | 52.7 | 52.6 | 49.0 | 40.5 | 23.4 | 1,800 |
| embeddinggemma-300m | 300M | 768 | 62.5 | 69.0 | 54.6 | 55.4 | 26.1 | 1,277 |
| gte-multilingual-base | 305M | 768 | 57.2 | 57.5 | 50.8 | 62.1 | 19.0 | 1,609 |
| snowflake-arctic-embed-m-v2.0 | 305M | 768 | 54.8 | 55.2 | 58.4 | 55.4 | 23.3 | 2,754 |
| **granite-embedding-311m-multilingual-r2** | **311M** | **768** | **64.0** | **63.9** | **52.6** | **71.7** | **28.0** | **3,075** |

### Matryoshka Embeddings (311M Multilingual)

The 311M multilingual model supports [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147), allowing embeddings to be truncated from the full 768 dimensions down to 512, 384, 256, or 128 with graceful quality degradation. This is useful when storage, memory, or similarity-computation cost is a concern.

| Model | Embed Dim | English Retrieval (10) | Code (12) | MTEB Multilingual Retrieval (18) |
|---|---:|---:|---:|---:|
| 311M (Matryoshka) | 768 | 52.6 | 63.9 | 63.9 |
| 311M (Matryoshka) | 512 | 52.5 | 63.8 | 63.9 |
| 311M (Matryoshka) | 384 | 52.1 | 63.7 | 63.8 |
| 311M (Matryoshka) | 256 | 51.6 | 63.4 | 63.5 |
| 311M (Matryoshka) | 128 | 50.4 | 62.3 | 62.5 |
| *97M (native)* | *384* | *48.9* | *58.3* | *58.0* |

Cutting from 768 to 256 dimensions (a 3x reduction in storage and computation cost) drops MTEB Multilingual Retrieval by just 0.4 points (63.9 to 63.5). Even at 128 dimensions (a 6x reduction), the model retains over 97% of its full-dimension performance.

### Vision Evaluation Results

Performance of granite-embedding-748m-vision-r2 on eight visual document retrieval benchmarks: nDCG@10 on ViDoRe v3 and nDCG@5 on the others, with the number of tasks in parentheses. Both heads come from the same forward pass, so they encode at the same speed. Encoding speed is measured in page images per second on a single NVIDIA H200 in bf16, over the 14 tasks of ViDoRe v1 and v2.

| Model | Params | Dims / token | ViDoRe v1 (10) | ViDoRe v2 (4) | ViDoRe v3 (8) | Real-MM-RAG (4) | JinaVDR (43) | M3DR (23) | VisRAG (6) | VisDocOOD (4) | **Avg.** | Throughput (pages/s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---:|
| **granite-embedding-748m-vision-r2, multi-vector** | **748M** | **128** | **86.4** | **62.9** | **56.1** | **85.9** | **74.7** | **54.8** | **85.7** | **70.4** | **72.1** | **35.3** |
| granite-embedding-748m-vision-r2, single vector | 748M | 768 (single) | 74.5 | 54.2 | 44.3 | 64.8 | 63.5 | 42.2 | 73.8 | 63.8 | 60.1 | 35.3 |
| granite-embedding-748m-vision-r2, single vector, 384-d | 748M | 384 (single) | 74.2 | 53.7 | 43.8 | 64.3 | 63.1 | 41.9 | 72.9 | 63.6 | 59.7 | 35.3 |

The multi-vector head keeps 94% of the quality of its 4B teacher (76.9), and its index takes about 149 KiB per page (128 dimensions per token, about 600 vectors per page). Truncated to its first 384 dimensions, the single vector loses 0.4 points for half the storage. At 35.3 pages/s, one GPU encodes a million pages in about eight hours. Comparisons with other visual document retrievers in accuracy, encoding speed and index size will be in the Granite Embedding Multimodal R2 report (coming soon).

The vision model also keeps text retrieval, so text passages and page images can share one index (nDCG@10 averaged over the retrieval tasks of MTEB(eng, v2) and MTEB(Multilingual, v2)):

| Model | MTEB Retrieval, English (10) | MTEB Retrieval, Multilingual (18) |
|---|---:|---:|
| **granite-embedding-748m-vision-r2, multi-vector** | **50.9** | **65.6** |
| granite-embedding-748m-vision-r2, single vector | 47.8 | 62.9 |
| granite-embedding-748m-vision-r2, single vector, 384-d | 47.7 | 62.8 |
| granite-embedding-311m-multilingual-r2 (text only) | 52.6 | 65.2 |

For text-only collections, granite-embedding-311m-multilingual-r2 gives about the same accuracy with well under half the parameters and one vector per passage.


### Model Architecture and Key Features

The Granite Embedding R2 release includes five models based on the ModernBERT architecture:

**English:**
- _granite-embedding-english-r2_ (**149M** parameters): with an output embedding size of _768_.
- _granite-embedding-small-english-r2_ (**47M** parameters): A reduced-size model, with fewer layers and a smaller output embedding size (_384_).

**Multilingual:**
- _granite-embedding-311m-multilingual-r2_ (**311M** parameters): A flagship multilingual model with 768-dimensional embeddings and Matryoshka dimension support (768/512/384/256/128), supporting 200+ languages with enhanced retrieval for 52 languages and programming code.
- _granite-embedding-97m-multilingual-r2_ (**97M** parameters): A compact multilingual model with 384-dimensional embeddings — the highest retrieval score for any open multilingual embedding model under 100M parameters.

**Vision:**
- _granite-embedding-748m-vision-r2_ (**748M** parameters): A multimodal model for visual document retrieval with a 768-dimensional single vector (Matryoshka) and 128-dimensional multi-vectors from the same forward pass, built on the granite-embedding-311m-multilingual-r2 text encoder.

The following table shows the structure of the R2 text models:

| Feature | small-english-r2 | english-r2 | 97m-multilingual-r2 | **311m-multilingual-r2** |
| :--------- | :-------: | :-------: | :-------: | :--------: |
| Embedding size | 384 | 768 | 384 | **768** |
| Number of layers | 12 | 22 | 12 | **22** |
| Number of attention heads | 12 | 12 | 12 | **12** |
| Intermediate size | 1536 | 1152 | 1536 | **1152** |
| Activation Function | GeGLU | GeGLU | SiLU | **GeGLU** |
| Vocabulary Size | 50,368 | 50,368 | 180,000 | **262,000** |
| Max. Sequence Length | 8,192 | 8,192 | 32,768 | **32,768** |
| Matryoshka Support | No | No | No | **Yes (768/512/384/256/128)** |
| Code Languages | — | — | Python, Go, Java, JS, PHP, Ruby, SQL, C, C++ | **Python, Go, Java, JS, PHP, Ruby, SQL, C, C++** |
| Languages | English | English | 200+ (52 enhanced) | **200+ (52 enhanced)** |
| # Parameters | 47M | 149M | 97M | **311M** |

In granite-embedding-748m-vision-r2, a SigLIP2 NaFlex vision tower splits each page image into 16×16 patches at close to its native aspect ratio, and a one-layer window Q-Former connector compresses each 4×4 window of patches (4x fewer visual tokens) and maps them into the input space of the granite-embedding-311m-multilingual-r2 text encoder, adding 2D position and row-newline embeddings that help recover the page layout. The `[CLS]` token gives the single vector, and a linear projection of every token gives the multi-vectors.

| Feature | **748m-vision-r2** |
| :--------- | :-------: |
| Vision encoder | SigLIP2 so400m NaFlex, patch 16, 27 layers (417.1M) |
| Image input | Native aspect ratio, 576 to 2,880 patches per page (about 0.15 to 0.74 megapixels) |
| Connector | 1-layer window Q-Former over 4×4 patch windows (19.4M) |
| Text encoder | granite-embedding-311m-multilingual-r2: 22 layers, hidden size 768 (110.3M) |
| Vocabulary Size | 262,152 |
| Single-vector embedding | 768, `[CLS]` pooling; Matryoshka (768/512/384/256/128) |
| Multi-vector embedding | 128 per token, MaxSim; about 600 vectors per page, poolable to 32–300 vectors |
| Max. Sequence Length | 2,048 |
| # Parameters | 748M |


### Training and Optimization

The granite embedding r2 models incorporate key enhancements from the ModernBERT architecture, including: 
- Alternating attention lengths to accelerate processing 
- Rotary position embeddings for extended sequence length 
- A newly trained tokenizer optimized with code and text data 
- Flash Attention 2.0 for improved efficiency 
- Streamlined parameters, eliminating unnecessary bias terms

granite-embedding-748m-vision-r2 is trained in two stages: modality alignment of the vision tower and connector with the text encoder, then knowledge distillation from a 4B multi-vector multimodal teacher and an 8B text teacher, training both heads on the same targets with pooling-aware multi-vectors and a Matryoshka single vector. Its text-encoder weights are finally merged with those of granite-embedding-311m-multilingual-r2 to recover text retrieval quality.


## Data Collection

All embedding models are trained on a mixture of IBM‑curated datasets, publicly available data, and internally generated or synthetic data. Public web‑derived data used in training is selected and filtered using IBM‑developed quality, deduplication, and governance processes, that are designed to contribute to responsible use and enterprise deployment. We intentionally avoid the use of the MS‑MARCO training dataset and datasets with explicit non‑commercial licensing restrictions. The models are first pretrained using [GneissWeb](https://huggingface.co/datasets/ibm-granite/GneissWeb), an IBM‑curated dataset derived from publicly available web content and processed using IBM's data preparation and governance tooling—along with additional IBM‑curated and other publicly available sources. Datasets undergo IBM governance review to assess licensing considerations, ownership signals, and personal data risks. These processes are designed to contribute to responsible use and enterprise deployment. 

granite-embedding-748m-vision-r2 is additionally trained on document and image question-answering data, query–page retrieval pairs, chart and table question-answering data, IBM-generated synthetic queries for document images, and multilingual page retrieval data, all of which went through IBM's data clearance process.

## Infrastructure

We trained the granite embedding r2 models using IBM's computing cluster, BlueVela Cluster, which is outfitted with NVIDIA H100 80GB GPUs (and NVIDIA H200 GPUs for the vision model). This cluster provides a scalable and efficient infrastructure for training our models over multiple GPUs.

## Ethical Considerations and Limitations

The training data for all R2 models was filtered to remove text containing hate, abuse, and profanity.

**English models** (granite-embedding-english-r2, granite-embedding-small-english-r2) are trained only for English texts and have a context length of 8,192 tokens (longer texts will be truncated).

**Multilingual models** (granite-embedding-311m-multilingual-r2, granite-embedding-97m-multilingual-r2) support 200+ languages with enhanced retrieval quality for 52 languages, and have a context length of 32,768 tokens. While the models cover a broad set of languages, retrieval quality may vary across languages depending on the amount and quality of training data available for each language. Performance on low-resource languages may be lower than on the 52 enhanced languages.

**Vision model** (granite-embedding-748m-vision-r2) retrieves pages from their visual appearance, so retrieval quality depends on legibility: low-resolution scans, very dense pages or very small print may be retrieved less reliably, since each page is read as at most 2,880 patches of 16×16 pixels. Languages represented in the visual training data generally retrieve better than languages the model handles only through the multilingual text encoder. The multi-vector index is much larger than a single-vector index (about 149 KiB per page against 1.5 KiB), and for collections that contain only text, the multilingual R2 text models are more accurate.
