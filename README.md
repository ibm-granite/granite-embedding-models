---
## Introduction to Granite Embedding Models

- **Huggingface Repository:** [ibm-granite/granite-embedding-models](https://huggingface.co/collections/ibm-granite/granite-embedding-models-6750b30c802c1926a35550bb)
- **Documentation**: [Granite Docs](https://www.ibm.com/granite/docs/models/embedding/)
- **Granite Community**: [ibm-granite-community](https://github.com/ibm-granite-community)
 
The Granite Embedding collection delivers innovative sentence-transformer models purpose-built for retrieval-based applications. Featuring a bi-encoder architecture, these models generate high-quality embeddings for textual inputs such as queries, passages, and documents, enabling seamless comparison through cosine similarity. Built using retrieval oriented pretraining, contrastive finetuning, knowledge distillation, and model merging, the Granite Embedding lineup is optimized to ensure strong alignment between query and passage embeddings. 

Built on a foundation of carefully curated, permissibly licensed public datasets, the Granite Embedding models set a high standard for performance, maintaining competitive scores not only on academic benchmarks such as BEIR, but also out-perfoming models of the same size on many enterprise use cases. Developed to meet enterprise-grade expectations, they are crafted transparently in accordance with IBM's AI Ethics principles and offered under the Apache 2.0 license for both research and commercial innovation. 

The R2 models show strong performance across standard and IBM-built information retrieval benchmarks (BEIR, ClapNQ),
code retrieval (COIR), long-document search benchmarks (MLDR, LongEmbed), conversational multi-turn (MTRAG),
table retrieval (NQTables, OTT-QA, AIT-QA, MultiHierTT, OpenWikiTables), and on many enterprise use cases. For details on improvements over R1, see [What's New in R2](whats_new_in_r2.md).

These models use a bi-encoder architecture to generate high-quality embeddings from text inputs such as queries, passages, and documents, enabling seamless comparison through cosine similarity. Built using retrieval oriented pretraining, contrastive finetuning, knowledge distillation, and model merging, the Granite Embedding R2 models are optimized to ensure strong alignment between query and passage embeddings.

The Granite Embedding R2 release introduces English and Multilingual models, all based on the ModernBERT architecture:

**English:**
- [granite-embedding-english-r2](https://huggingface.co/ibm-granite/granite-embedding-english-r2) (**149M** parameters): with an output embedding size of _768_, replacing _granite-embedding-125m-english_.
- [granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2) (**47M** parameters): A reduced-size model, with fewer layers and a smaller output embedding size (_384_), replacing _granite-embedding-30m-english_.

**Multilingual:**
- [granite-embedding-311m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2) (**311M** parameters): A flagship multilingual model with 768-dimensional embeddings, Matryoshka dimension support, and top-tier multilingual retrieval quality.
- [granite-embedding-97m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2) (**97M** parameters): A compact multilingual model with 384-dimensional embeddings supporting 200+ languages with a 32,768-token context window.

## Model Details

- **Developed by:** Granite Embedding Team, IBM
- **Repository:** [ibm-granite/granite-embedding-models](https://github.com/ibm-granite/granite-embedding-models)
- **Paper:** [Techincal Report](papers/GraniteEmbeddingR2.pdf)
- **Language(s) (NLP):** English (all models), 200+ languages with enhanced retrieval for 52 languages (multilingual models)
- **Release Date**: Aug 15, 2025 (English) / April 29, 2026 (Multilingual)
- **Model Size:** 47M & 149M (English) / 97M & 311M (Multilingual)
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Usage

**Intended Use:** The model is designed to produce fixed length vector representations for a given text, which can be used for text similarity, retrieval, and search applications.

For efficient decoding, these models use Flash Attention 2. Installing it is optional, but can lead to faster inference.

```shell
pip install flash_attn
```use t

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


### Model Architecture and Key Features

The Granite Embedding R2 release includes four models based on the ModernBERT architecture:

**English:**
- _granite-embedding-english-r2_ (**149M** parameters): with an output embedding size of _768_.
- _granite-embedding-small-english-r2_ (**47M** parameters): A reduced-size model, with fewer layers and a smaller output embedding size (_384_).

**Multilingual:**
- _granite-embedding-311m-multilingual-r2_ (**311M** parameters): A flagship multilingual model with 768-dimensional embeddings and Matryoshka dimension support (768/512/384/256/128), supporting 200+ languages with enhanced retrieval for 52 languages and programming code.
- _granite-embedding-97m-multilingual-r2_ (**97M** parameters): A compact multilingual model with 384-dimensional embeddings — the highest retrieval score for any open multilingual embedding model under 100M parameters.

The following table shows the structure of all R2 models:

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


### Training and Optimization

The granite embedding r2 models incorporate key enhancements from the ModernBERT architecture, including: 
- Alternating attention lengths to accelerate processing 
- Rotary position embeddings for extended sequence length 
- A newly trained tokenizer optimized with code and text data 
- Flash Attention 2.0 for improved efficiency 
- Streamlined parameters, eliminating unnecessary bias terms


## Data Collection
Granite embedding r2 models are trained using data from four key sources: 
1. Unsupervised title-body paired data scraped from the web
2. Publicly available paired with permissive, enterprise-friendly license
3. IBM-internal paired data targetting specific technical domains
4. IBM-generated synthetic data

Notably, we _do not use_ the popular MS-MARCO retrieval dataset in our training corpus due to its non-commercial license (many open-source models use this dataset due to its high quality). 

The underlying encoder models using GneissWeb, an IBM-curated dataset composed exclusively of open, commercial-friendly sources.

For governance, all our data undergoes a data clearance process subject to technical, business, and governance review. This comprehensive process captures critical information about the data, including but not limited to their content description ownership, intended use, data classification, licensing information, usage restrictions, how the data will be acquired, as well as an assessment of sensitive information (i.e, personal information). 

## Infrastructure
We trained the granite embedding english r2 models using IBM's computing cluster, BlueVela Cluster, which is outfitted with NVIDIA H100 80GB GPUs. This cluster provides a scalable and efficient infrastructure for training our models over multiple GPUs.

## Ethical Considerations and Limitations
Granite-embedding-english-r2 leverages both permissively licensed open-source and select proprietary data for enhanced performance. The training data for the base language model was filtered to remove text containing hate, abuse, and profanity. Granite-embedding-english-r2 is trained only for English texts, and has a context length of 8192 tokens (longer texts will be truncated to this size).
