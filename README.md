---
## Introduction to Granite Embedding Models

- **Huggingface Repository:** [ibm-granite/granite-embedding-models](https://huggingface.co/collections/ibm-granite/granite-embedding-models-6750b30c802c1926a35550bb)
- **Documentation**: [Granite Docs](https://www.ibm.com/granite/docs/models/embedding/)
- **Granite Community**: [ibm-granite-community](https://github.com/ibm-granite-community)
 
The Granite Embedding collection delivers innovative sentence-transformer models purpose-built for retrieval-based applications. Featuring a bi-encoder architecture, these models generate high-quality embeddings for textual inputs such as queries, passages, and documents, enabling seamless comparison through cosine similarity. Built using retrieval oriented pretraining, contrastive finetuning, knowledge distillation, and model merging, the Granite Embedding lineup is optimized to ensure strong alignment between query and passage embeddings. 

Built on a foundation of carefully curated, permissibly licensed public datasets, the Granite Embedding models set a high standard for performance, maintaining competitive scores not only on academic benchmarks such as BEIR, but also out-perfoming models of the same size on many enterprise use cases. Developed to meet enterprise-grade expectations, they are crafted transparently in accordance with IBM's AI Ethics principles and offered under the Apache 2.0 license for both research and commercial innovation. 

Developed to replace the very popular [r1 english models](README_r1.md), r2 models show strong performance across standard and IBM-built information retrieval benchmarks (BEIR, ClapNQ), 
code retrieval (COIR), long-document search benchmarks (MLDR, LongEmbed), conversational multi-turn (MTRAG), 
table retrieval (NQTables, OTT-QA, AIT-QA, MultiHierTT, OpenWikiTables), and on many enterprise use cases.

These models use a bi-encoder architecture to generate high-quality embeddings from text inputs such as queries, passages, and documents, enabling seamless comparison through cosine similarity. Built using retrieval oriented pretraining, contrastive finetuning, knowledge distillation, and model merging, granite-embedding-english-r2 is optimized to ensure strong alignment between query and passage embeddings.

The latest granite embedding r2 release introduces two English embedding models, both based on the ModernBERT architecture:
- [granite-embedding-english-r2](https://huggingface.co/ibm-granite/granite-embedding-english-r2) (**149M** parameters): with an output embedding size of _768_, replacing _granite-embedding-125m-english_. 
- [granite-embedding-small-english](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2) (**47M** parameters): A _first-of-its-kind_ reduced-size model, with fewer layers and a smaller output embedding size (_384_), replacing _granite-embedding-30m-english_. 

## Model Details

- **Developed by:** Granite Embedding Team, IBM
- **Repository:** [ibm-granite/granite-embedding-models](https://github.com/ibm-granite/granite-embedding-models)
- **Paper:** [Techincal Report](papers/GraniteEmbeddingR2.pdf)
- **Language(s) (NLP):** English
- **Release Date**: Aug 15, 2025
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Usage

**Intended Use:** The model is designed to produce fixed length vector representations for a given text, which can be used for text similarity, retrieval, and search applications.

For efficient decoding, these models use Flash Attention 2. Installing it is optional, but can lead to faster inference.

```shell
pip install flash_attn==2.6.1
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

## Evaluation Results
Granite embedding r2 models show a strong performance across tasks diverse tasks. 

Performance of the granite models on MTEB Retrieval (i.e., BEIR), MTEB-v2, code retrieval (CoIR), long-document search benchmarks (MLDR, LongEmbed), conversational multi-turn (MTRAG), 
table retrieval (NQTables, OTT-QA, AIT-QA, MultiHierTT, OpenWikiTables),  benchmarks is reported in the below tables. 

The r2 models demonstrates speed and efficiency, while mainintaining competitive performance. The average speed to encode documents on a single H100 GPU using a sliding window with 512 context length chunks is also reported. 

| Model                              | Parameters (M) | Embedding Size | BEIR Retrieval (15) | MTEB-v2 (41)| CoIR (10) | MLDR (En) | MTRAG (4) |  Encoding Speed (docs/sec) |
|------------------------------------|:--------------:|:--------------:|:-------------------:|:-----------:|:---------:|:---------:|:---------:|:-------------------------------:|
| granite-embedding-125m-english     |      125       |      768       |        52.3         |     62.1   |   50.3    |   35.0    |   49.4   |               149             |
| granite-embedding-30m-english      |       30       |      384       |        49.1         |     60.2   |   47.0    |   32.6    |   48.6   |               198             |
| granite-embedding-english-r2       |      149       |      768       |        53.1         |     62.8   |   55.3    |   40.7    |   56.7   |               144              |
| granite-embedding-small-english-r2 |       47       |      384       |        50.9         |     61.1   |   53.8    |   39.8    |   48.1   |               199             |


|Model                              | Parameters (M) | Embedding Size |**AVERAGE**|MTEB-v2 Retrieval (10) | CoIR (10) | MLDR (En) | LongEmbed (6)| Table IR (5)| MTRAG(4) |  Encoding Speed (docs/sec) |
|-----------------------------------|:--------------:|:--------------:|:---------:|:---------------------:|:---------:|:---------:|:------------:|:-----------:|:--------:|-------------------------------:|
|e5-base-v2                         |109|768|47.5|49.7|50.3|32.5|41.1|74.09|37.0| 115|
|bge-base-en-v1.5                   |109|768|46.9|54.8|46.6|33.5|33.9|73.98|38.8| 116|
|snowflake-arctic-embed-m-v2.0      |305|768|51.4|58.4|52.2|32.4|55.4|80.75|29.2| 106|
|gte-base-en-v1.5                   |137|768|52.8|55.5|42.4|42.7|59.4|80.52|36.0| 116|
|gte-modernbert-base                |149|768|57.5|57.0|71.5|46.2|57.0|76.68|36.8| 142|
|nomic-ai/modernbert-embed-base     |149|768|48.0|48.7|48.8|31.3|56.3|66.69|36.2| 141|
|||||||||||
|granite-embedding-english-r2       |149|768|**59.5**|56.4|54.8|41.6|67.8|78.53|57.6| 144|
|granite-embedding-small-english-r2 | 47|384|55.6|53.9|53.4|40.1|61.9|75.51|48.9|199|


### Model Architecture and Key Features

The latest granite embedding r2 release introduces two English embedding models, both based on the ModernBERT architecture:
- _granite-embedding-english-r2_ (**149M** parameters): with an output embedding size of _768_, replacing _granite-embedding-125m-english_. 
- _granite-embedding-small-english-r2_ (**47M** parameters): A _first-of-its-kind_ reduced-size model, with fewer layers and a smaller output embedding size (_384_), replacing _granite-embedding-30m-english_. 

The following table shows the structure of the two models:

| Model                     | granite-embedding-small-english-r2 | **granite-embedding-english-r2**   |
| :---------                | :-------:|:--------:| 
| Embedding size            | 384      | **768**      | 
| Number of layers          | 12       | **22**       | 
| Number of attention heads | 12       | **12**       | 
| Intermediate size         | 1536     | **1152**     | 
| Activation Function       | GeGLU    | **GeGLU**    | 
| Vocabulary Size           | 50368    | **50368**    | 
| Max. Sequence Length      | 8192     | **8192**     | 
| # Parameters              | 47M      | **149M**     | 


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
