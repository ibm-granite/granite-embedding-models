---
## Introduction to Granite Embedding Models
The Granite Embedding collection delivers innovative sentence-transformer models purpose-built for retrieval-based applications. Featuring a bi-encoder architecture, these models generate high-quality embeddings for textual inputs such as queries, passages, and documents, enabling seamless comparison through cosine similarity. Built using retrieval oriented pretraining, contrastive finetuning, knowledge distillation, and model merging, the Granite Embedding lineup is optimized to ensure strong alignment between query and passage embeddings. 

Built on a foundation of carefully curated, permissibly licensed public datasets, the Granite Embedding models set a high standard for performance, maintaining competitive scores not only on academic benchmarks such as BEIR, but also out-perfoming models of the same size on many enterprise use cases. Developed to meet enterprise-grade expectations, they are crafted transparently in accordance with IBM's AI Ethics principles and offered under the Apache 2.0 license for both research and commercial innovation. 

The Granite Embedding lineup includes four different models of varying sizes:
- granite-embedding-30m-english: English only model that produces embedding vectors of size 384. 
- granite-embedding-125m-english: English only model that produces embedding vectors of size 768. 
- granite-embedding-107m-multilingual: Multilingual model that produces embedding vectors of size 384. 
- granite-embedding-278m-multilingual: Multilingual model that produces embedding vectors of size 768. 

Accordingly, these options provide a range of models with different compute requirements to choose from, with appropriate trade-offs with their performance on downstream tasks.

## Data Collection
Granite embedding models are trained using data from of four key sources: (1) unsupervised title-body paired data scraped from the web, (2) publicly available paired with permissive, enterprise-friendly license, (3) IBM-internal paired data targeting specific technical domains, and (4) IBM-generated synthetic data. Notably, we do not use the popular MS-MARCO retrieval dataset in our training corpus due to its non-commercial license, while other open-source models train on this dataset due to its high quality. 

For governance, all our data undergoes a data clearance process subject to technical, business, and governance review. This comprehensive process captures critical information about the data, including but not limited to their content description ownership, intended use, data classification, licensing information, usage restrictions, how the data will be acquired, as well as an assessment of sensitive information (i.e, personal information). 

## Evaluation Results
The performance of the Granite Embedding English models on MTEB Retrieval (i.e., BEIR) and code retrieval (CoIR) benchmarks is reported below. The average time required to encode and retrieve per query is also reported. granite-embedding-30m-english is twice as fast as other models with similar embedding dimensions, while maintaining competitive performance.

| Model                           | Paramters (M)| Embedding Dimension |  MTEB Retrieval (15) |  CoIR (10) | Retrieval Time (seconds/query)|
|---------------------------------|:------------:|:-------------------:|:-------------------: |:----------:|:-----------------------------:|
|granite-embedding-30m-english    |30            |384                  |49.1                  |47.0        | 0.16                          |
|granite-embedding-125m-english   |125           |768                  |52.3                  |50.3        | 0.64                          |

The average performance of the Granite Embedding Multilingual models on Multilingual Miracl (across 18 langauges), Mintaka Retrieval (across 8 languages) and MTEB Retrieval for English (across 15 tasks), German (across 4 tasks), Spanish (across 2 tasks), Frenc (across 5 tasks), Japanese (across 2 tasks), Arabic (1 task), Korean (1 task) and Chinese (across 8 tasks) is reported below. The average time required to encode and retrieve per query is also reported. 

| Model                              | Paramters (M)| Embedding Dimension | Miracl (18)   |  Mintaka Retrieval (8) | MTEB English (15) | MTEB German (4) |MTEB Spanish (2) | MTEB French (5) | MTEB Japanese (2) |  MTEB Arabic (1) | MTEB Korean (1) | MTEB Chinese (8) | Retrieval Time (seconds/query)|
|------------------------------------|:------------:|:-------------------:|:-------------:| :---------------------:|:-----------------:|:---------------:|:---------------:|:---------------:|:----------------:|:----------------:|----------------:|-----------------:|------------------------------:|
|granite-embedding-107m-multilingual | 107 | 384 | 55.9 | 22.6 | 45.3 | 70.3 | 48.7 | 51.1 | 59.0 | 63.2 | 70.5 | 40.8 | 0.17|
|granite-embedding-278M-multilingual | 278 | 768 | 58.3 | 23.2 | 48.2 | 71.2 | 52.6 | 54.1 | 61.7 | 64.2 | 71.8 | 45.2 | 0.67|




**Model Architecture:**
Granite-Embedding-125m-English is based on an encoder-only RoBERTa like transformer architecture, trained internally at IBM Research.

| Model                     | granite-embedding-30m-english | granite-embedding-125m-english    | granite-embedding-107m-multilingual | granite-embedding-278m-multilingual |
| :---------                | :-------:|:--------:| :-----:| :-----:|
| Embedding size            | 384  | 768          | 384    | 768    |
| Number of layers          | 6    | 12           | 6      | 12     |
| Number of attention heads | 12   | 12           | 12     | 12     |
| Intermediate size         | 1536 | 3072         | 1536   | 3072   |
| Activation Function       | GeLU | GeLU         | GeLU   | GeLU   |
| Vocabulary Size           | 50265| 50265        | 250002 | 250002 |
| Max. Sequence Length      | 512  | 512          | 512    | 512    |
| # Parameters              | 30M  | 125M         | 107M   | 278M   |


## How to Use our Models?
To use any of our models, pick an appropriate `model_path` from:
1. `ibm-granite/granite-embedding-30m-english`
2. `ibm-granite/granite-embedding-125m-english`
3. `ibm-granite/granite-embedding-107m-multilingual`
4. `ibm-granite/granite-embedding-278m-multilingual`

### Inference
**Usage with Sentence Transformers:** 
This is a simple example of how to use granite-embedding-30m-english model with sentence_transformers.

First, install the sentence transformers library
```shell
pip install sentence_transformers
```

The model can then be used to encode pairs of text and find the similarity between their representations

```python
from sentence_transformers import SentenceTransformer, util

model_path = "ibm-granite/granite-embedding-30m-english"
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

# encode queries and passages
query_embeddings = model.encode(input_queries)
passage_embeddings = model.encode(input_passages)

# calculate cosine similarity
print(util.cos_sim(query_embeddings, passage_embeddings))
```

**Usage with Huggingface Transformers:** 
This is a simple example of how to use the granite-embedding-30m-english model with the Transformers library and PyTorch.

First, install the required libraries
```shell
pip install transformers torch
```

The model can then be used to encode pairs of text

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_path = "ibm-granite/granite-embedding-30m-english"

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
    # Perform pooling. granite-embedding-30m-english uses CLS Pooling
    query_embeddings = model_output[0][:, 0]

# normalize the embeddings
query_embeddings = torch.nn.functional.normalize(query_embeddings, dim=1)

```

## How to Download our Models?
The model of choice (granite-embedding-30m-english in this example) can be cloned using:
```shell
git clone https://huggingface.co/ibm-granite/granite-embedding-30m-english
```
## How to Contribute to this Project?
Plese check our [Guidelines](/CONTRIBUTING.md) and [Code of Conduct](/CODE_OF_CONDUCT.md) to contribute to our project.

## Model Cards
The model cards for each model variant are available in their respective HuggingFace repository. Please visit our collection [here](https://huggingface.co/collections/ibm-granite/granite-embedding-models-6750b30c802c1926a35550bb).

## License 
All Granite Embedding Models are distributed under [Apache 2.0](./LICENSE) license.

## Would you like to provide feedback?
Please let us know your comments about our family of embedding models by visiting our [collection](https://huggingface.co/collections/ibm-granite/granite-embedding-models-6750b30c802c1926a35550bb). Select the repository of the model you would like to provide feedback about. Then, go to *Community* tab, and click on *New discussion*. Alternatively, you can also post any questions/comments on our [github discussions page](https://github.com/orgs/ibm-granite/discussions).

## Citation
If you find granite models useful, please cite:

```
@misc{granite2024embedding,
  title={Granite Embedding Models},
  url={https://github.com/ibm-granite/granite-embedding-models/},
  author={Granite Embedding Team, IBM},
  month={December},
  year={2024}
}
```
