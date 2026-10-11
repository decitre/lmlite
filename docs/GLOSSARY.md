# Glossary

Terminology used in lmLite. Descriptions of proposed index features do not imply they are implemented.

| Term | Meaning |
|:-----|:--------|
| Embedding | A numerical vector representing a text passage or query. Comparisons require compatible embedding models and preparation settings. |
| Dimension (D) | The number of components in each embedding vector. Equal dimensions do not imply compatible embedding spaces. |
| Collection size (N) | The number of document vectors stored in the index. A document may be a short passage rather than an entire source file. |
| Vector index | A structure that retains vectors and associated document references so they can be searched without re-embedding the collection. |
| Cosine similarity | The dot product divided by the product of vector lengths. It measures alignment of nonzero vectors, not the probability that a document answers a question. |
| Inner product / dot product | The sum of products of corresponding vector components. It equals cosine similarity when both vectors have unit length. |
| L2 norm | A vector's Euclidean length: the square root of the sum of its squared components. |
| Normalization | Dividing a nonzero vector by its L2 norm to give it unit length. This does not make embeddings from different models compatible. |
| Pooling | Combining token-level representations into one passage vector. lmLite uses attention-mask-weighted mean pooling for embeddings. |
| Top-k | The k highest-ranked results under the chosen scoring and tie-breaking contract; it does not limit how many vectors exact search examines. |
| Exact / flat search | Exhaustively scoring all stored vectors to obtain the highest-ranked results under the index's numerical precision and tie rules. |
| FAISS | Facebook AI Similarity Search, a library for vector similarity search. Its index designs are useful references; the proposed browser index does not require the library. |
| IndexFlatIP | A FAISS flat index that performs exhaustive inner-product search. Normalized vectors make it suitable for cosine ranking. |
| Bounded min-heap | A structure holding at most k candidates, with the worst retained candidate at its root so better candidates can replace it efficiently. |
| Float32Array | A JavaScript typed array storing 32-bit floating-point values, using four bytes per component. |
| Provenance | Metadata identifying how vectors were produced, including model revision, dtype, tokenizer, input preparation, pooling, and normalization. |
| Dtype | The numerical format selected for model weights or computation, such as fp32 or q8. Model quantization is distinct from the proposed float32 index storage. |
| Backend | The model execution mechanism, such as WASM or WebGPU. It is separate from the algorithm used to search stored vectors. |
| Python-JavaScript bridge | The runtime interface used to exchange data and invoke JavaScript inference or search from browser Python. Large repeated transfers can add memory and latency costs. |
| Stable document ID | An identifier that remains associated with a document across updates, regardless of its current matrix row. |
| Row compaction | Reorganizing matrix rows after deletions to reclaim unused storage, while maintaining correct ID-to-row mappings. |
| ANN | Approximate nearest-neighbor search: trading potentially missed top results for lower search costs. It is not part of the initial exact index proposal. |
| HNSW | Hierarchical Navigable Small World, a graph-based ANN approach that navigates links between vectors rather than scanning every vector. |
| IVF | Inverted File index, which partitions vectors into clusters and searches selected clusters. It requires training and query-time probe selection. |
| Probe count | The number of IVF clusters examined for a query. More probes generally improve recall at the cost of more comparisons. |
| Recall@k | The fraction of exact top-k neighbors recovered by approximate search, measured using the same scoring and tie convention. |
| Latency | Time taken to complete an operation. Measure query embedding and vector search separately to identify the bottleneck. |
| Peak / steady-state memory | Maximum memory used during an operation versus memory retained after it completes. Construction can require more memory than the finished index. |
| MB / MiB | Decimal megabytes (1,000,000 bytes) versus binary mebibytes (1,048,576 bytes). |
| RAG | Retrieval-augmented generation: retrieving passages and supplying them to a generator as context. A vector index alone does not implement RAG or guarantee factual answers. |
