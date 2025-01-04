# System Report


## Overview
The system is a machine learning-based nearest neighbor search. It has three services: **Extraction Service**, **Retrieval Service**, and **Interface Service**. These components work together to preprocess image inputs, extract embeddings, index and retrieve embeddings for nearest neighbor search, and provide an API for the user.

## System Components and Processes
1. **Extraction Service**  
   - **Description**: This service preprocesses image inputs (resizing, normalization, etc) and extracts feature embeddings using a pre-trained deep learning model like `casia-webface` or `vggface2`. 
   - **Process**:  
     1. Load and preprocess the input image.
     2. Pass the image through the embedding extractor.
     3. Return embedding vector.

2. **Retrieval Service**  
   - **Description**: This service indexes embeddings and performs nearest neighbor searches using FAISS, a library for similarity search on large datasets.  
   - **Process**:  
     1. Add embeddings to the FAISS index during precomputation/incremental addition.
     2. Search the FAISS index for knn of a probe embedding.

3. **Interface Service**  
   - **Description**: This service gives endpoints (`/identify`, `/add`, and `/history`) through a Flask API. These endpoints allow user interaction, incremental addition of data, and retrieval of historical results.  
   - **Process**:  
     1. Input user input through HTTP requests.
     2. Invoke the appropriate service method.
     3. Return results as JSON.

---

## Metrics Definition

### Offline Metrics
**Purpose**: Offline metrics evaluate the system's performance during development and testing, so  that it meets accuracy and efficiency requirements.

1. **Precision-Recall**: Measures the quality of the nearest neighbor search by calculating precision and recall. 
2. **Embedding Space Analysis**: Evaluates the separability of embedding vectors for cluster performance.

### Online Metrics
**Purpose**: Online metrics are used to monitor the system in production.

1. **Latency**: Measures the time taken for the system to process requests (`/identify` and `/add`). Monitored via API logs.
2. **Throughput**: Monitors the number of requests handled per second. Especially important for high loads.
3. **Accuracy of Search Results**: Occasionally validated using a subset of labeled data.

**Monitoring**: Online metrics are logged and visualized.

---

## Analysis of System Parameters and Configurations

### Significant Design Decisions

#### **1. Choice of Embedding Model**
   - **Decision**: Selected `casia-webface` and `vggface2` as pre-trained models for feature extraction.  
   - **Impact**: The choice affects the accuracy and generalization of embeddings. `vggface2` allows better performance for general datasets, while `casia-webface` is perfect for specific facial features.  
   - **Analysis**: The models were evaluated using precision-recall curves and embedding separability metrics. Results showed that `vggface2` had a slight edge in diverse datasets.  

#### **2. FAISS Indexing Strategy**
   - **Decision**: Used `IndexFlatL2` for simplicity and precision, with scalability to billions of embeddings.  
   - **Impact**: Provides exact nearest neighbor search but requires significant memory.  
   - **Analysis**: Compared `IndexFlatL2` with `IndexIVFPQ`. While `IndexIVFPQ` reduced memory usage, its recall was lower for large datasets. 

#### **3. k Parameter for Nearest Neighbor Search**
   - **Decision**: Allowed user-defined `k` with a default value of 5.  
   - **Impact**: Larger `k` increases query time but improves result diversity.  
   - **Analysis**: Optimal `k` was determined by analyzing precision-recall trade-offs. A value of 5 achieved a good balance of performance and accuracy.

#### **4. Preprocessing Pipeline**
   - **Decision**: Standardized preprocessing with resizing, normalization, and center cropping to 160x160 dimensions.  
   - **Impact**: Allows for compatibility with the embedding model while maintaining input consistency.  
   - **Analysis**: Image transformations and its effects on embedding quality showed that center cropping reduced edge artifacts which improved separability. 

#### **5. Incremental Addition of Data**
   - **Decision**: Supported incremental addition of embeddings to the index via the `/add` endpoint.  
   - **Impact**: Allows the system to dynamically grow without retraining.  
   - **Analysis**: Evaluated the overhead of incremental updates. There was a minimal impact on query latency for up to 1M embeddings.

### Extraction Service Analysis

1. **Embedding Model Comparison**  
   **Graph**: Precision-Recall curves for `casia-webface` and `vggface2`.  
   **Findings**: `vggface2` had higher recall at equivalent precision and accuracy, making it suitable for general use cases.

2. **Preprocessing Effects**  
   **Graph**: Accuracy vs. Preprocessing Pipeline (with/without normalization).  
   **Findings**: Normalization improves embedding separability, reducing intra-class variance. Noise reduction wasn't that helpful aside for images with poor quality, and sometimes reduced the performance of the general model.

### Retrieval Service Analysis

1. **Indexing Strategy Comparison**  
   **Findings**: `IndexFlatL2` was chosen for its high recall despite higher memory usage.

2. **k-Value Tuning**  
   **Graph**: Precision vs. k for Nearest Neighbor Search.  
   **Findings**: Increasing `k` beyond 5 improved recall at the expense of precision and response speeds, making `k=5` the best value.

### Evidence-Based Design Influences

1. **Embedding Model**: Adopted `vggface2` as the default for its robust generalization.  
2. **Index Strategy**: Used `IndexFlatL2` for its precision, accepting higher memory usage as a trade-off.  
3. **k Parameter**: Defaulted to 5, balancing accuracy and latency for real-time applications.  
4. **Preprocessing Pipeline**: Standardized pipeline ensured model compatibility and performance.  
5. **Incremental Updates**: Implemented seamless addition to the index, avoiding downtime or retraining.

---

## Supporting Notebook Files


1. **notebooks/embedding_analysis.ipynb**:  Embedding models and noise transformations.
2. **notebooks/test_extraction.ipynb**: The preprocessing and embedding extraction pipeline.

---
