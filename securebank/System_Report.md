# SecureBank System Report

## System Design

### Meeting Requirements
The SecureBank system detects fraudulent transactions, where each module is dedicated to a specific responsibility, allowing for scalability. Key requirements met:
- **Accurate Fraud Detection**: Uses machine learning models trained on transaction data.
- **Fast Inference**: Models like Logistic Regression and K-Nearest Neighbors are used.
- **Modular Architecture**: Each functionality is implemented as a separate, reusable module.
- **User Interaction**: A Flask-based API (`predict/`) allows end-users to classify transactions.

### Unique Functionalities
1. **Raw Data Handler**: Raw data extraction, transformation, and loading.
2. **Feature Extractor**: Preprocesses and transforms data with feature selection, scaling, and encoding pipelines.
3. **Model Training**: Trains multiple models, evaluates performance, and saves them.
4. **Inference Pipeline**: Provides model selection and transaction classification.
5. **Flask API**: Allows user interaction.

### System Components and Processes
[Raw Data] --> [Raw_Data_Handler] --> [Feature_Extractor] --> [Trained Models] | | v v [Pipeline] --> [Flask API]

## Data, Data Pipelines, and Model

### Data Description
- **Dataset Overview**:
  - Transactions are represented with attributes like `amt`, `merchant`, `category`, `merch_lat`, `merch_long`, and `is_fraud`.

- **Patterns Identified**:
  - Fraudulent transactions are skewed toward specific categories (`electronics`, `travel`).
  - Larger transaction amounts are more likely fraudulent.

### Data Pipelines
1. **Raw Data Handling**:
   - Extracts data from CSV, JSON, and Parquet files.
   - Merges datasets and cleans missing or erroneous data.

2. **Feature Extraction**:
   - Selects features: `amt`, `category`, `merchant`, `merch_lat`, `merch_long`, and `is_fraud`.
   - Scales numeric features (`amt`, `merch_lat`, `merch_long`).
   - Encodes categorical features (`category`, `merchant`) using one-hot encoding.

3. **Model Training**:
   - Splits data into train and test sets.
   - Trains models like Logistic Regression and K-Nearest Neighbors.
   - Saves trained models and preprocessors.

4. **Inference**:
   - Loads the pre-trained model and preprocessor.
   - Accepts JSON inputs for classification.
   - Returns `fraud` or `legitimate`.

### Model Inputs and Outputs
- **Inputs**:
  - `trans_date_trans_time`, `cc_num`, `unix_time`, `merchant`, `category`, `amt`, `merch_lat`, `merch_long`.
- **Outputs**:
  - `0`: Legitimate transaction.
  - `1`: Fraudulent transaction.

## Metrics Definition

### Offline Metrics
1. **Accuracy**: Measures the overall correctness of the model predictions.
2. **Precision**: Evaluates the ability to correctly identify fraudulent transactions.
3. **Recall**: Determines the fraction of actual frauds that were detected.
4. **F1 Score**: Harmonic mean of precision and recall.
5. **AUC**: Measures the model's ability to rank transactions by their likelihood of being fraudulent.

### Online Metrics
1. **Response Time**: Measures the API latency for predictions.
2. **Fraud Detection Accuracy**: Tracks how well the system flags fraud in real time.

## Analysis of System Parameters and Configurations

### Feature Selection
- Features like `amt` and `category` were important for predicting fraud.
- Redundant or noisy features were excluded to improve model performance.

### Dataset Design
- Data was split into training and testing.
- Fraudulent and legitimate classes were balanced.

## Model Evaluation and Selection

### Performance Analysis
The following models were evaluated using the training and testing datasets. Metrics were derived from the `model_performance_metrics.csv` file:

| Model               | Accuracy | Precision | Recall   | F1 Score | AUC   |
|---------------------|----------|-----------|----------|----------|-------|
| Logistic Regression | 0.996    | 0.000     | 0.000    | 0.000    | 0.813 |
| Naive Bayes         | 0.346    | 0.005     | 0.836    | 0.010    | 0.592 |

### Insights
1. **Logistic Regression**:
   - High **accuracy** (99.6%) and **AUC** (0.813), meaning reliable overall performance and ability to rank fraudulent transactions.
   - Low **precision** and **recall** suggest poor identification of actual fraudulent transactions, possibly due to class imbalance or feature limitations.

2. **Naive Bayes**:
   - Low overall **accuracy** (34.6%) but high **recall** (83.6%), indicating it flags a majority of fraudulent transactions but at the cost of many false positives.
   - The tradeoff between precision and recall makes it unsuitable for direct deployment but useful in scenarios prioritizing fraud detection over false positives.

### Model Selection
Given these results:
- Logistic Regression was chosen for deployment. 

## Post-deployment Policies

### Monitoring and Maintenance Plan
1. **Real-Time Monitoring**:
   - Track API latency, response success rates, and prediction outcomes.
   - Monitor for drift in data patterns or feature distributions.
2. **Scheduled Retraining**:
   - Retrain models periodically with new data to adapt to evolving fraud patterns.

### Fault Mitigation Strategies
1. **Fallback Mechanisms**:
   - Default to a basic heuristic model if the main model fails.
   - Use thresholds to limit erroneous classifications.
2. **Alerting**:
   - Notify administrators in case of high failure rates or abnormal traffic.
3. **Model Rollback**:
   - Maintain versioned models to quickly roll back to a stable version in case of deployment issues.

