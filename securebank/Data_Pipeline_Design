# Data Pipeline Design Documentation - Assignment 2

This document discusses the outline and design of the data pipeline for the SecureBank fraud detection system. 

## Module 1: Raw Data Handler

### Purpose
The `Raw_Data_Handler` class is the initial entry point for raw data, performing extraction and cleaning. It prepares the data for future processing and analysis.

### Design Decisions
- **Data Extraction**: The data is extracted from three locations: `customers.csv`, `transactions.parquet`, and `fraud_information.json`. 
- **Transformation**: The transformation method merges these three datasets based off of the relevant keys (`cc_num` for customer and transaction data, and `trans_num` for transaction and fraud data). THis keeps all information for a transaction organized in one record. 
- **Handling Missing Data**: The `is_fraud` column is important for our analysis. We fill missing values with `0` (assuming a transaction is not fraudulent). This is a standard practice in analyzing fraud data.

## Module 2: Dataset Designer

### Purpose
The `Dataset_Designer` class handles the partitioning of the dataset into training, testing, and validation subsets. This is standard for any machine learning pipeline.

### Design Decisions
- **Sampling**: We use stratified sampling to for the distribution of the `is_fraud` class across different datasets. Stratified sampling helps avoid overfitting of our machine learning model. 
- **Partition Sizes**: The default partition sizes are 70% training, 20% testing, and 10% validation. These ratios are standard to provide a balance between training the model and testing it, as well as validation.

## Module 3: Feature Extractor

### Purpose
The `Feature_Extractor` class is prepares the data for training by scaling numerical features and encoding categorical variables.

### Design Decisions
- **Feature Scaling**: We scale the `amt` (amount) feature using `StandardScaler`. Normalization helps standardize variables that are continuous so that models can find convergence faster. 
- **Return Structured Data**: This ensures each dataset (training, testing, and validation) are returned in a structured way.

## Conclusion

The design of every module in our pipeline meets the expectations and standards of a fraud detection system in banking transactions and is organized so that a nontechnical person will also be able to understand our methodology. 