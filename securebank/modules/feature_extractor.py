class Feature_Extractor:
    def extract(self, training_dataset_filename, testing_dataset_filename, validation_dataset_filename=None):
        training_dataset = pd.read_parquet(training_dataset_filename)
        testing_dataset = pd.read_parquet(testing_dataset_filename)
        
        if validation_dataset_filename:
            validation_dataset = pd.read_parquet(validation_dataset_filename)
            return training_dataset, testing_dataset, validation_dataset
        else:
            return training_dataset, testing_dataset
    
    def transform(self, training_dataset, testing_dataset, validation_dataset=None):
        scaler = StandardScaler()
        
        training_dataset['amt_scaled'] = scaler.fit_transform(training_dataset[['amt']])
        testing_dataset['amt_scaled'] = scaler.transform(testing_dataset[['amt']])
        
        if validation_dataset is not None:
            validation_dataset['amt_scaled'] = scaler.transform(validation_dataset[['amt']])
            return [training_dataset, testing_dataset, validation_dataset]
        else:
            return [training_dataset, testing_dataset]
    
    def describe(self, *datasets):
        descriptions = {}
        for i, dataset in enumerate(datasets):
            descriptions[f'dataset_{i+1}'] = {
                'num_records': len(dataset),
                'num_fraudulent': dataset['is_fraud'].sum(),
                'num_non_fraudulent': len(dataset) - dataset['is_fraud'].sum(),
            }
        return {
            'version': '1.0',
            'storage': 'securebank/storage/feature_data/',
            'description': descriptions
        }
    
    def load(self, datasets, filenames):
        os.makedirs('securebank/storage/feature_data/', exist_ok=True)
        
        for dataset, filename in zip(datasets, filenames):
            output_path = os.path.join('securebank/storage/feature_data/', filename)
            dataset.to_parquet(output_path, index=False)
            print(f"Feature data saved to {output_path}")
