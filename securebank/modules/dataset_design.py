class Dataset_Designer:
    def extract(self, raw_dataset_filename):
        raw_dataset = pd.read_parquet(raw_dataset_filename)
        return raw_dataset
    
    def sample(self, raw_dataset, test_size=0.2, validation_size=0.1):
        train_data, test_data = train_test_split(raw_dataset, test_size=test_size, random_state=42)
        
        if validation_size > 0:
            train_data, validation_data = train_test_split(train_data, test_size=validation_size, random_state=42)
            return [train_data, test_data, validation_data]
        else:
            return [train_data, test_data]
    
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
            'storage': 'securebank/storage/dataset_splits/',
            'description': descriptions
        }
    
    def load(self, datasets, filenames):
        os.makedirs('securebank/storage/dataset_splits/', exist_ok=True)
        
        for dataset, filename in zip(datasets, filenames):
            output_path = os.path.join('securebank/storage/dataset_splits/', filename)
            dataset.to_parquet(output_path, index=False)
            print(f"Dataset saved to {output_path}")

