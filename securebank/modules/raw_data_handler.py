class Raw_Data_Handler:
    def extract(self, customer_information_filename, transaction_filename, fraud_information_filename):
        customer_information = pd.read_csv(customer_information_filename)
        
        transaction_information = pd.read_parquet(transaction_filename)
        
        with open(fraud_information_filename, 'r') as f:
            fraud_information = json.load(f)
        fraud_information = pd.DataFrame(list(fraud_information.items()), columns=['trans_num', 'is_fraud'])
        
        return customer_information, transaction_information, fraud_information

    def transform(self, customer_information, transaction_information, fraud_information):
        raw_data = pd.merge(transaction_information, fraud_information, on='trans_num', how='left')
        raw_data = pd.merge(raw_data, customer_information, on='cc_num', how='left')
        
        raw_data['is_fraud'].fillna(0, inplace=True)  
        
        
        return raw_data
    
    def describe(self, raw_data):
        description = {
            'version': '1.0',
            'storage': 'securebank/storage/raw_data/',
            'description': {
                'num_records': len(raw_data),
                'num_fraudulent': raw_data['is_fraud'].sum(),
                'num_non_fraudulent': len(raw_data) - raw_data['is_fraud'].sum(),
            }
        }
        return description
    
    def load(self, raw_data, output_filename):
        output_path = os.path.join('securebank/storage/raw_data/', output_filename)
        raw_data.to_parquet(output_path, index=False)
        print(f"Data saved to {output_path}")
