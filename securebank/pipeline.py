import os
import pickle
from typing import Dict
import pandas as pd

class Pipeline:
    def __init__(self, version: str):
        self.version = version
        self.model_path = f'storage/models/artifacts/{version}.pkl'
        self.preprocessor_path = f'storage/models/artifacts/preprocessor.pkl'

        self.model = None
        
        if os.path.exists(self.preprocessor_path):
            with open(self.preprocessor_path, 'rb') as preprocessor_file:
                self.preprocessor = pickle.load(preprocessor_file)
        else:
            self.preprocessor = None  

        # history for predictions
        self.history = []

    def select_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        
        with open(self.model_path, 'rb') as model_file:
            return pickle.load(model_file)

    def predict(self, input_data: Dict) -> bool:
        input_features = pd.DataFrame([input_data])
        if self.preprocessor:
            input_features = self.preprocessor.transform(input_features)

        prediction = self.model.predict(input_features)[0]
        self.history.append({'input': input_data, 'prediction': prediction})
        return bool(prediction)

    def get_history(self) -> Dict:
        return {'history': self.history}
