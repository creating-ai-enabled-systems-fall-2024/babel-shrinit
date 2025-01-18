from flask import Flask, request, jsonify
from pipeline import Pipeline
import os

app = Flask(__name__)

pipeline = Pipeline(version='random_forest')

@app.route('/predict/', methods=['POST'])
def predict():
    """
    Endpoint to classify a transaction as legitimate/fraudulent.
    Accepts JSON input.
    """
    try:
        input_data = request.get_json()
        
        required_keys = [
            'trans_date_trans_time', 'cc_num', 'unix_time',
            'merchant', 'category', 'amt', 'merch_lat', 'merch_long'
        ]
        missing_keys = [key for key in required_keys if key not in input_data]
        if missing_keys:
            return jsonify({
                "error": f"Missing required keys: {', '.join(missing_keys)}"
            }), 400
        
        prediction = pipeline.predict(input_data)
        result = "Fraudulent" if prediction else "Legitimate"
        
        return jsonify({
            "prediction": result,
            "status": "success"
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failure"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
