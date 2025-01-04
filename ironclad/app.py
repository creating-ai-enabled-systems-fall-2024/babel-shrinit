from flask import Flask, request, jsonify
from pipeline import Pipeline
import os
import json

app = Flask(__name__)

# Initialize the pipeline
model_name = os.getenv('MODEL_NAME', 'casia-webface')
device = os.getenv('DEVICE', 'cpu')
pipeline = Pipeline(model_name=model_name, device=device)

# Load embeddings (ensure the catalog exists)
catalog_dir = os.getenv('CATALOG_DIR', 'storage/catalog')
if os.path.exists(catalog_dir):
    pipeline.index.load(catalog_dir)
else:
    os.makedirs(catalog_dir, exist_ok=True)

# Track history of predictions
history = []

@app.route('/identify', methods=['POST'])
def identify():
    file = request.files['file']
    k = int(request.args.get('k', 5))
    probe_path = os.path.join('temp', file.filename)
    os.makedirs('temp', exist_ok=True)
    file.save(probe_path)

    try:
        results = pipeline.search_gallery(probe_path, k)
        os.remove(probe_path)

        # Record history
        history_entry = {"probe": file.filename, "results": results}
        history.append(history_entry)

        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add', methods=['POST'])
def add():
    file = request.files['file']
    identity = request.form['identity']
    identity_dir = os.path.join('storage/gallery', identity)
    os.makedirs(identity_dir, exist_ok=True)

    image_path = os.path.join(identity_dir, file.filename)
    file.save(image_path)

    try:
        # Precompute embedding and add to index
        embedding = pipeline._Pipeline__encode(image_path)
        pipeline.index.add(embedding, image_path)
        pipeline.index.save(catalog_dir)

        return jsonify({'message': f'Added {file.filename} to {identity}.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(history)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
