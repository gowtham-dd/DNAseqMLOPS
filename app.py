from flask import Flask, render_template, request, redirect, url_for, jsonify
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import math
import os
import json
from flask import request, jsonify, render_template, redirect, url_for


# Load trained models
MODEL_DIR = "artifacts/model_trainer/models"

models = {
    "RandomForest": joblib.load(os.path.join(MODEL_DIR, "RandomForest.joblib")),
    "SVM": joblib.load(os.path.join(MODEL_DIR, "SVM.joblib")),
    "XGBoost": joblib.load(os.path.join(MODEL_DIR, "XGBoost.joblib"))
}

app = Flask(__name__)
from collections import Counter
import math

# Helper functions
def nucleotide_composition(seq):
    return {
        'length': len(seq),
        'A_perc': seq.count('A')/len(seq),
        'C_perc': seq.count('C')/len(seq),
        'G_perc': seq.count('G')/len(seq),
        'T_perc': seq.count('T')/len(seq),
        'GC_content': (seq.count('G') + seq.count('C')) / len(seq)
    }

def get_kmers(sequence, k=3):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def shannon_entropy(seq):
    counts = Counter(seq)
    probs = [c / len(seq) for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def preprocess_sequences(sequences):
    # Load feature engineering artifacts
    vectorizer = joblib.load("artifacts/data_transformation/kmer_vectorizer.joblib")
    selector = joblib.load("artifacts/data_transformation/feature_selector.joblib")

    features = []
    for seq in sequences:
        comp = nucleotide_composition(seq)
        kmers = ' '.join(get_kmers(seq, 3))
        kmer_features = vectorizer.transform([kmers]).toarray()[0]
        complexity = {
            'entropy': shannon_entropy(seq),
            'unique_kmers': len(set(get_kmers(seq, 3))),
            'repeats': len(seq) - len(set(seq))
        }

        all_features = list(comp.values()) + list(kmer_features) + list(complexity.values())
        features.append(all_features)

    return selector.transform(features)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result')
def result():
    results_json = request.args.get('data')
    if results_json:
        try:
            results = json.loads(results_json)
            return render_template('result.html', results=results)
        except json.JSONDecodeError:
            return redirect(url_for('home'))
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Load sequences from file or JSON body
        if 'jsonFile' in request.files:
            file = request.files['jsonFile']
            if file.filename == '':
                return jsonify({"error": "Empty file uploaded"}), 400
            try:
                data = json.load(file)
                sequences = data.get('sequences', [])
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid JSON file"}), 400
        elif request.is_json:
            # 2. Handle JSON from fetch()
            data = request.get_json()
            sequences = data.get('sequences', [])
        else:
            # 3. Handle textarea input from form
            sequences = request.form.get('sequences', '').split('\n')
            sequences = [seq.strip() for seq in sequences if seq.strip()]

        if not sequences:
            return jsonify({"error": "No sequences provided"}), 400

        # 4. Preprocess and predict
        X = preprocess_sequences(sequences)
        results = []

        for model_name, model in models.items():
            preds = model.predict(X)
            probs = model.predict_proba(X)

            model_results = []
            for i, seq in enumerate(sequences):
                model_results.append({
                    "sequence": seq,
                    "prediction": int(preds[i]),
                    "confidence": float(probs[i][preds[i]]),
                    "probabilities": {
                        "class_0": float(probs[i][0]),
                        "class_1": float(probs[i][1])
                    }
                })

            results.append({
                "model": model_name,
                "predictions": model_results
            })

        # 5. Return results depending on request type
        if request.is_json or 'jsonFile' in request.files:
            return jsonify(results)  # API call via fetch()
        else:
            return redirect(url_for('result', data=json.dumps(results)))  # Traditional form
    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        return jsonify({"error": str(e)}), 500




if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    app.run(debug=True, host='0.0.0.0', port=5000)
