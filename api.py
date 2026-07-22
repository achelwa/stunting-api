"""
Flask REST API — Stunting Prediction
=====================================
"""

import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from predict_pipeline import StuntingPredictor

app = Flask(__name__)
CORS(app)

# Tentukan path absolut folder artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')

# Load predictor sekali di startup
try:
    predictor = StuntingPredictor(artifacts_dir=ARTIFACTS_DIR)
    model_loaded = True
    print("=" * 60)
    print("✅ STUNTING PREDICTION API — SIAP")
    print("=" * 60)
    print(f"Model Path: {ARTIFACTS_DIR}")
    print(f"Model Class: {predictor.model.__class__.__name__}")
    print(f"Features   : {len(predictor.feature_order)}")
    print(f"Threshold  : {predictor.threshold}")
    print("=" * 60)
    print("\nTekan Ctrl+C untuk stop API.\n")
except Exception as e:
    predictor = None
    model_loaded = False
    print("=" * 60)
    print(f"❌ GAGAL LOAD MODEL: {e}")
    print("Detail Traceback:")
    traceback.print_exc()
    print("=" * 60)
    print(f"Pastikan folder berikut ada: {ARTIFACTS_DIR}")
    print("Dan berisi file model (e.g., model_xgb_full.pkl)")


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'api': 'Stunting Prediction API',
        'version': '1.0',
        'status': 'online' if model_loaded else 'model not loaded',
        'endpoints': {
            'GET /health': 'Health check',
            'POST /predict': 'Single prediction',
            'POST /predict/batch': 'Batch prediction',
        },
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded,
    }), 200 if model_loaded else 503


@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({
            'status': 'error',
            'message': 'Model tidak berhasil dimuat di server. Cek log konsol.'
        }), 503

    if not request.is_json:
        return jsonify({
            'status': 'error',
            'message': 'Request harus dalam format JSON'
        }), 400

    data = request.get_json()

    try:
        result = predictor.predict(data)
        return jsonify({
            'status': 'success',
            'data': result
        }), 200
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'type': 'validation_error'
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'type': 'internal_error'
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    if not model_loaded:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 503

    payload = request.get_json()
    if 'data' not in payload or not isinstance(payload['data'], list):
        return jsonify({
            'status': 'error',
            'message': "Request harus berisi 'data' sebagai list"
        }), 400

    results = []
    for i, item in enumerate(payload['data']):
        try:
            result = predictor.predict(item, return_explanation=False)
            results.append({'index': i, 'success': True, 'result': result})
        except Exception as e:
            results.append({'index': i, 'success': False, 'error': str(e)})

    return jsonify({
        'status': 'success',
        'total': len(results),
        'success_count': sum(1 for r in results if r['success']),
        'data': results
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint tidak ditemukan',
    }), 404


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
