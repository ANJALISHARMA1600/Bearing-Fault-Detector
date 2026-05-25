import numpy as np
import scipy.io
from scipy.fft import fft
from scipy.stats import kurtosis, skew
from flask import Flask, request, render_template, jsonify, Response
import joblib
import os
import json
import time

app = Flask(__name__)

pipeline = joblib.load('pipeline.pkl')
le = joblib.load('label_encoder.pkl')

def extract_features(signal, fs=12000, N=2048):
    seg = signal[:N]
    seg = seg - np.mean(seg)
    rms = np.sqrt(np.mean(seg**2))
    peak = np.max(np.abs(seg))

    return {
        'mean':         np.mean(seg),
        'std':          np.std(seg),
        'rms':          rms,
        'peak':         peak,
        'kurtosis':     kurtosis(seg),
        'skewness':     skew(seg),
        'crest_factor': peak / (rms + 1e-10),
        'peak_to_peak': np.max(seg) - np.min(seg),
        'fft_mean':     np.mean(np.abs(fft(seg))[:N//2]),
        'fft_std':      np.std(np.abs(fft(seg))[:N//2]),
        'fft_peak':     np.max(np.abs(fft(seg))[:N//2]),
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.endswith('.mat'):
        return jsonify({'error': 'Only .mat files are supported'}), 400

    try:
        temp_path = os.path.join('static', 'temp_upload.mat')
        file.save(temp_path)

        mat = scipy.io.loadmat(temp_path)
        keys = [k for k in mat.keys() if not k.startswith('__')]
        de_key = [k for k in keys if 'DE_time' in k]

        if not de_key:
            return jsonify({'error': 'No Drive End (DE_time) signal found'}), 400

        signal = mat[de_key[0]].flatten()

        if len(signal) < 2048:
            return jsonify({'error': f'Signal too short. Need 2048 samples, got {len(signal)}'}), 400

        features = extract_features(signal)
        feature_array = np.array(list(features.values())).reshape(1, -1)

        prediction_encoded = pipeline.predict(feature_array)[0]
        prediction_label = le.inverse_transform([prediction_encoded])[0]
        probabilities = pipeline.predict_proba(feature_array)[0]
        confidence = round(float(np.max(probabilities)) * 100, 2)

        os.remove(temp_path)

        return jsonify({
            'prediction': prediction_label,
            'confidence': confidence,
            'signal_length': len(signal),
            'features': {k: round(float(v), 4) for k, v in features.items()}
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/simulate', methods=['POST'])
def simulate():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if not file.filename.endswith('.mat'):
        return jsonify({'error': 'Only .mat files are supported'}), 400

    try:
        temp_path = os.path.join('static', 'temp_simulate.mat')
        file.save(temp_path)

        mat = scipy.io.loadmat(temp_path)
        keys = [k for k in mat.keys() if not k.startswith('__')]
        de_key = [k for k in keys if 'DE_time' in k]

        if not de_key:
            return jsonify({'error': 'No DE_time signal found'}), 400

        signal = mat[de_key[0]].flatten()
        np.save(os.path.join('static', 'sim_signal.npy'), signal)

        return jsonify({
            'status': 'ready',
            'total_samples': len(signal),
            'total_windows': (len(signal) - 2048) // 1024
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stream')
def stream():
    def generate():
        signal_path = os.path.join('static', 'sim_signal.npy')
        if not os.path.exists(signal_path):
            yield f"data: {json.dumps({'error': 'No signal loaded'})}\n\n"
            return

        signal = np.load(signal_path)
        N = 2048
        stride = 1024
        fs = 12000
        start = 0
        window_num = 0

        while start + N <= len(signal):
            window = signal[start:start + N]
            features = extract_features(window, fs, N)
            feature_array = np.array(list(features.values())).reshape(1, -1)

            prediction_encoded = pipeline.predict(feature_array)[0]
            prediction_label = le.inverse_transform([prediction_encoded])[0]
            probabilities = pipeline.predict_proba(feature_array)[0]
            confidence = round(float(np.max(probabilities)) * 100, 2)

            time_ms = round(start / fs * 1000, 1)

            payload = {
                'window': window_num,
                'time_ms': time_ms,
                'prediction': prediction_label,
                'confidence': confidence,
                'features': {k: round(float(v), 4) for k, v in features.items()},
                'waveform': window[::10].tolist()
            }

            yield f"data: {json.dumps(payload)}\n\n"

            start += stride
            window_num += 1
            time.sleep(0.5)

        yield f"data: {json.dumps({'done': True, 'total_windows': window_num})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True, threaded=True)