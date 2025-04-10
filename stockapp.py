from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import joblib
import os
import yfinance as yf
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# --- LSTM model class definition ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64)
        c0 = torch.zeros(2, x.size(0), 64)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# --- Load model and scaler ---
MODEL_PATH = "model/lstm_model.pt"
SCALER_PATH = "model/scaler.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = LSTMModel()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    scaler = joblib.load(SCALER_PATH)
else:
    model = None
    scaler = None

# --- Routes ---
@app.route('/')
def home():
    return render_template("display.html")

@app.route('/predictor')
def predictor_page():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded.'}), 500

    try:
        data = request.get_json()
        sequence = np.array(data['sequence'])
        if sequence.shape != (60, 5):
            return jsonify({'error': 'Sequence must be shape [60, 5].'}), 400

        sequence_scaled = scaler.transform(sequence)
        sequence_tensor = torch.tensor(sequence_scaled, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            prediction = model(sequence_tensor).numpy()

        dummy_input = np.zeros((1, 5))
        dummy_input[0, 3] = prediction[0][0]  # 'Close' is at index 3
        predicted_price = scaler.inverse_transform(dummy_input)[0][3]

        return jsonify({'predicted_price': round(float(predicted_price), 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fetch', methods=['POST'])
def fetch_stock_data():
    try:
        ticker = request.json['ticker'].upper()
        end_date = datetime.today()
        start_date = end_date - timedelta(days=90)
        df = yf.download(ticker, start=start_date, end=end_date)

        df = df.tail(60)[['Open', 'High', 'Low', 'Close', 'Volume']]
        data = df.values.tolist()

        if len(data) < 60:
            return jsonify({'error': 'Not enough data for this stock.'}), 400

        return jsonify({'data': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
