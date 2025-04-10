
import streamlit as st
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import joblib
from datetime import datetime, timedelta

# Load model and scaler
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

@st.cache_resource
def load_model():
    model = LSTMModel()
    model.load_state_dict(torch.load("model/lstm_model.pt", map_location=torch.device('cpu')))
    model.eval()
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# UI
st.title("📈 Stock Price Predictor (LSTM)")
st.markdown("Enter a stock ticker to fetch the last 60 days of data and predict the next closing price.")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")

if st.button("Fetch & Predict"):
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=90)
        df = yf.download(ticker, start=start_date, end=end_date)
        df = df.tail(60)[['Open', 'High', 'Low', 'Close', 'Volume']]
        if df.shape[0] < 60:
            st.error("Not enough data to make prediction. Need full 60 days.")
        else:
            sequence = df.values
            scaled = scaler.transform(sequence)
            tensor_input = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                prediction = model(tensor_input).numpy()
            dummy_input = np.zeros((1, 5))
            dummy_input[0, 3] = prediction[0][0]
            predicted_price = scaler.inverse_transform(dummy_input)[0][3]
            st.success(f"📊 Predicted Closing Price: **${predicted_price:.2f}**")
            st.dataframe(df.reset_index())
    except Exception as e:
        st.error(f"Error: {str(e)}")
