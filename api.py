import pandas as pd
import numpy as np
import requests
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model
model = load_model('stock_lstm_model.h5')

# Function to load data from the external API
def load_data(symbol):
    BASE_URL = f"https://peridotnepal.xyz/api/company/get_company_graph_range/{symbol}/1371100000"
    headers = {
        "Permission": "2021D@T@f@RSt6&%2-D@T@"
    }
    response = requests.get(BASE_URL, headers=headers)
    
    if response.status_code != 200:
        return pd.DataFrame()  # Return an empty dataframe if there's an issue
    
    data = response.json().get('data', [])
    
    if not data:
        return pd.DataFrame()  # Return an empty dataframe if no data is found
    
    # Convert the response data into a pandas DataFrame
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['t'], unit='s')
    df.set_index('date', inplace=True)
    return df.filter(['c'])

# Function to predict stock price
def predict_price(data, model, scaler):
    last_60_days = data[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)

    X_test = []
    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    
    return float(pred_price[0][0])  # Convert numpy.float32 to Python float

# Define the endpoint
@app.route('/predict', methods=['GET'])
def get_predicted_price():
    # Get the stock symbol from the query parameters
    symbol = request.args.get('symbol')
    print(symbol)
    if not symbol:
        return jsonify({"error": "No symbol provided"}), 400

    # Load and filter data based on the symbol
    df = load_data(symbol)

    if df.empty:
        return jsonify({"error": f"No data found for symbol: {symbol}"}), 404

    # Initialize the scaler with the data for the given symbol
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(df.values)

    # Predict the stock price
    predicted_price = predict_price(df, model, scaler)
    
    # Return the result as JSON
    return jsonify({
        "symbol": symbol,
        "predicted_price": predicted_price
    })

if __name__ == '__main__':
    app.run(debug=True)
