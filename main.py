from flask import Flask, request, jsonify
from flask import Flask, render_template, request, redirect, url_for, session,send_file
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from flask import Flask
from flask_cors import CORS
app = Flask(__name__, template_folder="templates")
CORS(app)

# Load the saved SARIMA model
model_sarima = joblib.load('finalized_model.sav')


def calculate_average(lst):
    if not lst:
        return 0  # Return 0 if the list is empty to avoid division by zero error
    total = sum(lst)
    return total / len(lst)


@app.route('/predict_demand', methods=['POST'])
def predict_demand():
    data = request.get_json()
    product_code=data['Product_code']
    start_date = pd.to_datetime(data['start_date'])
    end_date = pd.to_datetime(data['end_date'])

    dates = []
    predicted_demand_values = []

    # Iterate over the range of dates
    for prediction_date in pd.date_range(start=start_date, end=end_date, freq='MS'):
        # Predict demand for the current date
        demand_prediction = model_sarima.predict(start=prediction_date, end=prediction_date)
        # Getting only the value of zeroth index since the diff() operation loses the first value
        demand_prediction = demand_prediction.iloc[0]
        # Append the current date and predicted demand value to the lists
        dates.append(prediction_date)
        predicted_demand_values.append(demand_prediction)

    avg = calculate_average(predicted_demand_values)

    # Plot the predicted demand
    plt.figure(figsize=(12, 6))
    plt.plot(dates, predicted_demand_values, marker='o', linestyle='-')
    plt.title(f'The trend of {product_code} from {start_date} to {end_date}')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('predicted_demand_plot.png')



    return jsonify({'average_demand': avg, 'plot_image': 'predicted_demand_plot.png'})

@app.route('/')
def home():

    return render_template('index.html')


@app.route('/get_image', methods=['POST'])
def get_image():

    return send_file("predicted_demand_plot.png", mimetype='image/jpg')


if __name__ == '__main__':
    app.run(debug=True)
