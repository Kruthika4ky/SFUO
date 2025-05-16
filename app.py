from flask import Flask, request, jsonify, render_template # type: ignore
from flask_mysqldb import MySQL # type: ignore
import numpy as np # type: ignore
import pickle
import pandas as pd # type: ignore

app = Flask(__name__)

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'  # Change if your MySQL password is different
app.config['MYSQL_DB'] = 'FertilizerOptimizer'

mysql = MySQL(app)

# Load the trained ML model
model = pickle.load(open('fertilizer_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')  # Main page for input

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from form
    data = request.form
    ph = float(data['pH'])
    nitrogen = float(data['nitrogen'])
    phosphorus = float(data['phosphorus'])
    potassium = float(data['potassium'])
    moisture = float(data['moisture'])
    crop_type = data['crop_type']
    weather_condition = data['weather_condition']

    # Prepare the input features for the model
    input_features = np.array([[ph, nitrogen, phosphorus, potassium, moisture]])

    # Get fertilizer recommendation from the ML model
    prediction = model.predict(input_features)
    
    # Store the recommendation in the MySQL database
    cursor = mysql.connection.cursor()
    cursor.execute("INSERT INTO recommendations (ph, nitrogen, phosphorus, potassium, moisture, crop_type, weather_condition, recommended_fertilizer) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                   (ph, nitrogen, phosphorus, potassium, moisture, crop_type, weather_condition, prediction[0]))
    mysql.connection.commit()

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
