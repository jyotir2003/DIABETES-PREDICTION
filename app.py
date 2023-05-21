import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = pickle.load(open('models/lr.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Route for home page
@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route('/predict_datapoint', methods=['POST','GET'])
def predict_datapoint():
    if request.method == 'POST':
        pregnancies = float(request.form.get('Pregnancies'))
        glucose = float(request.form.get('Glucose'))
        blood_pressure = float(request.form.get('BloodPressure'))
        skin_thickness = float(request.form.get('SkinThickness'))
        insulin = float(request.form.get('Insulin'))
        bmi = float(request.form.get('BMI'))
        diabetes_predigree_function = float(request.form.get('DiabetesPedigreeFunction'))
        age = float(request.form.get('Age'))

        new_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_predigree_function, age]], 
                                columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        new_data_scaled = standard_scaler.transform(new_data)
        
        prediction = model.predict(new_data_scaled)
        if prediction[0] == 1:
            result = "diabetic"
        else:
            result = "non-diabetic"
        
        return render_template('single_prediction.html', result=result)
        


        

    else:
        return render_template('home.html')
if __name__=="__main__":
    app.run(host="0.0.0.0",port=3000)
