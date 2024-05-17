# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the Random Forest CLassifier model
# filename = 'diabetes-prediction-rfc-model.pkl'
# classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        input_data = (preg,glucose,bp,st,insulin,bmi,dpf,age)
# changing the input_data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

        diabetes_dataset = pd.read_csv('diabetes.csv')
        X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
        Y = diabetes_dataset['Outcome']
        scaler = StandardScaler()

        scaler.fit(X)

        standardized_data = scaler.transform(X)
        X = standardized_data
        Y = diabetes_dataset['Outcome']

        
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

# standardize the input data
        classifier = svm.SVC(kernel='linear')
        classifier.fit(X_train, Y_train)

        X_train_prediction = classifier.predict(X_train)
        training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

        X_test_prediction = classifier.predict(X_test)
        test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
        # scaler = StandardScaler().fit_transform(input_data_reshaped)
        std_data = scaler.transform(input_data_reshaped)
        # print(std_data)

        my_prediction = classifier.predict(std_data)
        # print(prediction)

        # data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        # my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)
        
        return render_template('result.html', prediction=my_prediction)
        
if __name__ == '__main__':
	app.run(debug=True)
