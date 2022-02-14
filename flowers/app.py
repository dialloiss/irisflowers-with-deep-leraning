import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf 
from keras.models import load_model
import pandas as pd



app = Flask(__name__)
model = load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    SepalLength=request.form['Sepal.Length']

    Sepalwidth=request.form['Sepal.Width']

    PetalLength =request.form['Petal.Length']

    Petalwidth=request.form['Petal.Width']

    dataframe = pd.DataFrame({"SepalLength":[float(SepalLength)],
                 "Sepalwidth":[float(Sepalwidth)],
                 "PetalLength":[float(PetalLength)],
                 "Petalwidth":[float(Petalwidth)]})
    dataframevalues=dataframe.values

    tensordata=tf.cast(dataframevalues,tf.float32)

    np.reshape(tensordata,(4,1))

    prediction=model.predict(tensordata)

    output=np.argmax(prediction,axis=1)

    if output==[0]:
        return render_template('index.html', prediction_text='the flower is setosa')
    if output==[1]:
        return render_template('index.html', prediction_text='the flower is versicolor')
    if output==[2]:
        return render_template('index.html', prediction_text='the flower is virginica')

if __name__ == "__main__":
    app.run(debug=True)     