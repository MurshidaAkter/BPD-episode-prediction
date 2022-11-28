import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def hello_world():
    return render_template("htmlPred.html")

@app.route('/note.html')
def hello_world1():
    return (render_template("note.html"))

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    test= [np.array(int_features)]

    prediction = model.predict(test)[0]
    print('predict=',prediction)

    if prediction == 0:
        return render_template('htmlPred.html', pred='Your state is Normal',
                               hue="Clear hue")
    if prediction == 1:
            return render_template('htmlPred.html', pred='High chance of having Depression! You need consulation.',
                                   hue="Clouded hue")
    if prediction == 2:
            return render_template('htmlPred.html', pred='High chance of having Manic! You need consulation.',
                                   hue="Clouded hue")


#dataset = dataset.replace(to_replace="N", value=0)
#dataset = dataset.replace(to_replace="D", value=1)
#dataset = dataset.replace(to_replace="M", value=2)

if __name__ == '__main__':
    app.run(debug=True)
