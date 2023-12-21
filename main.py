from flask import Flask , render_template , request
import pickle
import numpy as np

transformer = pickle.load(open('transformer_for_sentiment.pkl' , 'rb'))


app = Flask(__name__)

@app.route("/")
def first():
    return render_template("index.html")

@app.route("/predict" ,methods = ["POST"])
def predict():
    string = request.form.get('sentiment')
    result = transformer([string])
    result = str(result[0]['label']).upper() + " : " + str(np.round(result[0]['score'] , 2))
    return render_template('index.html' , result = result)


if "__main__" == __name__:
    app.run(debug = True)