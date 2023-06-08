import flask
import json
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
#,redirect,flash,session,escape
import numpy as np
import pandas as pd

print("Hello")
app=Flask(__name__)
## load the model
dtmodel=pickle.load(open('completeDT.pkl','rb'))
lrmodel = pickle.load(open('logisticRegression.pkl','rb'))
knnmodel = pickle.load(open('knnmodel.pkl','rb'))
svmmodel = pickle.load(open('svmmodel.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=np.array(list(data.values())).reshape(1,-1)
    output=dtmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

#Get method for Decision Tree
@app.route('/DT-GET')
def DTGET():
    return render_template("decisionTree.html")

#Get method for Logistic Regression
@app.route('/LR-GET')
def LRGET():
    return render_template("logisticRegression.html")

#Get method for SVM
@app.route('/SVM-GET')
def SVMGET():
    return render_template("SVM.html")

#Get method for KNN
@app.route('/KNN-GET')
def KNNGET():
    return render_template("KNN.html")


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=np.array(data).reshape(1,-1)
    print(final_input)
    output=dtmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The classification prediction for choosen model is {}".format(output))

@app.route('/predictLR',methods=['POST'])
def predictLR():
    data=[float(x) for x in request.form.values()]
    final_input=np.array(data).reshape(1,-1)
    print(final_input)
    output=lrmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The Logistic Regression model prediction for choosen input is {}".format(output))

@app.route('/predictKNN',methods=['POST'])
def predictKNN():
    data=[float(x) for x in request.form.values()]
    final_input=np.array(data).reshape(1,-1)
    print(final_input)
    output=knnmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The KNN model prediction for choosen input is {}".format(output))

@app.route('/predictSVM',methods=['POST'])
def predictSVM():
    data=[float(x) for x in request.form.values()]
    final_input=np.array(data).reshape(1,-1)
    print(final_input)
    output=svmmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The SVM model prediction for choosen input is {}".format(output))    
   
if __name__=="__main__":
    app.run(debug=True)



