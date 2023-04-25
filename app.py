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


@app.route('/')
def home():
    return "Hello world I did it"

#@app.route('/predict_api',methods=['POST'])
#def predict_api():
    #data=request.json['data']
    #print(data)
    #print(np.array(list(data.values())).reshape(1,-1))
   # data1 = request.form['a']
   # data2 = request.form['b']
  #  data3 = request.form['c']
   # arr = np.array([[data1, data2, data3]])
   # output = dtmodel.predict(arr)
   # return render_template('after.html',data=output)

if __name__=="__main__":
    app.run(debug=True)



