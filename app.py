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

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=np.array(data).reshape(1,-1)
    print(final_input)
    output=dtmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))

    
    #data = [float(x) for x in request.form.values()]
    #data1 = request.form['AGE']
    #data2 = request.form['JOB']
    #data3 = request.form['MARITAL']
    #data4 = request.form['EDUCATION']
    #data5 = request.form['DEFAULT']
    # data6 = request.form['BALANCE']
    # data7 = request.form['HOUSING']
    # data8 = request.form['LOAN']
    # data9 = request.form['CONTACT']
    # data10 = request.form['DAY']
    # data11 = request.form['MONTH']
    # data12 = request.form['DURATION']
    # data13 = request.form['CAMPAIGN']
    # data14 = request.form['PDAYS']
    # data15 = request.form['PREVIOUS']
    # data16 = request.form['POUTCOME']
    # arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15, data16]]).reshape(1,-1)
    #arr = np.array(data).reshape(1,-1)
    # print(arr)
    # output = dtmodel.predict(arr)
    # render_template("home.html",prediction_text="The output of data is {}".format(output))
   # return render_template('after.html',data=output)

if __name__=="__main__":
    app.run(debug=True)



