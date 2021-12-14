#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 22:13:41 2021

@author: shashankbhatnagar
"""

import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle
import pandas as pd
import statsmodels.api as stats

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

scaler = pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictor',methods=['POST'])
def predictor():
    
    """
    For rendering results on GUI
    
    """
    print("inside")
    
    features = [x for x in request.form.values()]
    
    sql_data = pd.DataFrame()

    sql_data['yr'] = [1] if features[3] == '2019' else [0]
    sql_data['workingday'] = [0] if features[1] == 'NO' else [1]
    sql_data['temp'] = [float(features[2])]
    sql_data['mnth_mar'] = [1] if features[6] == 'MAR' else [0]
    sql_data['mnth_may'] = [1] if features[6] == 'MAY' else [0]
    sql_data['mnth_sept'] = [1] if features[6] == 'SEP' else [0]
    sql_data['season_spring'] = [1] if features[0] == 'SPRING' else [0]
    sql_data['season_winter'] = [1] if features[0] == 'WINTER' else [0]
    sql_data['weathersit_Mist'] = [1] if features[4] == 'MIST' else [0]
    sql_data['windspeed'] = [float(features[7])]
    sql_data['hum'] = [float(features[5])]
    sql_data['atemp'] = [13.12]
    
    scaled_list = ['temp','atemp','hum','windspeed']
    
    sql_data[scaled_list] = scaler.transform(sql_data[scaled_list])
    
    sql_data = sql_data[['yr', 'workingday', 'temp', 'mnth_mar', 'mnth_may', 'mnth_sept',
       'season_spring', 'season_winter', 'weathersit_Mist', 'windspeed',
       'hum']]
    
    sql_data_sm = stats.add_constant(sql_data, has_constant='add')

    sql_data_pred = model.predict(sql_data_sm)

    output = round(sql_data_pred[0])
    
    return render_template('index.html',prediction_text=output)


if __name__ == "__main__":
    app.run(debug=True)