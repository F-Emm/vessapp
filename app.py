from __future__ import absolute_import, division, print_function, unicode_literals
import os

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
# from six.moves import urllib
import tensorflow as tf
# import tensorflow.compat.v2.feature_column as fc

from flask import Flask, request, render_template
import pickle
import joblib

app = Flask(__name__)

#run_with_ngrok(app)
model = joblib.load("models/model_bi.pkl")
# daysbi.pkl
# Ush Engineering\Projects\hero_project\Vessel_Prediction\

model1 = joblib.load("models/model_gc.pkl")

model2 = joblib.load("models/model_gp.pkl")

model3 = joblib.load("models/model_mt.pkl")

model4 = joblib.load("models/model_mtgc.pkl")

model5 = joblib.load("models/model_st.pkl")

model6 = joblib.load("models/model_wt.pkl")

model7 = joblib.load("models/model_zc.pkl")

DAYS = joblib.load("models/daysbi.pkl")

DAYS1 = joblib.load("models/daysgc.pkl")

DAYS2 = joblib.load("models/daysgp.pkl")

DAYS3 = joblib.load("models/daysmt.pkl")

DAYS4 = joblib.load("models/daysmtgc.pkl")

DAYS5 = joblib.load("models/daysst.pkl")

DAYS6 = joblib.load("models/dayswt.pkl")

DAYS7 = joblib.load("models/dayszc.pkl")

@app.route('/',methods=['GET', 'POST'])
def home():
    return render_template('/startup2-1.0.0/iindex.html')
    # return redirect(url_for('/startup2-1.0.0/iindex.html'))

@app.route('/quote',methods=['GET', 'POST'])
def quote():
    return render_template('/startup2-1.0.0/quote.html')

@app.route('/contact',methods=['POST'])
def contact():
    return render_template('/startup2-1.0.0/contact.html')

@app.route('/getprediction',methods=['GET', 'POST'])
def getprediction():    

  def input_fn(features, batch_size=256):
    #Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  # print("Please type numeric values as prompted.")

  keys = request.form.keys()
  # keys = sorted(keys)

  for key in keys:
    print(key, request.form[key])
    for i in request.form[key]:
      predict.setdefault(key, []).append(float(i))
      print(predict)
    # print(key, request.form.get(key))

  predictions = model.predict(input_fn=lambda: input_fn(predict))
  for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    # print('Prediction is "{} Days" ({:.1f}%)'.format(
    #     DAYS[class_id], 100 * probability))

  return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS[class_id], 100 * probability))
  # return redirect(url_for('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS[class_id], 100 * probability))

# General Cargo session
@app.route('/getprediction1',methods=['POST'])
def getprediction1():    

  def input_fn(features, batch_size=256):
    #Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  # features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  keys = request.form.keys()

  for key in keys:
    print(key, request.form[key])
    for i in request.form[key]:
      predict.setdefault(key, []).append(float(i))
      print(predict)

  predictions = model1.predict(input_fn=lambda: input_fn(predict))
  for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS1[class_id], 100 * probability))

  return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS1[class_id], 100 * probability))

# Gypsum session
@app.route('/getprediction2',methods=['POST'])
def getprediction2():    

  def input_fn(features, batch_size=256):
    #Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  # features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  keys = request.form.keys()
  
  for key in keys:
    print(key, request.form[key])
    for i in request.form[key]:
      predict.setdefault(key, []).append(float(i))
      print(predict)

  predictions = model2.predict(input_fn=lambda: input_fn(predict))
  for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS2[class_id], 100 * probability))

  return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS2[class_id], 100 * probability))

# Malt session
@app.route('/getprediction3',methods=['POST'])
def getprediction3():    

  def input_fn(features, batch_size=256):
    #Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  # features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  keys = request.form.keys()
  
  for key in keys:
    print(key, request.form[key])
    for i in request.form[key]:
      predict.setdefault(key, []).append(float(i))
      print(predict)

  predictions = model3.predict(input_fn=lambda: input_fn(predict))
  for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS3[class_id], 100 * probability))

  return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS3[class_id], 100 * probability))

# Malt & General Cargo session
@app.route('/getprediction4',methods=['POST'])
def getprediction4():    

  def input_fn(features, batch_size=256):
    #Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  # features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  keys = request.form.keys()
  
  for key in keys:
    print(key, request.form[key])
    for i in request.form[key]:
      predict.setdefault(key, []).append(float(i))
      print(predict)

  predictions = model4.predict(input_fn=lambda: input_fn(predict))
  for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS4[class_id], 100 * probability))

  return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS4[class_id], 100 * probability))

# Salt session
@app.route('/getprediction5',methods=['POST'])
def getprediction5():    

  def input_fn(features, batch_size=256):
    #Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  # features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  keys = request.form.keys()
  
  for key in keys:
    print(key, request.form[key])
    for i in request.form[key]:
      predict.setdefault(key, []).append(float(i))
      print(predict)

  predictions = model5.predict(input_fn=lambda: input_fn(predict))
  for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS5[class_id], 100 * probability))

  return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS5[class_id], 100 * probability))

# Wheat session
@app.route('/getprediction6',methods=['POST'])
def getprediction6():    

  def input_fn(features, batch_size=256):
    #Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  # features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  keys = request.form.keys()
  
  for key in keys:
    print(key, request.form[key])
    for i in request.form[key]:
      predict.setdefault(key, []).append(float(i))
      print(predict)

  predictions = model6.predict(input_fn=lambda: input_fn(predict))
  for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS6[class_id], 100 * probability))

  return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS6[class_id], 100 * probability))

# Zinc session
@app.route('/getprediction7',methods=['POST'])
def getprediction7():    

  def input_fn(features, batch_size=256):
    #Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  # features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  keys = request.form.keys()
  
  for key in keys:
    print(key, request.form[key])
    for i in request.form[key]:
      predict.setdefault(key, []).append(float(i))
      print(predict)

  predictions = model7.predict(input_fn=lambda: input_fn(predict))
  for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS7[class_id], 100 * probability))

  # return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS7[class_id], 100 * probability))
  return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" '.format(DAYS7[class_id]))

# app.run(debug=True)
if __name__ == "__main__":
  # app.run(host="0.0.0.0", port=8080)
  # from waitress import serve
  # serve(app, host="0.0.0.0", port=8080)
  app.run(host="0.0.0.0", debug=True)
