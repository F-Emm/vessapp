from __future__ import absolute_import, division, print_function, unicode_literals
from ast import Try
import os

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
# from six.moves import urllib
import tensorflow as tf
# import tensorflow.compat.v2.feature_column as fc

from flask import Flask, request, render_template, redirect, session, url_for
import pickle
import joblib

app = Flask(__name__)

model = joblib.load("models/model_bi.pkl")

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

# global target_value
# target_value = None

@app.route('/quote',methods=['GET', 'POST'])
def quote():
    # global target_value
    target_value = request.args.get('target')
    session['target_value'] = target_value
    return render_template('/startup2-1.0.0/quote.html', target_value=target_value)

@app.route('/contact',methods=['POST'])
def contact():
    return render_template('/startup2-1.0.0/contact.html')

db_store = {}

output_store = ''

key_tonnage = ''

delivery = ''

type_nm = ''

@app.route('/getprediction',methods=['GET', 'POST'])
def getprediction():
  global db_store
  global output_store
  global key_tonnage
  global delivery
  global type_nm

  type_nm = 'Bitumen'
  # print(f'del = {delivery}')
  


  def input_fn(features, batch_size=256):
    #Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  # print("Please type numeric values as prompted.")

  keys = request.form.keys()
# Use this in hosting:

# Start
  for key in keys:
    print(key, request.form[key])
    print(f"key tonnage = {request.form['Tonnage']}")
    key_tonnage = {request.form['Tonnage']}
    delivery = {request.form['Packed']}
    deliv = "".join(delivery)
    deliv = int(deliv)
    if deliv == 1 :
      delivery = 'Suction'

    try:
      reag = request.form['Tonnage']
      reag = int(reag)
      try:
        if 2000 <= reag <= 3500:
          rea = 1
        elif 3501 <= reag <= 5000:
          rea = 2
        elif 5001 <= reag <= 6500:
          rea = 3
        elif 6501 <= reag <= 8000:
          rea = 4
        elif reag < 2000:
          rea = 0
        else:
          rea = 0
        print(f"remodel {rea}")
      except:
        print(f"remodel passed = {rea}")
    except:
      pass
    for i in request.form[key]:
      predict.setdefault(key, []).append(float(i))
    print(f"predict = {predict}")
    law = request.form['Tonnage']
  try:
    if reag < 2000:
      return render_template('/startup2-1.0.0/output.html', output=f'Value Tonnage = "{law}" is below tonnage values in dataset')
    elif reag >= 8001:
      return render_template('/startup2-1.0.0/output.html', output=f'Value Tonnage = "{law}" is higher than tonnage values in dataset')
    else:
      predict['Tonnage'] = [float(rea)]
      print(f"predict update = {predict}")
      db_store = predict


      predictions = model.predict(input_fn=lambda: input_fn(predict))
      for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS[class_id], 100 * probability))

        print(f"database cycle = {db_store}")
        print(f"output = {DAYS[class_id]}")
        
        output_store = DAYS[class_id]

      # return render_template('/startup2-1.0.0/output.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS[class_id], 100 * probability))
      return render_template('/startup2-1.0.0/output.html', output='Prediction is "{} Days"'.format(DAYS[class_id], 100 * probability))

  except Exception:
    pass
# End

# General Cargo session
@app.route('/getprediction1',methods=['POST'])
def getprediction1():
  global db_store
  global output_store
  global key_tonnage
  global delivery
  global type_nm

  type_nm = 'General Cargo'

  def input_fn(features, batch_size=256):
    #Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  # features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  keys = request.form.keys()

# Start
  for key in keys:
    print(key, request.form[key])
    print(f"key tonnage = {request.form['Tonnage']}")
    key_tonnage = {request.form['Tonnage']}
    delivery = {request.form['Packed']}
    deliv = "".join(delivery)
    deliv = int(deliv)
    if deliv == 1 :
      delivery = 'Bagged'


    try:
      reag = request.form['Tonnage']
      reag = int(reag)
      try:
        if 1000 <= reag <= 2500:
          rea = 1
        elif 2501 <= reag <= 4000:
          rea = 2
        elif 4001 <= reag <= 5500:
          rea = 3
        elif 5501 <= reag <= 7000:
          rea = 4
        elif 7001 <= reag <= 8500:
          rea = 5
        elif 8501 <= reag <= 10000:
          rea = 6
        elif reag < 1000:
          rea = 0
        else:
          rea = 0
        print(f"remodel {rea}")
      except:
        print(f"remodel passed = {rea}")
    except:
      pass
    for i in request.form[key]:
      predict.setdefault(key, []).append(float(i))
    print(f"predict = {predict}")
    law = request.form['Tonnage']
  try:
    if reag < 1000:
      return render_template('/startup2-1.0.0/output.html', output=f'Value Tonnage = "{law}" is below tonnage values in dataset')
    elif reag >= 10001:
      return render_template('/startup2-1.0.0/output.html', output=f'Value Tonnage = "{law}" is higher than tonnage values in dataset')
    else:
      predict['Tonnage'] = [float(rea)]
      print(f"predict update = {predict}")
      db_store = predict



      predictions = model1.predict(input_fn=lambda: input_fn(predict))
      for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS1[class_id], 100 * probability))

      print(f"database cycle = {db_store}")
      print(f"output = {DAYS1[class_id]}")
      output_store = DAYS1[class_id]
      
      # return render_template('/startup2-1.0.0/output.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS1[class_id], 100 * probability))
      return render_template('/startup2-1.0.0/output.html', output='Prediction is "{} Days"'.format(DAYS1[class_id], 100 * probability))


  except Exception:
    pass
# End

# Gypsum session
@app.route('/getprediction2',methods=['POST'])
def getprediction2():
  global db_store
  global output_store
  global key_tonnage
  global delivery
  global type_nm

  type_nm = 'Gypsum'

  def input_fn(features, batch_size=256):
    #Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  # features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  keys = request.form.keys()
  
# Start
  for key in keys:
    print(key, request.form[key])
    print(f"key tonnage = {request.form['Tonnage']}")
    key_tonnage = {request.form['Tonnage']}
    delivery = {request.form['Packed']}
    deliv = "".join(delivery)
    deliv = int(deliv)
    if deliv == 1 :
      delivery = 'Bagged'
    elif deliv == 2:
      delivery = 'Loose'
    else:
      delivery = 'N/A'
    # Gypsum

    try:
      reag = request.form['Tonnage']
      reag = int(reag)
      try:
        if 10000 <= reag <= 12000:
          rea = 1
        elif 12001 <= reag <= 14000:
          rea = 2
        elif 14001 <= reag <= 16000:
          rea = 3
        elif 16001 <= reag <= 18000:
          rea = 4
        elif 18001 <= reag <= 20000:
          rea = 5
        elif 20001 <= reag <= 22000:
          rea = 6
        elif 22001 <= reag <= 24000:
          rea = 7
        elif 24001 <= reag <= 26000:
          rea = 8
        elif 26001 <= reag <= 28000:
          rea = 9
        elif 28001 <= reag <= 30000:
          rea = 10
        elif reag < 10000:
          rea = 0
        else:
          rea = 0
        print(f"remodel {rea}")
      except:
        print(f"remodel passed = {rea}")
    except:
      pass
    for i in request.form[key]:
      predict.setdefault(key, []).append(float(i))
    print(f"predict = {predict}")
    law = request.form['Tonnage']
  try:
    if reag < 10000:
      return render_template('/startup2-1.0.0/output.html', output=f'Value Tonnage = "{law}" is below tonnage values in dataset')
    elif reag >= 30001:
      return render_template('/startup2-1.0.0/output.html', output=f'Value Tonnage = "{law}" is higher than tonnage values in dataset')
    else:
      predict['Tonnage'] = [float(rea)]
      print(f"predict update = {predict}")
      db_store = predict

      predictions = model2.predict(input_fn=lambda: input_fn(predict))
      for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS2[class_id], 100 * probability))

      print(f"database cycle = {db_store}")
      print(f"output = {DAYS2[class_id]}")
      output_store = DAYS2[class_id]

      # return render_template('/startup2-1.0.0/output.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS2[class_id], 100 * probability))
      return render_template('/startup2-1.0.0/output.html', output='Prediction is "{} Days"'.format(DAYS2[class_id], 100 * probability))

  except Exception:
    pass
# End

# Malt session
@app.route('/getprediction3',methods=['POST'])
def getprediction3():
  global db_store
  global output_store
  global key_tonnage
  global delivery
  global type_nm

  type_nm = 'Malt'


  def input_fn(features, batch_size=256):
    #Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  # features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  keys = request.form.keys()
  
# Start
  for key in keys:
    print(key, request.form[key])
    print(f"key tonnage = {request.form['Tonnage']}")
    key_tonnage = {request.form['Tonnage']}
    delivery = {request.form['Packed']}
    deliv = "".join(delivery)
    deliv = int(deliv)
    if deliv == 1 :
      delivery = 'Haulage'
    elif deliv == 2:
      delivery = 'Conveyor'
    else:
      delivery = 'N/A'

    try:
      reag = request.form['Tonnage']
      reag = int(reag)
      try:
        if 10000 <= reag <= 12000:
          rea = 1
        elif 12001 <= reag <= 14000:
          rea = 2
        elif 14001 <= reag <= 16000:
          rea = 3
        elif 16001 <= reag <= 18000:
          rea = 4
        elif 18001 <= reag <= 20000:
          rea = 5
        elif 20001 <= reag <= 22000:
          rea = 6
        elif 22001 <= reag <= 24000:
          rea = 7
        elif 24001 <= reag <= 26000:
          rea = 8
        elif 26001 <= reag <= 28000:
          rea = 9
        elif 28001 <= reag <= 30000:
          rea = 10
        elif reag < 10000:
          rea = 0
        else:
          rea = 0
        print(f"remodel {rea}")
      except:
        print(f"remodel passed = {rea}")
    except:
      pass
    for i in request.form[key]:
      predict.setdefault(key, []).append(float(i))
    print(f"predict = {predict}")
    law = request.form['Tonnage']
  try:
    if reag < 10000:
      return render_template('/startup2-1.0.0/output.html', output=f'Value Tonnage = "{law}" is below tonnage values in dataset')
    elif reag >= 30001:
      return render_template('/startup2-1.0.0/output.html', output=f'Value Tonnage = "{law}" is higher than tonnage values in dataset')
    else:
      predict['Tonnage'] = [float(rea)]
      print(f"predict update = {predict}")
      db_store = predict

      predictions = model3.predict(input_fn=lambda: input_fn(predict))
      for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS3[class_id], 100 * probability))

      print(f"database cycle = {db_store}")
      print(f"output = {DAYS3[class_id]}")
      output_store = DAYS3[class_id]

      # return render_template('/startup2-1.0.0/output.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS3[class_id], 100 * probability))
      return render_template('/startup2-1.0.0/output.html', output='Prediction is "{} Days"'.format(DAYS3[class_id], 100 * probability))

  except Exception:
    pass
# End


# Malt & General Cargo session
@app.route('/getprediction4',methods=['POST'])
def getprediction4():
  global db_store
  global output_store
  global key_tonnage
  global delivery
  global type_nm

  type_nm = 'Malt & General Cargo'
  

  def input_fn(features, batch_size=256):
    #Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  # features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  keys = request.form.keys()
  
# Start
  for key in keys:
    print(key, request.form[key])
    print(f"key tonnage = {request.form['Tonnage']}")
    key_tonnage = {request.form['Tonnage']}
    delivery = {request.form['Packed']}
    deliv = "".join(delivery)
    deliv = int(deliv)
    if deliv == 1 :
      delivery = 'Haulage'
    elif deliv == 2:
      delivery = 'Conveyor'
    else:
      delivery = 'N/A'

    try:
      reag = request.form['Tonnage']
      reag = int(reag)
      try:
        if 10000 <= reag <= 12000:
          rea = 1
        elif 12001 <= reag <= 14000:
          rea = 2
        elif 14001 <= reag <= 16000:
          rea = 3
        elif 16001 <= reag <= 18000:
          rea = 4
        elif 18001 <= reag <= 20000:
          rea = 5
        elif 20001 <= reag <= 22000:
          rea = 6
        elif 22001 <= reag <= 24000:
          rea = 7
        elif 24001 <= reag <= 26000:
          rea = 8
        elif 26001 <= reag <= 28000:
          rea = 9
        elif 28001 <= reag <= 30000:
          rea = 10
        elif reag < 10000:
          rea = 0
        else:
          rea = 0
        print(f"remodel {rea}")
      except:
        print(f"remodel passed = {rea}")
    except:
      pass
    for i in request.form[key]:
      predict.setdefault(key, []).append(float(i))
    print(f"predict = {predict}")
    law = request.form['Tonnage']
  try:
    if reag < 10000:
      return render_template('/startup2-1.0.0/output.html', output=f'Value Tonnage = "{law}" is below tonnage values in dataset')
    elif reag >= 30001:
      return render_template('/startup2-1.0.0/output.html', output=f'Value Tonnage = "{law}" is higher than tonnage values in dataset')
    else:
      predict['Tonnage'] = [float(rea)]
      print(f"predict update = {predict}")
      db_store = predict

      predictions = model4.predict(input_fn=lambda: input_fn(predict))
      for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS4[class_id], 100 * probability))

      print(f"database cycle = {db_store}")
      print(f"output = {DAYS4[class_id]}")
      output_store = DAYS4[class_id]

      # return render_template('/startup2-1.0.0/output.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS4[class_id], 100 * probability))
      return render_template('/startup2-1.0.0/output.html', output='Prediction is "{} Days"'.format(DAYS4[class_id], 100 * probability))

  except Exception:
    pass
# End

# Salt session
@app.route('/getprediction5',methods=['POST'])
def getprediction5():
  global db_store
  global output_store
  global key_tonnage
  global delivery
  global type_nm

  type_nm = 'Salt'


  def input_fn(features, batch_size=256):
    #Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  # features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  keys = request.form.keys()
  
# Start
  for key in keys:
    print(key, request.form[key])
    print(f"key tonnage = {request.form['Tonnage']}")
    key_tonnage = {request.form['Tonnage']}
    delivery = {request.form['Packed']}
    deliv = "".join(delivery)
    deliv = int(deliv)
    if deliv == 1 :
      delivery = 'Haulage'
    elif deliv == 2:
      delivery = 'Conveyor'
    else:
      delivery = 'N/A'

    try:
      reag = request.form['Tonnage']
      reag = int(reag)
      try:
        if 10000 <= reag <= 12000:
          rea = 1
        elif 12001 <= reag <= 14000:
          rea = 2
        elif 14001 <= reag <= 16000:
          rea = 3
        elif 16001 <= reag <= 18000:
          rea = 4
        elif 18001 <= reag <= 20000:
          rea = 5
        elif 20001 <= reag <= 22000:
          rea = 6
        elif 22001 <= reag <= 24000:
          rea = 7
        elif 24001 <= reag <= 26000:
          rea = 8
        elif 26001 <= reag <= 28000:
          rea = 9
        elif 28001 <= reag <= 30000:
          rea = 10
        elif reag < 10000:
          rea = 0
        else:
          rea = 0
        print(f"remodel {rea}")
      except:
        print(f"remodel passed = {rea}")
    except:
      pass
    for i in request.form[key]:
      predict.setdefault(key, []).append(float(i))
    print(f"predict = {predict}")
    law = request.form['Tonnage']
  try:
    if reag < 10000:
      return render_template('/startup2-1.0.0/output.html', output=f'Value Tonnage = "{law}" is below tonnage values in dataset')
    elif reag >= 30001:
      return render_template('/startup2-1.0.0/output.html', output=f'Value Tonnage = "{law}" is higher than tonnage values in dataset')
    else:
      predict['Tonnage'] = [float(rea)]
      print(f"predict update = {predict}")
      db_store = predict

      predictions = model5.predict(input_fn=lambda: input_fn(predict))
      for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS5[class_id], 100 * probability))

      print(f"database cycle = {db_store}")
      print(f"output = {DAYS5[class_id]}")
      output_store = DAYS5[class_id]
      
      # return render_template('/startup2-1.0.0/output.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS5[class_id], 100 * probability))
      return render_template('/startup2-1.0.0/output.html', output='Prediction is "{} Days"'.format(DAYS5[class_id], 100 * probability))

  except Exception:
    pass
# End

# Wheat session
@app.route('/getprediction6',methods=['POST'])
def getprediction6():
  global db_store
  global output_store
  global key_tonnage
  global delivery
  global type_nm

  type_nm = 'Wheat'


  def input_fn(features, batch_size=256):
    #Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  # features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  keys = request.form.keys()
  
# Start
  for key in keys:
    print(key, request.form[key])
    print(f"key tonnage = {request.form['Tonnage']}")
    key_tonnage = {request.form['Tonnage']}
    delivery = {request.form['Packed']}
    deliv = "".join(delivery)
    deliv = int(deliv)
    if deliv == 1 :
      delivery = 'Bagged'
    elif deliv == 2:
      delivery = 'Loose'
    else:
      delivery = 'N/A'

    try:
      reag = request.form['Tonnage']
      reag = int(reag)
      try:
        if 10000 <= reag <= 12000:
          rea = 1
        elif 12001 <= reag <= 14000:
          rea = 2
        elif 14001 <= reag <= 16000:
          rea = 3
        elif 16001 <= reag <= 18000:
          rea = 4
        elif 18001 <= reag <= 20000:
          rea = 5
        elif 20001 <= reag <= 22000:
          rea = 6
        elif 22001 <= reag <= 24000:
          rea = 7
        elif 24001 <= reag <= 26000:
          rea = 8
        elif 26001 <= reag <= 28000:
          rea = 9
        elif 28001 <= reag <= 30000:
          rea = 10
        elif reag < 10000:
          rea = 0
        else:
          rea = 0
        print(f"remodel {rea}")
      except:
        print(f"remodel passed = {rea}")
    except:
      pass
    for i in request.form[key]:
      predict.setdefault(key, []).append(float(i))
    print(f"predict = {predict}")
    law = request.form['Tonnage']
  try:
    if reag < 10000:
      return render_template('/startup2-1.0.0/output.html', output=f'Value Tonnage = "{law}" is below tonnage values in dataset')
    elif reag >= 30001:
      return render_template('/startup2-1.0.0/output.html', output=f'Value Tonnage = "{law}" is higher than tonnage values in dataset')
    else:
      predict['Tonnage'] = [float(rea)]
      print(f"predict update = {predict}")
      db_store = predict

      predictions = model6.predict(input_fn=lambda: input_fn(predict))
      for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS6[class_id], 100 * probability))

      print(f"database cycle = {db_store}")
      print(f"output = {DAYS6[class_id]}")
      output_store = DAYS6[class_id]

      # return render_template('/startup2-1.0.0/output.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS6[class_id], 100 * probability))
      return render_template('/startup2-1.0.0/output.html', output='Prediction is "{} Days"'.format(DAYS6[class_id], 100 * probability))

  except Exception:
    pass
# End

# Zinc session
@app.route('/getprediction7',methods=['POST'])
def getprediction7():
  global db_store
  global output_store
  global key_tonnage
  global delivery
  global type_nm

  type_nm = 'Zinc'


  def input_fn(features, batch_size=256):
    #Convert the inputs to a Dataset without labels.
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

    # features = ['Agent', 'receiver', 'Type', 'Month', 'Tonnage', 'LOA', 'Draft', 'Berth']
	
  # features = ['crew_Motivation', 'vessel_Condtion', 'consignee_throughput', 'weather', 'Tonnage', 'Packed'] 
  predict = {}

  keys = request.form.keys()
  
# Start
  for key in keys:
    print(key, request.form[key])
    print(f"key tonnage = {request.form['Tonnage']}")
    key_tonnage = {request.form['Tonnage']}
    delivery = {request.form['Packed']}
    deliv = "".join(delivery)
    deliv = int(deliv)
    if deliv == 1 :
      delivery = 'Haulage'
    elif deliv == 2:
      delivery = 'Conveyor'
    else:
      delivery = 'N/A'

    try:
      reag = request.form['Tonnage']
      reag = int(reag)
      try:
        if 10000 <= reag <= 12000:
          rea = 1
        elif 12001 <= reag <= 14000:
          rea = 2
        elif 14001 <= reag <= 16000:
          rea = 3
        elif 16001 <= reag <= 18000:
          rea = 4
        elif 18001 <= reag <= 20000:
          rea = 5
        elif 20001 <= reag <= 22000:
          rea = 6
        elif 22001 <= reag <= 24000:
          rea = 7
        elif 24001 <= reag <= 26000:
          rea = 8
        elif 26001 <= reag <= 28000:
          rea = 9
        elif 28001 <= reag <= 30000:
          rea = 10
        elif reag < 10000:
          rea = 0
        else:
          rea = 0
        print(f"remodel {rea}")
      except:
        print(f"remodel passed = {rea}")
    except:
      pass
    for i in request.form[key]:
      predict.setdefault(key, []).append(float(i))
    print(f"predict = {predict}")
    law = request.form['Tonnage']
  try:
    if reag < 10000:
      return render_template('/startup2-1.0.0/output.html', output=f'Value Tonnage = "{law}" is below tonnage values in dataset')
    elif reag >= 30001:
      return render_template('/startup2-1.0.0/output.html', output=f'Value Tonnage = "{law}" is higher than tonnage values in dataset')
    else:
      predict['Tonnage'] = [float(rea)]
      print(f"predict update = {predict}")
      db_store = predict


      predictions = model7.predict(input_fn=lambda: input_fn(predict))
      for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS7[class_id], 100 * probability))

      print(f"database cycle = {db_store}")
      print(f"output = {DAYS7[class_id]}")
      output_store = DAYS7[class_id]

      # return render_template('/startup2-1.0.0/output.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS7[class_id], 100 * probability))
      return render_template('/startup2-1.0.0/output.html', output='Prediction is "{} Days"'.format(DAYS7[class_id], 100 * probability))

  except Exception:
    pass
# End

def get_db():
  return db_store

def get_output():
  return output_store

def get_toonage():
  return key_tonnage

# def get_target():
#   return target_value

@app.route('/output',methods=['GET', 'POST'])
def output():

  db_cycle = db_store
  output_cycle = output_store
  true_tonnage = key_tonnage

  c1 = db_cycle['crew_Motivation']

  v1 = db_cycle['vessel_Condtion']

  co1 = db_cycle['consignee_throughput']

  w1 = db_cycle['weather']

  # t1 = db_cycle['Tonnage']

  pca1 = db_cycle['Packed'] #aka delivery
  pca1 = delivery

  c_tp = type_nm

  tar_val = session.get('target_value')

  print(f'del = {pca1}')

  t1 = "".join(true_tonnage)

  print(f' true_ton = {t1}')

  if c1 == 0:
    c1 = 'Low'
  else:
    c1 = 'High'
  print(f'c1 = {c1}')

  if v1 == 0:
    v1 = 'Bad'
  else:
    v1 = 'Good'

  if co1 == 0:
    co1 = 'Bad'
  else:
    co1 = 'Good'

  if w1 == 0:
    w1 = 'Dry'
  else:
    w1 = 'Rainny'

  print(f"db_cycle = {db_cycle}")
  print(f"output_cycle = {output_cycle}")
  direct_to = f'https://ushvesapp.ushmoney.net/dashboard.php?crew_motivation={c1}&vessel_condition={v1}&consignee_throughput={co1}&weather_condition={w1}&tonnage={t1}&delivery={pca1}&output={output_cycle}&terminal=__&cargo_Type={c_tp}&target={tar_val}'

  return redirect(direct_to)

if __name__ == "__main__":
  port = os.environ.get("PORT", 5000)
  app.run(debug=False, host="0.0.0.0", port=port)
