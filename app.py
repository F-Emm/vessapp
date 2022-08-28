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
  #print(f"here it is = {keys}")

# Use this in hosting:

# Start
  for key in keys:
    print(key, request.form[key])
    print(f"key tonnage = {request.form['Tonnage']}")

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
      return render_template('/startup2-1.0.0/quote.html', output=f'Value Tonnage = "{law}" is below tonnage values in dataset')
    elif reag >= 8001:
      return render_template('/startup2-1.0.0/quote.html', output=f'Value Tonnage = "{law}" is higher than tonnage values in dataset')
    else:
      predict['Tonnage'] = [float(rea)]
      print(f"predict update = {predict}")

      predictions = model.predict(input_fn=lambda: input_fn(predict))
      for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS[class_id], 100 * probability))

      return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS[class_id], 100 * probability))

  except Exception:
    pass

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

# Start
  for key in keys:
    print(key, request.form[key])
    print(f"key tonnage = {request.form['Tonnage']}")

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
      return render_template('/startup2-1.0.0/quote.html', output=f'Value Tonnage = "{law}" is below tonnage values in dataset')
    elif reag >= 10001:
      return render_template('/startup2-1.0.0/quote.html', output=f'Value Tonnage = "{law}" is higher than tonnage values in dataset')
    else:
      predict['Tonnage'] = [float(rea)]
      print(f"predict update = {predict}")

      predictions = model1.predict(input_fn=lambda: input_fn(predict))
      for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS1[class_id], 100 * probability))

      return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS1[class_id], 100 * probability))

  except Exception:
    pass
# End

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
  
# Start
  for key in keys:
    print(key, request.form[key])
    print(f"key tonnage = {request.form['Tonnage']}")

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
      return render_template('/startup2-1.0.0/quote.html', output=f'Value Tonnage = "{law}" is below tonnage values in dataset')
    elif reag >= 30001:
      return render_template('/startup2-1.0.0/quote.html', output=f'Value Tonnage = "{law}" is higher than tonnage values in dataset')
    else:
      predict['Tonnage'] = [float(rea)]
      print(f"predict update = {predict}")

      predictions = model2.predict(input_fn=lambda: input_fn(predict))
      for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS2[class_id], 100 * probability))

      return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS2[class_id], 100 * probability))

  except Exception:
    pass
# End

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
  
# Start
  for key in keys:
    print(key, request.form[key])
    print(f"key tonnage = {request.form['Tonnage']}")

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
      return render_template('/startup2-1.0.0/quote.html', output=f'Value Tonnage = "{law}" is below tonnage values in dataset')
    elif reag >= 30001:
      return render_template('/startup2-1.0.0/quote.html', output=f'Value Tonnage = "{law}" is higher than tonnage values in dataset')
    else:
      predict['Tonnage'] = [float(rea)]
      print(f"predict update = {predict}")

      predictions = model3.predict(input_fn=lambda: input_fn(predict))
      for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS3[class_id], 100 * probability))

      return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS3[class_id], 100 * probability))

  except Exception:
    pass
# End


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
  
# Start
  for key in keys:
    print(key, request.form[key])
    print(f"key tonnage = {request.form['Tonnage']}")

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
      return render_template('/startup2-1.0.0/quote.html', output=f'Value Tonnage = "{law}" is below tonnage values in dataset')
    elif reag >= 30001:
      return render_template('/startup2-1.0.0/quote.html', output=f'Value Tonnage = "{law}" is higher than tonnage values in dataset')
    else:
      predict['Tonnage'] = [float(rea)]
      print(f"predict update = {predict}")

      predictions = model4.predict(input_fn=lambda: input_fn(predict))
      for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS4[class_id], 100 * probability))

      return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS4[class_id], 100 * probability))

  except Exception:
    pass
# End

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
  
# Start
  for key in keys:
    print(key, request.form[key])
    print(f"key tonnage = {request.form['Tonnage']}")

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
      return render_template('/startup2-1.0.0/quote.html', output=f'Value Tonnage = "{law}" is below tonnage values in dataset')
    elif reag >= 30001:
      return render_template('/startup2-1.0.0/quote.html', output=f'Value Tonnage = "{law}" is higher than tonnage values in dataset')
    else:
      predict['Tonnage'] = [float(rea)]
      print(f"predict update = {predict}")

      predictions = model5.predict(input_fn=lambda: input_fn(predict))
      for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS5[class_id], 100 * probability))

      return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS5[class_id], 100 * probability))

  except Exception:
    pass
# End

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
  
# Start
  for key in keys:
    print(key, request.form[key])
    print(f"key tonnage = {request.form['Tonnage']}")

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
      return render_template('/startup2-1.0.0/quote.html', output=f'Value Tonnage = "{law}" is below tonnage values in dataset')
    elif reag >= 30001:
      return render_template('/startup2-1.0.0/quote.html', output=f'Value Tonnage = "{law}" is higher than tonnage values in dataset')
    else:
      predict['Tonnage'] = [float(rea)]
      print(f"predict update = {predict}")

      predictions = model6.predict(input_fn=lambda: input_fn(predict))
      for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS6[class_id], 100 * probability))

      return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS6[class_id], 100 * probability))

  except Exception:
    pass
# End

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
  
# Start
  for key in keys:
    print(key, request.form[key])
    print(f"key tonnage = {request.form['Tonnage']}")

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
      return render_template('/startup2-1.0.0/quote.html', output=f'Value Tonnage = "{law}" is below tonnage values in dataset')
    elif reag >= 30001:
      return render_template('/startup2-1.0.0/quote.html', output=f'Value Tonnage = "{law}" is higher than tonnage values in dataset')
    else:
      predict['Tonnage'] = [float(rea)]
      print(f"predict update = {predict}")

      predictions = model7.predict(input_fn=lambda: input_fn(predict))
      for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print('Prediction is "{} Days" ({:.1f}%)'.format(DAYS7[class_id], 100 * probability))

      return render_template('/startup2-1.0.0/quote.html', output='Prediction is "{} Days" ({:.1f}%)'.format(DAYS7[class_id], 100 * probability))

  except Exception:
    pass
# End

# app.run(debug=True)
if __name__ == "__main__":
#   app.run(host="0.0.0.0", port=8080)
#   from waitress import serve
#   serve(app, host="0.0.0.0", port=5000)
   app.run(host="0.0.0.0", debug=True)
