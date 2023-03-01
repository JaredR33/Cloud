from flask import Flask, request, jsonify
import os
import pickle
from sklearn.model_selection import cross_val_score
import pandas as pd
import sqlite3
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import datetime



os.chdir(os.path.dirname(__file__))

os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido a mi API del modelo advertising de Jaredorito"

# 1. Endpoint que devuelva la predicción de los nuevos datos enviados mediante argumentos en la llamada
@app.route('/v1/predict', methods=['GET'])
def predict():
    model = pickle.load(open('data/advertising_model','rb'))

    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    if tv is None or radio is None or newspaper is None:
        return "Missing args, the input values are needed to predict"
    else:
        prediction = model.predict([[tv,radio,newspaper]])
        return "The prediction of sales investing that amount of money in TV, radio and newspaper is: " + str(round(prediction[0],2)) + 'k €'

#2 Un endpoint para almacenar nuevos registros en la base de datos que deberá estar previamente creada.(/v2/ingest_data)

@app.route('/v2/ingest_data', methods=['POST'])
def ingest_data():
    # Obtener los datos de la solicitud
    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)
    sales = request.args.get('sales', None)


    # Establecer una conexión con la base de datos y crear un cursor
    conn = sqlite3.connect('data/advertising2.db')
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS campanias
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                   TV REAL,
                   radio REAL,
                   newspaper REAL,
                   sales REAL);''')
    
    # Insertar los datos en la tabla "campanias"
    cursor.execute('INSERT INTO campanias (TV, radio, newspaper, sales) VALUES (?, ?, ?, ?)',
                   (tv, radio, newspaper, sales))
    
    # Confirmar los cambios en la base de datos y cerrar la conexión
    conn.commit()
    
    # Devolver una respuesta indicando que los datos se han insertado correctamente
    return 'Los datos se han insertado correctamente.'


#3 Posibilidad de reentrenar de nuevo el modelo con los posibles nuevos registros que se recojan. (/v2/retrain)

@app.route('/v2/retrain model', methods=['GET'])
def retrain():
    # Conectarse a la base de datos
    conn = sqlite3.connect('data/advertising2.db')
    cursor = conn.cursor()

    # Cargar los datos de entrenamiento desde la base de datos
    training_data = pd.read_sql_query ('''
                               SELECT
                               *
                               FROM campanias
                               ''', conn)
  
    # Cargar el modelo entrenado desde un archivo con pickle
    model = pickle.load(open('data/advertising_model','rb'))

    # Preparar los datos de entrenamiento
    X_train = training_data.drop('sales', axis=1)
    y_train = training_data['sales']

    # Reentrenar el modelo
    model.fit(X_train, y_train)

    # Guardar el modelo reentrenado en un archivo con pickle
    now = datetime.datetime.now()
    formatted_date = now.strftime("%m-%d")
    model_name = 'modelo_actualizado' + formatted_date + '.pkl'
    with open(model_name, 'wb') as f:
        pickle.dump(model, f)


app.run()