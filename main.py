from flask import Flask, render_template, session, redirect, url_for
from flask import request, jsonify
import numpy as np
import pickle
from tensorflow import keras
from FlowerForm import FlowerForm
import os


#---------  prediction function --------------
def return_prediction(model, scaler, sample_json):
    sepal_length = sample_json['sepal_length']
    sepal_width = sample_json['sepal_width']
    petal_length = sample_json['petal_length']
    petal_width = sample_json['petal_width']
    
    # Define flower as a list of lists
    flower = [[sepal_length, sepal_width, petal_length, petal_width]]
    
    # scaling
    flower = scaler.transform(flower)

    # model prediction
    class_index = np.argmax(model.predict(flower), axis=-1)[0]

    # return the class index and class name
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    return classes[class_index]


#########  Flask App #####################

# Crea una instancia de Flask
app = Flask(__name__)
# Configura la llave secreta
# La llave secreta es necesaria para poder utilizar la variable `session` en Flask.
# app.config es un diccionario que contiene las configuraciones de la aplicación Flask.
# Config tiene las siguientes keys: SECRET_KEY, DEBUG, TESTING, PROPAGATE_EXCEPTIONS, PRESERVE_CONTEXT_ON_EXCEPTION, PERMANENT_SESSION_LIFETIME, USE_X_SENDFILE, SERVER_NAME, APPLICATION_ROOT, SESSION_COOKIE_NAME, SESSION_COOKIE_DOMAIN, SESSION_COOKIE_PATH, SESSION_COOKIE_HTTPONLY, SESSION_COOKIE_SECURE, SESSION_REFRESH_EACH_REQUEST, MAX_CONTENT_LENGTH, SEND_FILE_MAX_AGE_DEFAULT, TRAP_BAD_REQUEST_ERRORS, TRAP_HTTP_EXCEPTIONS, PREFERRED_URL_SCHEME, JSON_AS_ASCII, JSON_SORT_KEYS, JSONIFY_PRETTYPRINT_REGULAR, JSONIFY_MIMETYPE, TEMPLATES_AUTO_RELOAD, EXPLAIN_TEMPLATE_LOADING, MAX_COOKIE_SIZE, BUNDLE_ERRORS, PREFERRED_URL_SCHEME y many more.
app.config['SECRET_KEY'] = 'mysecretkey'


@app.route('/', methods=['GET', 'POST'])
def index():
    # Crea una instancia de FlowerForm
    # FlowerForm es una clase que hereda de FlaskForm
    form = FlowerForm()
    
    # Si el formulario es enviado y es valido
    if form.validate_on_submit():
        
        # Comprueba si se han enviado datos
        if form.sepal_length.data and form.sepal_width.data and form.petal_length.data and form.petal_width.data:
            # La variable `session` es una variable especial en Flask que se utiliza para almacenar datos del usuario en una sesión entre solicitudes HTTP. 
            # Esta variable está disponible en el contexto de la aplicación de Flask y no necesita ser declarada explícitamente.

            # Cuando un usuario accede a tu aplicación web, se le asigna una sesión única que se utiliza para almacenar información específica del usuario. 
            # Esta información se puede acceder y actualizar en diferentes solicitudes HTTP durante la sesión del usuario.
            # Esto permitirá que el valor se pueda recuperar en solicitudes posteriores y se pueda utilizar para diferentes propósitos en la aplicación.

            # Ten en cuenta que para utilizar la variable `session` en Flask, primero debes haber habilitado el soporte de sesiones en tu aplicación. 
            # Esto se puede hacer mediante la configuración de una clave `SECRET_KEY` en tu aplicación Flask.
            session['sepal_length'] = form.sepal_length.data
            session['sepal_width'] = form.sepal_width.data
            session['petal_length'] = form.petal_length.data
            session['petal_width'] = form.petal_width.data
            
            # Redirecciona a la pagina de prediccion
            # redirect() es una función que redirige a la URL dada.
            # url_for() es una función que genera la URL para una función dada.
            return redirect(url_for('prediction'))

    # Renderiza la plantilla home.html con el formulario
    # render_template() es una función que se utiliza para renderizar una plantilla con un contexto.
    return render_template('home.html', form=form)

# Carga el modelo y el scaler
# joblib es una biblioteca de Python que proporciona herramientas para serializar y deserializar objetos de Python.
# serializar es el proceso de convertir un objeto en una secuencia de bytes para almacenarlo o transmitirlo a la memoria, un archivo o una base de datos.
# joblib.load() es una función que carga un archivo serializado.
iris_model = pickle.load(open('final_iris_model.pkl', 'rb'))
iris_scaler = pickle.load(open('iris_scaler.pkl', 'rb'))


@app.route('/prediction')
def prediction():
    content = {}
    
    # Obtenemos los datos de la sesion
    content['sepal_length'] = float(session['sepal_length'])
    content['sepal_width'] = float(session['sepal_width'])
    content['petal_length'] = float(session['petal_length'])
    content['petal_width'] = float(session['petal_width'])
    
    # Obtenemos la prediccion
    results = return_prediction(iris_model, iris_scaler, content)
    
    return render_template('prediction.html', results=results)


@app.route('/api/iris', methods=['POST'])

def predict_flower():
    # request.json es un atributo de la clase request que devuelve un diccionario con los datos enviados en la solicitud.
    # Estos datos pueden ser enviados en formato JSON o en formato de formulario
    print(request.json)
    content = request.json
    results = return_prediction(iris_model, iris_scaler, content)
    return jsonify(results)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
    
