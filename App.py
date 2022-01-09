import keras as keras
import numpy as np
from flask import Flask, flash, request, redirect, url_for, render_template
from keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
app = Flask(__name__)
UPLOAD_FOLDER = 'Uploaded_Paintings'
UPLOADED = False
ROOT_DIRECTORY = r"C:/Users/vanto/Documents/SCHOOL(AI-2)/Deep_Learning/Taak_Paintings"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
PAINTERS = ["Mondriaan", "Rubens", "Picasso", "Rembrandt"]


@app.route('/')
def index():
    os.chdir(ROOT_DIRECTORY)
    return render_template('index.html')


@app.route('/upload', methods=["GET", "POST"])
def upload():
    os.chdir(ROOT_DIRECTORY)
    os.chdir(UPLOAD_FOLDER)
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        predicted_painter = predict_painter()
        return render_template('index.html', predicted_painter=predicted_painter, success=True)


def predict_painter():
    os.chdir(ROOT_DIRECTORY)
    reconstructed_model = keras.models.load_model("DefinitiveModel.h5")
    os.chdir(UPLOAD_FOLDER)
    img = load_img(os.listdir(os.getcwd())[0])
    img = img.resize((621, 655))
    img = img_to_array(img)
    img = img.reshape(-1, 621, 655, 3)
    predicted_painter = PAINTERS[np.where(np.ravel(reconstructed_model.predict(img)) == 1)[0][0]]
    os.chdir(os.pardir)
    [f.unlink() for f in Path(UPLOAD_FOLDER).glob("*") if f.is_file()]
    return predicted_painter
