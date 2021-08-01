from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import os
from image_captioning import encode_image,predict_caption
import matplotlib.pyplot as plt


app = Flask(__name__)
uploaded_file = ''

app.config['image_upload'] = os.path.join('static','upload image')
app.config['SEND_FILE_MAX_AGE_DEFAULT']=1

@app.route('/',methods = ['GET','POST'])
def main():
    return render_template('main.html')


@app.route('/after' , methods = ['POST'])
def after():

   if request.method == 'POST':
    uploaded_file = request.files['upload_img']
    uploaded_file.save(os.path.join(app.config['image_upload'],'file.jpg'))
    plt.style.use("seaborn")
    i = plt.imread("static/upload image/file.jpg")
    photo=encode_image("static/upload image/file.jpg").reshape((1,2048))
    caption = predict_caption(photo)
    return render_template('predict.html',caption=caption)


if __name__ == '__main__':
    app.run(debug=True,threaded=False)