import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import re
import nltk
from nltk.corpus import stopwords
import string
import json
from time import time
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add



word_to_idx = None
with open("word_to_idx.txt",'r') as f:
    word_to_idx= f.read()
json_acceptable_string = word_to_idx.replace("'","\"")
word_to_idx = json.loads(json_acceptable_string)

idx_to_word = None
with open("idx_to_word.txt",'r') as f:
    idx_to_word= f.read()
json_acceptable_string = idx_to_word.replace("'","\"")
idx_to_word = json.loads(json_acceptable_string)


train_descriptions = None
with open("train_descriptions.txt",'r') as f:
    train_descriptions= f.read()
json_acceptable_string = train_descriptions.replace("'","\"")
train_descriptions = json.loads(json_acceptable_string)

descriptions = None
with open("descriptions_1.txt",'r') as f:
    descriptions= f.read()
json_acceptable_string = descriptions.replace("'","\"")
descriptions = json.loads(json_acceptable_string)


def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    # Normalisation
    img = preprocess_input(img)
    return img

def encode_image(img):
    img = preprocess_img(img)
    model_new = load_model('encoding_model.h5')
    feature_vector = model_new.predict(img)
    
    feature_vector = feature_vector.reshape((-1,))
    #print(feature_vector.shape)
    return feature_vector

def length():
    max_len = 0 
    for key in train_descriptions.keys():
        for cap in train_descriptions[key]:
            max_len = max(max_len,len(cap.split()))
    return max_len
        


def predict_caption(photo):
    model = load_model('model_9.h5')
    in_text = "startseq"
    for i in range(length()):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen=length(),padding='post')
        
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax() #WOrd with max prob always - Greedy Sampling
        word = idx_to_word[str(ypred)]
        in_text += (' ' + word)
        
        if word == "endseq":
            break
    
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption







# plt.style.use("seaborn")
# i = plt.imread("myphoto.jpg")
# photo=encode_image("myphoto.jpg").reshape((1,2048))
# caption = predict_caption(photo)
# plt.title(caption)
# plt.imshow(i)
# plt.axis("off")
# plt.show()

