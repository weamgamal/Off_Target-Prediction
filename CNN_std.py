# -*- coding: utf-8 -*-
"""
Created on Sun May 23 18:19:55 2021

@author: 10
"""

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization
from keras.models import Model
from keras.models import model_from_yaml
import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn import metrics

#np.random.seed(5)
#from tensorflow import set_random_seed
#set_random_seed(12)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def cnn_model(X_train, X_test, y_train, y_test):

    # X_train, y_train = load_data()

    inputs = Input(shape=(1, 23, 4), name='main_input')
    conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(inputs)
    conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(inputs)
    conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(inputs)
    conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(inputs)

    conv_output = keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

    bn_output = BatchNormalization()(conv_output)

    pooling_output = keras.layers.MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

    flatten_output = Flatten()(pooling_output)

    x = Dense(100, activation='relu')(flatten_output)
    x = Dense(23, activation='relu')(x)
    x = keras.layers.Dropout(rate=0.15)(x)

    prediction = Dense(2, name='main_output')(x)

    model = Model(inputs, prediction)

    adam_opt = keras.optimizers.adam(lr = 0.0001)

    model.compile(loss='binary_crossentropy', optimizer = adam_opt)
    
    model.fit(X_train, y_train, batch_size=100, epochs=200, shuffle=True)

    # later...
    # X_test, y_test = load_crispor_data()
    y_pred = model.predict(X_test).flatten()
    model.summary()

def cnn_predict(guide_seq, off_seq):

    code_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    gRNA_list = list(guide_seq)
    off_list = list(off_seq)
    #print(len(gRNA_list))
    if len(gRNA_list) != len(off_list):
        print("the length of sgRNA and DNA are not matched!")
        return 0
    pair_code = []

    for i in range(len(gRNA_list)):
        if gRNA_list[i] == 'N':
            gRNA_list[i] = off_list[i]
        gRNA_base_code = code_dict[gRNA_list[i]]
        DNA_based_code = code_dict[off_list[i]]
        pair_code.append(list(np.bitwise_or(gRNA_base_code, DNA_based_code)))
        
    #print("55555555555555555555555555555")
    input_code = np.array(pair_code).reshape(1, 1, 23, 4)

    # load YAML and create model
    yaml_file = open('CNN_std_model/model_cnn_v1.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    # load weights into new model
    loaded_model.load_weights("CNN_std_model/model_cnn_v1.h5")
   #print("Loaded model from disk")

    y_pred = loaded_model.predict(input_code).flatten()
    #print(y_pred)
    return y_pred
    
s=['A','T','T','A','G','C','A','T','T','A','G','C','A','T','T','A','G','C','C','A','G','G','G']
guide_seq=''.join(s)

s2=['C','C','A','A','A','G','A','T','T','A','G','C','A','T','T','A','G','C','C','A','G','G','G']
off_seq=''.join(s2)

C=cnn_predict(guide_seq,off_seq)
#print(C)

#________________________________________________________________________________

def loadData(inputpath):
   
    with open(inputpath) as f:
        sgRNA_item=[]
        DNA_item=[]
        for line in f:
            ll = [i for i in line.strip().split(',')]
            sgRNA_item.append(ll[0])
            DNA_item.append(ll[1])
            #print(DNA_item)
    return DNA_item,sgRNA_item

# glove_inputpath = "output\keras_GloVeVec_5_100_10000.csv"
# hek_inputpath = "output\hek293_off_Glove.txt"
# K562_inputpath = "output\K562_off_Glove.txt"
input="Model_save\input_example.txt"
#input="output\hek293_off_Glove.txt"
# DNA_item,sgRNA_item = loadData(hek_inputpath)
DNA_item,sgRNA_item = loadData(input)

#print(DNA_item)
#print("####################################")
#print(sgRNA_item)
#print("####################################")



scor=[]
f=0 
while f<len(sgRNA_item):
    off =str(DNA_item[f])
    scor.append(off)
    #print(off)
    wt = str(sgRNA_item[f])
    scor.append(wt)
    #print(wt)
    score = cnn_predict(wt,off)
    #print(score)
    scor.append(score)
    scor.append("_____________________________________")
    f=f+1
def retu():
    return scor