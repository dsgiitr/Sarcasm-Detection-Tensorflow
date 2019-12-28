# -*- coding: utf-8 -*-

#Importing Libraries

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from keras.models import Model,Sequential
from keras.layers import LSTM,Activation,Dense,Dropout,Input,Embedding,SpatialDropout1D
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#Concatenating the two Datasets
df1 = pd.read_json('Sarcasm_Headlines_Dataset.json',lines=True)
df2 = df = pd.read_json('Sarcasm_Headlines_Dataset_v2.json',lines=True)
frames = [df1,df2]
df = pd.concat(frames)


#Tokenize to vectorize and convert texts into features
for idx,row in df.iterrows():
    row[0] = row[0].replace('rt',' ')

max_features = 2500
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['headline'].values)
X = tokenizer.texts_to_sequences(df['headline'].values)
X = pad_sequences(X)
Y = pd.get_dummies(df['is_sarcastic']).values

#Splitting into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)

#Building the model

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(lstm_out, dropout=0.1, recurrent_dropout=0.1))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

batch_size = 32 #You may change batch size

#Fitting the model on the training data
model_fit = model.fit(X_train, Y_train, epochs = 20, batch_size=batch_size)


#Evaluating the model score and accuracy on the test set
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)

#Print score and Accuracy
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))
