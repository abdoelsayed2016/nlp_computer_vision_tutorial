import numpy as np
import pandas as pd
import re
from gensim.models import Word2Vec,Phrases

from tensorflow import keras 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Embedding,Dropout,Conv1D,MaxPool1D,GRU,LSTM,Dense,Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve
def build_model(embedding_martix:np.ndarray, input_length: int, use_lstm:bool):
    model = Sequential()
    model.add(Embedding(
        input_dim=embedding_martix.shape[0],
        output_dim=embedding_martix.shape[1],
        input_length=input_length,
        weights= [embedding_martix],
        trainable=False
    ))
    if use_lstm:
        model.add(Bidirectional(LSTM(128,recurrent_dropout=0.1)))
    else:
        model.add(Bidirectional(GRU(128,recurrent_dropout=0.1)))
    
    model.add(Dropout(0.25))
    model.add(Dense(64))
    model.add(Dropout(0.3))
    model.add(Dense(1,activation='sigmoid'))
    model.summary()
    return model