import os
import numpy as np
import keras

from preprocess import preprocess
from featurizer import Featurizer
from featurizer import VOCAB_SIZE
from featurizer import SENTENCE_MAX_SIZE

EMBEDDED_DIM = 50
BATCH_SIZE = 50
TRAINING_EPOCHS = 3

class SentimentPredictor:
    def __init__(self):
        '''Class to predict whether a movie review is positive or negative.
           It is necessary to train a model first (.fit option) and then apply it to new samples.'''
    
        self.featurizer = Featurizer()
        self.model = None
    
    def fit(self, X_train, y_train):
        '''
        INPUTS
        X_train: list of movie reviews in string format
        y_train: list of labeled reviews with 1 for positives and 0 for negatives'''
        
        X_train = [preprocess(text) for text in X_train]
        
        self.featurizer.fit(X_train)
        X_train = self.featurizer.apply(X_train)
        y_train = np.array(y_train)
        
        self.model = self.model_architecture()
        self.model.fit(X_train, y_train,
                       batch_size = BATCH_SIZE, 
                       epochs = TRAINING_EPOCHS,
                       shuffle=True,
                       verbose = 1)    


    def predict(self, X_data):
        if self.model == None:
            print('You need to train the model first')
            return []
        
        X_data = [preprocess(text) for text in X_data]
        X_data = self.featurizer.apply(X_data)
        pred = self.model.predict(X_data)
        pred = [1 if pred >= 0.5 else 0 for pred in pred]
        return pred        
          
    def model_architecture(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(VOCAB_SIZE, EMBEDDED_DIM, input_length = SENTENCE_MAX_SIZE))
        model.add(keras.layers.Conv1D(filters=20, kernel_size=10, activation='relu'))
        model.add(keras.layers.MaxPooling1D(pool_size=2))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def save(self, savename = 'SentimentPredictor'):
        self.featurizer.save(savename + '-tokenizer.pickle')
        self.model.save(savename)
    
    def load(self, savename = 'SentimentPredictor'):
        token_save = savename + '-tokenizer.pickle'
        try:
            self.featurizer.load(token_save)
            self.model = keras.models.load_model(savename)
            return 'OK'
        except:
            print('Model not found, need to train it first')
            return 'FAIL'

