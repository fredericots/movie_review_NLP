import os
import keras

from preprocess import preprocess
from featurizer import Featurizer
from featurizer import VOCAB_SIZE
from featurizer import SENTENCE_MAX_SIZE

EMBEDDED_DIM = 50
BATCH_SIZE = 50
TRAINING_EPOCHS = 3

class MReviews_Model:
    def __init__(self):
        self.featurizer = Featurizer()
        self.model = None
        pass
    
    def fit(self, X_train, y_train):
        '''
        INPUTS
        X_train: list of movie reviews
        y_train: list of labeled reviews with 1 for positives and 0 otherwise'''
        
        X_train = [preprocess(text) for text in X_train]
        
        self.featurizer.fit(X_train)
        X_train = self.featurizer.apply(X_train)
        
        self.model = self.model_architecture()
        self.model.fit(X_train, y_train,
                       batch_size = BATCH_SIZE, 
                       epochs = TRAINING_EPOCHS, verbose = 1)    


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

    def save(self, savename = 'MReviews_model'):
        self.featurizer.save(savename + '-tokenizer.pickle')
        self.model.save(savename)
    
    def load(self, savename = 'MReviews_model'):
        self.featurizer.load(savename + '-tokenizer.pickle')
        self.model = keras.models.load_model(savename)

