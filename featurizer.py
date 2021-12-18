import pickle
import numpy as np
from keras import preprocessing

VOCAB_SIZE = 2000
SENTENCE_MAX_SIZE = 1000

class Featurizer:
    
    def __init__(self):
        self.tokenizer = preprocessing.text.Tokenizer(
            num_words=VOCAB_SIZE,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True, 
            split=' ')
        
    def fit(self, X_train):
        self.tokenizer.fit_on_texts(X_train)
    
    def apply(self, data):
        feat_data = self.tokenizer.texts_to_sequences(data)
        feat_data = preprocessing.sequence.pad_sequences(feat_data, maxlen=SENTENCE_MAX_SIZE, dtype='int32', 
                                                        padding='post', truncating='post', value=0)
        return np.array(feat_data) 
        
    def save(self, tokenizer_path = 'tokenizer.pickle'):
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, tokenizer_path = 'tokenizer.pickle'):
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def y_format(self, y_list):
        return [1 if label == 'pos' else 0 for label in y_list]
