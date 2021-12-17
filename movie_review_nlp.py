import os
import argparse
import numpy as np

from sklearn import metrics
from data_collection import read_aclImdb
from SentimentPredictor import SentimentPredictor

def test():

    model = SentimentPredictor()
    model.load('tests/SentimentPredictor')
    X_test = np.load('tests/X_test.npy', allow_pickle = True)
    y_test_pred = np.load('tests/y_test_pred.npy', allow_pickle = True)
    pred = model.predict(X_test)
    if (y_test_pred == np.array(pred)).all():
        print('Test saved model - passed')
    else:
        print('Test saved model - not passed')

def aclImdb_train():

    X_train, y_train = read_aclImdb('train')
    model = SentimentPredictor()
    model.fit(X_train, y_train)
    model.save()

def aclImdb_eval():
    X_test, y_test = read_aclImdb('test')
    model = SentimentPredictor()
    model.load()
    predictions = model.predict(X_test)
    print('Accuracy on test: ', metrics.accuracy_score(y_test, predictions))

def predict(file_name):
    if os.path.splitext(file_name)[-1] != '.txt':
        print('Need a .txt file for prediction')
    else:
        text = open(file_name, encoding="utf8").read()
        model = SentimentPredictor()
        model_flag = model.load()
        if model_flag != 'OK':
            pass
        else:
            pred = model.predict([text])
            if pred[0] == 1:
                print('Positive - flag: ', 1)
            else:
                print('Negative - flag: ', -1)


if __name__ == "__main__":
    print('\n\n MOVIE REVIEW NLP - A SOFTWARE TO PREDICT IF A REVIEW IS POSITIVE OR NEGATIVE\n\n')
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test', action = 'count', 
        help='Run test to check installation')
    parser.add_argument('--aclImdb_train', action = 'count', 
        help='Train a model using aclImdb data - see data_collection.py for details')
    parser.add_argument('--aclImdb_eval', action = 'count', 
        help='Evaluation results with the trained model and aclImdb test data.')
    parser.add_argument('--predict',
        help='Predict if a text in english is a positive review or a negative one. \
              Needs the .txt path with the text to be predicted')
   
    args = vars(parser.parse_args())
    if args['test'] != None:
        print('OPTION CHOSEN: test')
        test()
    elif args['aclImdb_train'] != None:
        print('OPTION CHOSEN: aclImdb_train')
        aclImdb_train()
    elif args['aclImdb_eval'] != None:
        print('OPTION CHOSEN: aclImdb_eval')
        aclImdb_eval()
    elif args['predict'] != None:
        print('OPTION CHOSEN: predict')
        predict(args['predict'])
    else:
        print('Option not found - type -h for help')


