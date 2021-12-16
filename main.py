import argparse
from MReviews_Model import MReviews_Model
import numpy as np

def test():

    model = MReviews_Model()
    model.load('tests/MReviews_model')
    X_test = np.load('tests/X_test.npy', allow_pickle = True)
    y_test_pred = np.load('tests/y_test_pred.npy', allow_pickle = True)
    pred = model.predict(X_test)
    if (y_test_pred == np.array(pred)).all():
        print('\n\nTest saved model - passed\n\n')
    else:
        print('\n\nTest saved model - not passed\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--test', action = 'count', help='run test to check installation')
    args = vars(parser.parse_args())
    if args['test'] != None:
        test()





