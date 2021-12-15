import glob
import tqdm
import pandas as pd

DATA_IMDB_PATH = "../aclImdb/"

DATA_TRAIN_POS_PATH = DATA_IMDB_PATH + "train/pos/*"
DATA_TRAIN_NEG_PATH = DATA_IMDB_PATH + "train/neg/*"
DATA_TEST_POS_PATH = DATA_IMDB_PATH + "test/pos/*"
DATA_TEST_NEG_PATH = DATA_IMDB_PATH + "test/neg/*"

def read_train_data():
    data = []
    print('Positive data collection progress')
    for file in tqdm.tqdm(glob.glob(DATA_TRAIN_POS_PATH)):
        text = open(file, encoding="utf8").read()
        data.append([text, 'pos'])

    print('Negative data collection progress')        
    for file in tqdm.tqdm(glob.glob(DATA_TRAIN_NEG_PATH)):
        text = open(file, encoding="utf8").read()
        data.append([text, 'neg'])
    
    return pd.DataFrame(data, columns = ['movie_review', 'label'])

def read_test_data():
    data = []
    print('Positive data collection progress')
    for file in tqdm.tqdm(glob.glob(DATA_TEST_POS_PATH)):
        text = open(file, encoding="utf8").read()
        data.append([text, 'pos'])

    print('Negative data collection progress')        
    for file in tqdm.tqdm(glob.glob(DATA_TEST_NEG_PATH)):
        text = open(file, encoding="utf8").read()
        data.append([text, 'neg'])
    
    return pd.DataFrame(data, columns = ['movie_reviews', 'label'])