import glob
import tqdm
import pandas as pd

DATA_IMDB_PATH = "aclImdb/"

DATA_TRAIN_POS_PATH = DATA_IMDB_PATH + "train/pos/*"
DATA_TRAIN_NEG_PATH = DATA_IMDB_PATH + "train/neg/*"
DATA_TEST_POS_PATH = DATA_IMDB_PATH + "test/pos/*"
DATA_TEST_NEG_PATH = DATA_IMDB_PATH + "test/neg/*"

def read_aclImdb(mode = 'train'):


    if mode == 'train':
        path_pos = DATA_TRAIN_POS_PATH
        path_neg = DATA_TRAIN_NEG_PATH
    elif mode == 'test':
        path_pos = DATA_TEST_POS_PATH
        path_neg = DATA_TEST_NEG_PATH
    else:
        raise Exception("mode " + str(mode) + "not supported")
    
    return read_aclImdb_folders(path_pos, path_neg)
    
def read_aclImdb_folders(path_pos, path_neg):   

    text = []
    target = []
    print('Positive data collection progress')
    for file in tqdm.tqdm(glob.glob(path_pos)):
        t = open(file, encoding="utf8").read()
        text.append(t)
        target.append(1)

    print('Negative data collection progress')        
    for file in tqdm.tqdm(glob.glob(path_neg)):
        t = open(file, encoding="utf8").read()
        text.append(t)
        target.append(0)

    return text, target    
