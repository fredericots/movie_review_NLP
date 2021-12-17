# movie_review_NLP

Movie review NLP is a software to predict if a movie review is positive or negative with
just its text (only english is supported).

## How to install

[IN PROGRESS] pip install requirements.txt

## How to use

After installation you can have available options typing:

python movie_review_nlp.py -h

Here we have:
  -h, --help         show this help message and exit
  --test             Run test to check installation
  --aclImdb_train    Train a model using aclImdb data - see data_collection.py
                     for details
  --aclImdb_eval     Evaluation results with the trained model and aclImdb
                     test data.
  --predict PREDICT  Predict if a text in english is a positive review or a
                     negative one. Needs the .txt path with the text to be
                     predicted

## About the reference data

The aclImdb data cited here can be found in the following website:

http://ai.stanford.edu/~amaas/data/sentiment/

It is assumed that the data will be in one directory above.
aclImdb data path is set inside data_collection.py in the following variable:

DATA_IMDB_PATH = "../aclImdb/"

## Experiments

All developed models and tests are present in the experiments.ipynb notebook.
