from string import punctuation
import nltk
from nltk.corpus import stopwords

STOP_WORDS = stopwords.words('english')

# TODO - add spelling correction
#from autocorrect import Speller 
#SPELLER = Speller('en')

def preprocess(text):
    text = text.lower()
    text = ''.join(c for c in text if c not in punctuation)
    text = ''.join(c for c in text if not c.isdigit())
    text_words = nltk.word_tokenize(text)
    #text_words = [SPELLER(w) for w in text_words] # TODO - add spelling correction
    text_words = [w for w in text_words if not w in STOP_WORDS]
    text = ' '.join(text_words)
    return text