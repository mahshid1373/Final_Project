import re

# Import the string dictionary that we'll use to remove punctuation
import string 
import nltk
from nltk.corpus import stopwords


def clean_text_syntax(text):
    """
    Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.
    """

    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def text_preprocessing(text):
    """
    Cleaning and parsing the text.
    """

    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    nopunc = clean_text_syntax(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    #remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    combined_text = ' '.join(tokenized_text)
    return combined_text