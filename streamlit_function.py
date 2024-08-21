
import nltk
import re
# Import the string dictionary that we'll use to remove punctuation
import string 

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


def clean_data(data):
    cleaned = data[["text", "sentiment"]]
    cleaned = cleaned.dropna(axis = 0, how ='any') 
    cleaned['text'] = data['text'].apply(str).apply(lambda x: text_preprocessing(x))

    return cleaned

# function to color the sentiment column
def sentiment_color(sentiment):
    if sentiment == "Positive":
        return "background-color: #1F77B4; color: white"
    else:
        return "background-color: #FF7F0E"
