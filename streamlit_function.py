from function import *

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
