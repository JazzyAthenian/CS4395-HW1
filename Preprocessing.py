import re
from nltk.stem import WordNetLemmatizer

def preprocessing(text):
    text = text.lower()
    text = re.sub('[^a-z0-9\.\-]', ' ', text) # remove special characters
    text = re.sub('\s{2,}', ' ', text) # change multiple spaces to 1 space
    text = '. ' + text

    wnl = WordNetLemmatizer()
    tokens = text.split()
    tokens = [wnl.lemmatize(token) for token in tokens]
    return tokens


if __name__ == '__main__':
    f = open("train.txt", "r")
    for line in f:
        preprocessing(line)