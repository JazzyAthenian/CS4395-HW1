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


def compute_bigram(fileName):
    f = open(fileName, "r")
    bigrams = dict()

    # going through training data one review at a time
    for line in f:
        tokens = preprocessing(line)

        # filling out dictionary
        for i in range(len(tokens)-1):
            word1 = tokens[i]
            word2 = tokens[i+1]

            if word1 not in bigrams:
                bigrams[word1] = dict()

            if word2 not in bigrams[word1]:
                bigrams[word1][word2] = 1
            else:
                bigrams[word1][word2] += 1

    # sorting and calculating probabilities
    for word in bigrams:
        bigrams[word] = dict(sorted(bigrams[word].items(), key=lambda item: -item[1]))
        total = sum(bigrams[word].values())
        bigrams[word] = dict([(k, (bigrams[word][k] + .2)/(total + ( 0.2 * len(bigrams)))) for k in bigrams[word]])
    return bigrams
    f.close()

def processing_test(bigram, fileName):
    f = open(fileName, "r")
    test_data = []
    
    for line in f:
        tokens = preprocessing(line)

        for i in range(len(tokens)):
            word = tokens[i]
            
            if word not in bigram:
                test_data.append("<UNK>")
            else: 
                test_data.append(word)
            
    return test_data

if __name__ == '__main__':
    bigrams = compute_bigram("train.txt")
    processing_test(bigrams, "val.txt")
