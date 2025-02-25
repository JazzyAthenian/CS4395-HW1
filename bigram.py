import re

def preprocessing(text):
    text = text.lower()
    text = re.sub('[^a-z0-9\.\-]', ' ', text)
    text = re.sub('\s{2,}', ' ', text)
    text = '. ' + text
    #print(text)
    return text
def compute_bigram(text):
    bigrams = dict()
    tokens = text.split()
    #print(tokens)

    for i in range(len(tokens)-2):
        word1 = tokens[i]
        word2 = tokens[i+1]

        if word1 not in bigrams:
            bigrams[word1] = dict()

        if word2 not in bigrams[word1]:
            bigrams[word1][word2] = 1
        else:
            bigrams[word1][word2] += 1

    #print(bigrams)
    print(bigrams['.'])
    print(bigrams['i'])

# def use_bigram(fileName):
#    f = open(fileName, "r")
#    for line in f:
#        features = compute_bigram(line)
#        # train set here

if __name__ == '__main__':
    f = open("train.txt", "r")
    text = ""
    for line in f:
        text = text + preprocessing(line) + '\n'

    compute_bigram(text)
